"""Benchmark GraphRAG's global and local search against a plain vector baseline.

For every question we run up to three retrieval modes over the same index, then ask an
LLM judge to compare the graph answer with the vector-baseline answer head to head.
Latency and token cost come from GraphRAG's own per-call metrics, not estimates.

Usage
-----
    python eval/benchmark.py --dry-run          # show the plan and the call count, spend nothing
    python eval/benchmark.py --limit 3          # quick smoke test
    python eval/benchmark.py                    # full run over every question
    python eval/benchmark.py --no-judge         # latency/cost only, no judging

Results land in eval/results/ as JSON (raw) and Markdown (summary).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The judge talks to an OpenAI-compatible chat endpoint. Gemini exposes one, so the same
# free-tier key that built the index can grade the answers.
JUDGE_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": None,
}
GRAPH_MODE_FOR = {"thematic": "global", "factual": "local"}
BASELINE_MODE = "basic"
CRITERIA = {
    "thematic": ["comprehensiveness", "diversity", "empowerment"],
    "factual": ["correctness", "directness"],
}

JUDGE_SYSTEM = """You are grading two answers to the same question about ISRO's space programme.
Judge only what is in the answers. Do not reward length for its own sake.

Criteria:
- comprehensiveness: how much of the relevant ground the answer covers.
- diversity: how varied the perspectives and supporting details are.
- empowerment: how well it helps a reader understand and reason about the topic.
- correctness: whether the specific claims are accurate and on-topic.
- directness: whether it answers the exact question without padding.

Reply with JSON only: {"<criterion>": {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}
Include exactly the criteria you are asked for."""


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
def load_dotenv(root: Path) -> None:
    """Minimal .env reader so the script works the same way GraphRAG's CLI does."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def default_judge_model(root: Path, provider: str) -> str:
    """Gemini: reuse the chat model from settings.yaml. OpenAI: gpt-4o-mini."""
    if provider == "openai":
        return "gpt-4o-mini"
    settings = yaml.safe_load((root / "settings.yaml").read_text(encoding="utf-8"))
    return settings["completion_models"]["default_completion_model"]["model"]


def load_index(root: Path) -> dict[str, pd.DataFrame]:
    out = root / "output"
    names = ["entities", "communities", "community_reports", "text_units", "relationships"]
    missing = [n for n in names if not (out / f"{n}.parquet").exists()]
    if missing:
        msg = f"index incomplete in {out} (missing: {', '.join(missing)}). Run `graphrag index --root .` first."
        raise SystemExit(msg)
    return {n: pd.read_parquet(out / f"{n}.parquet") for n in names}


# --------------------------------------------------------------------------------------
# Token / cost accounting from GraphRAG's own metrics files
# --------------------------------------------------------------------------------------
def metrics_snapshot(metrics_dir: Path) -> dict[str, int]:
    if not metrics_dir.exists():
        return {}
    return {
        p.name: sum(1 for _ in p.open(encoding="utf-8"))
        for p in metrics_dir.glob("*.jsonl")
    }


def metrics_delta(metrics_dir: Path, before: dict[str, int]) -> tuple[int, int, float]:
    """Return (llm_calls, total_tokens, total_cost) recorded since `before`."""
    calls = tokens = 0
    cost = 0.0
    if not metrics_dir.exists():
        return 0, 0, 0.0
    for path in metrics_dir.glob("*.jsonl"):
        start = before.get(path.name, 0)
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < start or not line.strip():
                    continue
                try:
                    metrics = json.loads(line).get("metrics", {})
                except json.JSONDecodeError:
                    continue
                calls += 1
                tokens += int(metrics.get("total_tokens", 0) or 0)
                cost += float(metrics.get("total_cost", 0) or 0)
    return calls, tokens, cost


# --------------------------------------------------------------------------------------
# Running one query
# --------------------------------------------------------------------------------------
@dataclass
class Run:
    question: str
    kind: str
    mode: str
    answer: str
    latency_s: float
    llm_calls: int
    total_tokens: int
    total_cost: float
    error: str = ""


def run_query(mode: str, question: str, config, tables, community_level: int, response_type: str) -> tuple[str, str]:
    """Execute one search. Returns (answer, error)."""
    import graphrag.api as api

    try:
        if mode == "global":
            response, _ = asyncio.run(api.global_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                community_level=community_level,
                dynamic_community_selection=False,
                response_type=response_type,
                query=question,
            ))
        elif mode == "local":
            response, _ = asyncio.run(api.local_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                text_units=tables["text_units"],
                relationships=tables["relationships"],
                covariates=None,
                community_level=community_level,
                response_type=response_type,
                query=question,
            ))
        else:
            response, _ = asyncio.run(api.basic_search(
                config=config,
                text_units=tables["text_units"],
                response_type=response_type,
                query=question,
            ))
        return str(response), ""
    except Exception as exc:  # noqa: BLE001 - recorded per run, the sweep continues
        return "", f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------------------
@dataclass
class Judgement:
    question: str
    kind: str
    graph_mode: str
    verdicts: dict = field(default_factory=dict)  # criterion -> {"winner": "graph"|"baseline"|"tie", "reason": str}
    error: str = ""


def judge_pair(client, model: str, question: str, kind: str, graph_answer: str, baseline_answer: str) -> Judgement:
    """Compare graph vs baseline with the A/B order randomised to blunt position bias."""
    graph_is_a = random.random() < 0.5
    answer_a, answer_b = (graph_answer, baseline_answer) if graph_is_a else (baseline_answer, graph_answer)
    criteria = CRITERIA[kind]

    prompt = (
        f"Question: {question}\n\n"
        f"Criteria to judge: {', '.join(criteria)}\n\n"
        f"--- Answer A ---\n{answer_a}\n\n--- Answer B ---\n{answer_b}"
    )
    judgement = Judgement(question=question, kind=kind, graph_mode=GRAPH_MODE_FOR[kind])
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = json.loads(completion.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        judgement.error = f"{type(exc).__name__}: {exc}"
        return judgement

    for criterion in criteria:
        verdict = raw.get(criterion) or {}
        winner_letter = str(verdict.get("winner", "tie")).strip().upper()
        if winner_letter == "A":
            winner = "graph" if graph_is_a else "baseline"
        elif winner_letter == "B":
            winner = "baseline" if graph_is_a else "graph"
        else:
            winner = "tie"
        judgement.verdicts[criterion] = {"winner": winner, "reason": verdict.get("reason", "")}
    return judgement


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def summarise(runs: list[Run], judgements: list[Judgement]) -> str:
    lines = ["# GRAIL benchmark", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]

    lines += ["## Cost and latency per query", "",
              "| mode | queries | mean latency | mean LLM calls | mean tokens | mean cost |",
              "|---|---|---|---|---|---|"]
    for mode in ["global", "local", "basic"]:
        subset = [r for r in runs if r.mode == mode and not r.error]
        if not subset:
            continue
        lines.append(
            f"| {mode} | {len(subset)} | {statistics.mean(r.latency_s for r in subset):.1f}s "
            f"| {statistics.mean(r.llm_calls for r in subset):.1f} "
            f"| {statistics.mean(r.total_tokens for r in subset):,.0f} "
            f"| ${statistics.mean(r.total_cost for r in subset):.4f} |"
        )

    failures = [r for r in runs if r.error]
    if failures:
        lines += ["", f"{len(failures)} run(s) failed:", ""]
        lines += [f"- `{r.mode}` / {r.question[:60]}... — {r.error}" for r in failures[:10]]

    if judgements:
        lines += ["", "## Answer quality vs the vector baseline", "",
                  "Win rates are the graph mode's share of head-to-head verdicts, ties counted separately.", "",
                  "| question type | criterion | graph wins | baseline wins | ties |", "|---|---|---|---|---|"]
        for kind in ["thematic", "factual"]:
            for criterion in CRITERIA[kind]:
                verdicts = [j.verdicts.get(criterion, {}).get("winner")
                            for j in judgements if j.kind == kind and not j.error]
                verdicts = [v for v in verdicts if v]
                if not verdicts:
                    continue
                graph = verdicts.count("graph")
                baseline = verdicts.count("baseline")
                ties = verdicts.count("tie")
                total = len(verdicts)
                lines.append(
                    f"| {kind} ({GRAPH_MODE_FOR[kind]}) | {criterion} | "
                    f"{graph}/{total} ({graph / total:.0%}) | {baseline}/{total} ({baseline / total:.0%}) | {ties} |"
                )

    total_cost = sum(r.total_cost for r in runs)
    lines += ["", f"Total measured spend for the search runs: ${total_cost:.3f} "
                  f"(judge calls are billed separately and not counted here).", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="project root (holds settings.yaml)")
    parser.add_argument("--limit", type=int, default=None, help="questions per type (default: all)")
    parser.add_argument("--modes", default="global,local,basic", help="comma-separated modes to run")
    parser.add_argument("--community-level", type=int, default=2)
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--judge-provider", choices=sorted(JUDGE_BASE_URLS), default="gemini",
                        help="API the judge calls (key: JUDGE_API_KEY, falling back to GRAPHRAG_API_KEY)")
    parser.add_argument("--judge-model", default=None,
                        help="default: the chat model in settings.yaml (gemini) or gpt-4o-mini (openai)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge comparison")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without calling any model")
    parser.add_argument("--seed", type=int, default=7, help="seed for A/B order randomisation")
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(args.root).resolve()
    load_dotenv(root)
    judge_model = args.judge_model or default_judge_model(root, args.judge_provider)

    questions = yaml.safe_load((Path(__file__).parent / "questions.yaml").read_text(encoding="utf-8"))
    for kind in questions:
        if args.limit:
            questions[kind] = questions[kind][: args.limit]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    n_questions = sum(len(v) for v in questions.values())
    n_searches = n_questions * len(modes)
    n_judge = 0 if args.no_judge else n_questions

    print(f"Index root : {root}")
    print(f"Questions  : {n_questions} ({', '.join(f'{k}={len(v)}' for k, v in questions.items())})")
    print(f"Modes      : {', '.join(modes)}")
    print(f"Plan       : {n_searches} searches + {n_judge} judge calls")
    if n_judge:
        print(f"Judge      : {args.judge_provider} / {judge_model}")
    if args.dry_run:
        print("\nDry run - nothing was called.")
        return 0

    if not os.environ.get("GRAPHRAG_API_KEY"):
        print("GRAPHRAG_API_KEY is not set (put it in .env).", file=sys.stderr)
        return 1

    from graphrag.config.load_config import load_config

    config = load_config(root_dir=root)
    tables = load_index(root)
    metrics_dir = root / "metrics"

    runs: list[Run] = []
    answers: dict[tuple[str, str], str] = {}

    for kind, items in questions.items():
        for i, question in enumerate(items, 1):
            for mode in modes:
                print(f"[{kind} {i}/{len(items)}] {mode:<7} {question[:58]}")
                before = metrics_snapshot(metrics_dir)
                started = time.time()
                answer, error = run_query(mode, question, config, tables, args.community_level, args.response_type)
                latency = time.time() - started
                calls, tokens, cost = metrics_delta(metrics_dir, before)
                runs.append(Run(question, kind, mode, answer, latency, calls, tokens, cost, error))
                answers[(question, mode)] = answer
                status = error or f"{latency:.1f}s, {calls} calls, {tokens:,} tokens, ${cost:.4f}"
                print(f"    {status}")

    judgements: list[Judgement] = []
    if not args.no_judge:
        graph_modes_needed = {GRAPH_MODE_FOR[k] for k in questions}
        if not graph_modes_needed.issubset(set(modes)) or BASELINE_MODE not in modes:
            print("\nSkipping judging: it needs the graph modes and the vector baseline in --modes.")
        else:
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ.get("JUDGE_API_KEY") or os.environ["GRAPHRAG_API_KEY"],
                base_url=JUDGE_BASE_URLS[args.judge_provider],
                max_retries=8,  # free-tier 429s are expected; the client backs off and retries
            )
            for kind, items in questions.items():
                graph_mode = GRAPH_MODE_FOR[kind]
                for i, question in enumerate(items, 1):
                    graph_answer = answers.get((question, graph_mode), "")
                    baseline_answer = answers.get((question, BASELINE_MODE), "")
                    if not graph_answer or not baseline_answer:
                        continue
                    print(f"[judge {kind} {i}/{len(items)}] {question[:58]}")
                    judgements.append(
                        judge_pair(client, judge_model, question, kind, graph_answer, baseline_answer)
                    )

    results_dir = root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (results_dir / f"run_{stamp}.json").write_text(
        json.dumps({"runs": [asdict(r) for r in runs], "judgements": [asdict(j) for j in judgements]}, indent=2),
        encoding="utf-8",
    )
    summary = summarise(runs, judgements)
    (results_dir / f"summary_{stamp}.md").write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"Saved to {results_dir}/run_{stamp}.json and summary_{stamp}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
