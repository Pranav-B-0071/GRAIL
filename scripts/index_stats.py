"""Report what the index actually contains - and print the numbers for your resume bullet.

Run after indexing:

    python scripts/index_stats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = ["documents", "text_units", "entities", "relationships", "communities", "community_reports"]


def corpus_stats(input_dir: Path) -> tuple[int, int]:
    files = sorted(input_dir.glob("*.txt"))
    words = sum(len(f.read_text(encoding="utf-8").split()) for f in files)
    return len(files), words


def index_stats(output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in TABLES:
        path = output_dir / f"{name}.parquet"
        if path.exists():
            counts[name] = len(pd.read_parquet(path))
    return counts


def indexing_cost(metrics_dir: Path) -> tuple[int, int, float]:
    """Total LLM calls, tokens and cost recorded across every metrics file."""
    calls = tokens = 0
    cost = 0.0
    for path in sorted(metrics_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    metrics = json.loads(line).get("metrics", {})
                except json.JSONDecodeError:
                    continue
                calls += 1
                tokens += int(metrics.get("total_tokens", 0) or 0)
                cost += float(metrics.get("total_cost", 0) or 0)
    return calls, tokens, cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()

    counts = index_stats(root / "output")
    if not counts:
        print(f"No index found in {root / 'output'}. Run `graphrag index --root .` first.")
        return 1

    n_articles, n_words = corpus_stats(root / "input")
    calls, tokens, cost = indexing_cost(root / "metrics")

    print(f"\nCorpus:  {n_articles} documents, {n_words:,} words (~{int(n_words * 1.35):,} tokens)\n")
    print("Index:")
    for name in TABLES:
        if name in counts:
            print(f"  {name:<20} {counts[name]:>8,}")

    if "communities" in counts:
        communities = pd.read_parquet(root / "output" / "communities.parquet")
        if "level" in communities.columns:
            by_level = communities.groupby("level").size().to_dict()
            levels = ", ".join(f"L{level}: {n}" for level, n in sorted(by_level.items()))
            print(f"  {'communities by level':<20} {levels}")

    if calls:
        print(f"\nIndexing cost (from metrics/): {calls:,} model calls, {tokens:,} tokens, ${cost:.3f}")
    else:
        print("\nNo metrics recorded yet (metrics/ is empty).")

    entities = counts.get("entities", 0)
    relationships = counts.get("relationships", 0)
    print("\n" + "-" * 78)
    print("Paste-ready numbers for the resume bullet:\n")
    print(f'  "...a {n_articles}-article, ~{n_words // 1000}k-word corpus ... producing a knowledge graph')
    print(f'   of {entities:,} entities and {relationships:,} relationships organised into')
    print(f'   {counts.get("communities", 0):,} Leiden communities with LLM-written summaries."')
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
