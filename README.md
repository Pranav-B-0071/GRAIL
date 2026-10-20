# GRAIL — Graph Retrieval and Inference Layer for Domain Q&A

A chat app that answers questions about ISRO's space programme over a knowledge graph built
with [Microsoft GraphRAG](https://github.com/microsoft/graphrag), with three retrieval modes
side by side and a benchmark that measures whether the graph actually helps.

| Mode | What it does | Good at |
|---|---|---|
| **Global** | Map-reduce over LLM-written community summaries | "What themes/patterns run across the corpus?" |
| **Local** | Finds the entities in your question, expands their neighbourhood, pulls source text | "What exactly did X do?" |
| **Vector baseline** | Plain embedding similarity over raw chunks (GraphRAG's `basic` search) | The control the other two are measured against |

**What this repo is:** the domain adaptation (entity types, tuned prompts, corpus), the chat
interface (mode toggle, streaming, resolved citations), and the evaluation harness.
**What GraphRAG does:** chunking, entity and relationship extraction, Leiden community
detection, community summarisation, and the three search algorithms themselves.

---

## Quick start (Windows)

```powershell
# 1. Environment — GraphRAG needs Python 3.11, 3.12 or 3.13
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. API key — settings.yaml is set up for the Gemini free tier
copy .env.example .env
#    then get a key at https://aistudio.google.com/apikey and put it in GRAPHRAG_API_KEY
#    (to use OpenAI instead, see the header comment in settings.yaml)

# 3. Corpus — 25 Wikipedia articles on ISRO missions, ~71k words
#    (already committed in input/; this only fetches anything missing)
python scripts/fetch_corpus.py

# 4. (Optional but recommended) adapt the extraction prompts to this domain
graphrag prompt-tune --root . --domain "Indian space missions, launch vehicles and spacecraft"

# 5. Build the index — this is the slow step (see "What indexing costs" below)
graphrag index --root .

# 6. Inspect the index: entity/relationship/community counts and indexing cost
python scripts/index_stats.py

# 7. Run the app
streamlit run app.py
```

On macOS or Linux, replace steps 1–2 with `python3.12 -m venv .venv`,
`source .venv/bin/activate`, and `cp .env.example .env`.

`graphrag` is pinned to 3.1.0: 3.1.1+ requires `litellm==1.92.0`, which has no Windows wheel
and fails to install without a Rust toolchain.

---

## What indexing costs

Indexing makes one LLM call per chunk for entity extraction (two with the default gleaning
setting), plus description summarisation and one call per community report. For a ~71k-word
corpus at the default 1200-token chunk size, expect a few hundred LLM calls for extraction
and community reports, plus description summarisation for every entity and relationship that
appears in more than one chunk — that step alone can run past a thousand calls.

On the Gemini free tier this costs nothing, but `settings.yaml` caps the chat model at 10
requests a minute to stay under the free rate limit, so a full index takes several hours and may
cross the daily request quota. If it stops, re-run `graphrag index --root .` — it resumes from
the cache. Gemini's 503 "high demand" responses and network drops are retried with backoff
rather than dropping chunks.

Two knobs if that is too slow or too expensive:

- `graphrag index --root . --method fast` swaps LLM entity extraction for NLP noun-phrase
  extraction. Far fewer calls, noisier graph.
- `concurrent_requests` in `settings.yaml` — lower it if you hit rate limits, raise it if
  your account has room.

The JSON cache in `cache/` means a failed or interrupted run resumes instead of re-paying
for every call. Deleting it forces a full re-index.

---

## Benchmark

```powershell
python eval/benchmark.py --dry-run     # show the plan and call count, spend nothing
python eval/benchmark.py --limit 3     # quick smoke test
python eval/benchmark.py               # full run: 40 questions x 3 modes + 40 judge calls
```

For each of 40 questions (20 thematic, 20 factual) it runs global, local and vector search,
records latency and real token cost from GraphRAG's per-call metrics, then asks an LLM judge
to compare the graph answer against the vector baseline. The A/B order is randomised per
comparison so the judge's position bias doesn't decide the result.

Thematic questions are judged on comprehensiveness, diversity and empowerment; factual ones on
correctness and directness. Results are written to `eval/results/` as raw JSON and a Markdown
summary table.

The judge calls Gemini's OpenAI-compatible endpoint with the same key, using the chat model
from `settings.yaml` by default. Pass `--judge-model` to grade with a different (ideally
stronger) model, or `--judge-provider openai` with an OpenAI key in `JUDGE_API_KEY`.

Factual questions are in the set on purpose. A vector baseline should do well on them, and a
benchmark that only tested the cases where the graph wins wouldn't be worth running.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub **including `output/`** — the parquet tables and the
   `output/lancedb/` embeddings that local and vector search read. The deployed app loads the
   pre-built index rather than re-indexing on startup. `.gitignore` is already set up for this.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at `app.py`.
3. In **Advanced settings**, choose Python 3.12.
4. Under **Settings → Secrets**, add:
   ```toml
   GRAPHRAG_API_KEY = "your-key"
   ```
5. First build takes a few minutes — GraphRAG pulls in roughly a gigabyte of dependencies.

Apps sleep after 12 hours without traffic; the first visitor after that clicks a button to
wake it.

---

## Layout

```
GRAIL/
├─ app.py                 Streamlit chat UI: mode toggle, streaming, citation resolution
├─ settings.yaml          GraphRAG config: models, domain entity types, chunking, vector store
├─ prompts/               Extraction and search prompts (regenerated by `graphrag prompt-tune`)
├─ input/                 Corpus .txt files (created by fetch_corpus.py)
├─ output/                The index: parquet tables + LanceDB vectors (created by indexing)
├─ metrics/               One JSONL record per model call — where the cost numbers come from
├─ scripts/
│  ├─ fetch_corpus.py     Downloads the Wikipedia corpus
│  └─ index_stats.py      Index contents and indexing cost
└─ eval/
   ├─ questions.yaml      20 thematic + 20 factual questions
   └─ benchmark.py        Three-way comparison + LLM-as-judge
```

---

## How the pieces fit

```
input/*.txt
    │  graphrag index
    ▼
chunks ──► entity + relationship extraction (LLM) ──► graph
                                                       │  Leiden clustering
                                                       ▼
                                             communities ──► community reports (LLM)
    │
    ├─ output/*.parquet   entities, relationships, communities, reports, text units
    └─ output/lancedb/    embeddings for entity and chunk lookup
             │
             ├─ global search   reports ──► map ──► reduce ──► answer
             ├─ local search    query ──► similar entities ──► neighbourhood + source text ──► answer
             └─ basic search    query ──► similar chunks ──► answer     (the baseline)
```

## License and credits

- Code in this repository is MIT licensed — see [LICENSE](LICENSE).
- Corpus text from Wikipedia, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Each file in `input/` keeps a header naming its source article.
- Indexing and retrieval by [Microsoft GraphRAG](https://github.com/microsoft/graphrag) (MIT).
