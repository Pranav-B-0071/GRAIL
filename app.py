"""GRAIL - Graph Retrieval and Inference Layer for Domain Q&A.

A chat UI over a GraphRAG index with three retrieval modes:

  Global  - map-reduce over community summaries. Good at "what themes/patterns" questions.
  Local   - entity-anchored retrieval. Good at "what exactly did X do" questions.
  Vector  - GraphRAG's basic search: plain embedding similarity over text chunks.
            Kept as the honest baseline the other two are measured against.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import queue
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import streamlit as st
from graphrag.callbacks.query_callbacks import QueryCallbacks

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

MODES = {
    "Global (themes)": "global",
    "Local (facts)": "local",
    "Vector baseline": "basic",
}

MODE_HELP = {
    "global": "Map-reduce over the community summaries. Use for broad questions about the whole corpus.",
    "local": "Finds the entities your question is about, then pulls their neighbourhood and source text.",
    "basic": "Plain vector similarity over raw chunks - no graph. The baseline for comparison.",
}

EXAMPLE_QUESTIONS = {
    "global": [
        "What recurring themes show up across ISRO's mission failures?",
        "How has ISRO's approach to lunar exploration evolved over time?",
        "What role has international collaboration played across these missions?",
    ],
    "local": [
        "Which launch vehicle carried Aditya-L1, and to what orbit?",
        "What instruments did the Pragyan rover carry?",
        "What caused the Chandrayaan-2 lander to fail?",
    ],
    "basic": [
        "What is the LVM3?",
        "When did the Mars Orbiter Mission launch?",
    ],
}

# GraphRAG answers cite their evidence inline, e.g.
#   "... a soft landing [Data: Reports (12, 15); Entities (3)]"
# These two patterns pull those references back out so we can show the real source.
DATA_BLOCK_RE = re.compile(r"\[Data:([^\]]+)\]")
DATA_PART_RE = re.compile(r"([A-Za-z ]+?)\s*\(([^)]*)\)")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def ensure_api_key() -> bool:
    """GraphRAG reads GRAPHRAG_API_KEY from the environment.

    Locally that comes from .env; on Streamlit Community Cloud it comes from st.secrets,
    so copy it across before any GraphRAG import touches the config.
    """
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if os.environ.get("GRAPHRAG_API_KEY"):
        return True
    try:
        key = st.secrets.get("GRAPHRAG_API_KEY")  # type: ignore[union-attr]
    except Exception:
        key = None
    if key:
        os.environ["GRAPHRAG_API_KEY"] = str(key)
        return True
    return False


@st.cache_resource(show_spinner=False)
def load_config(root: str):
    from graphrag.config.load_config import load_config as _load_config

    return _load_config(root_dir=Path(root))


@st.cache_data(show_spinner=False)
def load_tables(root: str, _cache_key: float) -> dict[str, pd.DataFrame]:
    """Load the index tables GraphRAG wrote. _cache_key busts the cache after re-indexing."""
    out = Path(root) / "output"
    names = ["entities", "communities", "community_reports", "text_units", "relationships", "documents"]
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = out / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
    return tables


def index_cache_key(root: str) -> float:
    out = Path(root) / "output"
    files = list(out.glob("*.parquet"))
    return max((f.stat().st_mtime for f in files), default=0.0)


# --------------------------------------------------------------------------------------
# Async -> sync streaming bridge
# --------------------------------------------------------------------------------------
def stream_sync(agen_factory: Callable[[], Any]) -> Iterator[str]:
    """Consume an async generator on a worker thread and yield its items synchronously.

    Streamlit's script thread is not async, and GraphRAG's streaming search is. A queue
    between the two lets tokens render as they arrive instead of after the full answer.
    """
    q: queue.Queue = queue.Queue()
    done = object()

    def run() -> None:
        async def consume() -> None:
            try:
                async for token in agen_factory():
                    q.put(token)
            except Exception as exc:  # re-raised on the Streamlit thread
                q.put(exc)
            finally:
                q.put(done)

        asyncio.run(consume())

    threading.Thread(target=run, daemon=True).start()

    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class ContextCollector(QueryCallbacks):
    """Captures the context GraphRAG built for a query.

    The streaming API yields only answer tokens; the evidence it retrieved arrives here
    instead. Must subclass QueryCallbacks - graphrag validates the argument type.
    """

    def __init__(self) -> None:
        self.context: Any = None

    def on_context(self, context: Any) -> None:
        self.context = context


def build_stream(mode: str, query: str, config: Any, tables: dict[str, pd.DataFrame],
                 community_level: int, response_type: str, collector: ContextCollector):
    """Return a zero-arg factory producing GraphRAG's async token generator for `mode`."""
    import graphrag.api as api

    callbacks = [collector]

    if mode == "global":
        return lambda: api.global_search_streaming(
            config=config,
            entities=tables["entities"],
            communities=tables["communities"],
            community_reports=tables["community_reports"],
            community_level=community_level,
            dynamic_community_selection=False,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
        )
    if mode == "local":
        return lambda: api.local_search_streaming(
            config=config,
            entities=tables["entities"],
            communities=tables["communities"],
            community_reports=tables["community_reports"],
            text_units=tables["text_units"],
            relationships=tables["relationships"],
            covariates=None,
            community_level=community_level,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
        )
    return lambda: api.basic_search_streaming(
        config=config,
        text_units=tables["text_units"],
        response_type=response_type,
        query=query,
        callbacks=callbacks,
    )


# --------------------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------------------
def parse_citations(answer: str) -> dict[str, set[int]]:
    """Pull {'reports': {12, 15}, 'entities': {3}} out of the inline [Data: ...] markers."""
    found: dict[str, set[int]] = defaultdict(set)
    for block in DATA_BLOCK_RE.findall(answer):
        for kind, ids in DATA_PART_RE.findall(block):
            key = kind.strip().lower()
            for token in ids.split(","):
                token = token.strip()
                if token.isdigit():
                    found[key].add(int(token))
    return found


def rows_for(table: pd.DataFrame | None, ids: set[int]) -> pd.DataFrame:
    if table is None or table.empty or "human_readable_id" not in table.columns:
        return pd.DataFrame()
    return table[table["human_readable_id"].isin(ids)]


def render_citations(answer: str, tables: dict[str, pd.DataFrame]) -> None:
    """Resolve the inline markers to the actual reports, entities and source chunks."""
    cited = parse_citations(answer)
    if not cited:
        return

    total = sum(len(v) for v in cited.values())
    with st.expander(f"Sources ({total} cited)", expanded=False):
        reports = rows_for(tables.get("community_reports"), cited.get("reports", set()))
        if not reports.empty:
            st.markdown("**Community reports**")
            for _, row in reports.iterrows():
                st.markdown(f"- **[{row['human_readable_id']}] {row['title']}** — {row['summary'][:300]}")

        entities = rows_for(tables.get("entities"), cited.get("entities", set()))
        if not entities.empty:
            st.markdown("**Entities**")
            for _, row in entities.iterrows():
                desc = (row.get("description") or "")[:200]
                st.markdown(f"- **[{row['human_readable_id']}] {row['title']}** ({row.get('type', 'n/a')}) — {desc}")

        rels = rows_for(tables.get("relationships"), cited.get("relationships", set()))
        if not rels.empty:
            st.markdown("**Relationships**")
            for _, row in rels.iterrows():
                desc = (row.get("description") or "")[:200]
                st.markdown(f"- **[{row['human_readable_id']}] {row['source']} → {row['target']}** — {desc}")

        sources = rows_for(tables.get("text_units"), cited.get("sources", set()))
        if not sources.empty:
            docs = tables.get("documents")
            st.markdown("**Source text**")
            for _, row in sources.iterrows():
                title = ""
                if docs is not None and "id" in docs.columns:
                    match = docs[docs["id"] == row.get("document_id")]
                    if not match.empty:
                        title = match.iloc[0].get("title", "")
                snippet = str(row["text"])[:400].replace("\n", " ")
                st.markdown(f"- **[{row['human_readable_id']}] {title}** — {snippet}...")


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="GRAIL", page_icon="🛰️", layout="centered")
    st.title("🛰️ GRAIL")
    st.caption("Graph Retrieval and Inference Layer — knowledge-graph Q&A over ISRO's mission history")

    if not ensure_api_key():
        st.error(
            "No API key found. Locally: put `GRAPHRAG_API_KEY=sk-...` in a `.env` file at the project root. "
            "On Streamlit Community Cloud: add it under **Settings → Secrets**."
        )
        st.stop()

    if not OUTPUT_DIR.exists() or not list(OUTPUT_DIR.glob("*.parquet")):
        st.error(
            "No index found in `output/`. Build one first:\n\n"
            "```\npython scripts/fetch_corpus.py\ngraphrag index --root .\n```"
        )
        st.stop()

    with st.spinner("Loading index..."):
        config = load_config(str(PROJECT_ROOT))
        tables = load_tables(str(PROJECT_ROOT), index_cache_key(str(PROJECT_ROOT)))

    missing = {"entities", "communities", "community_reports", "text_units", "relationships"} - set(tables)
    if missing:
        st.error(f"The index is incomplete — missing tables: {', '.join(sorted(missing))}. Re-run indexing.")
        st.stop()

    max_level = int(tables["communities"]["level"].max()) if "level" in tables["communities"] else 2

    with st.sidebar:
        st.subheader("Retrieval mode")
        mode_label = st.radio("Mode", list(MODES), label_visibility="collapsed")
        mode = MODES[mode_label]
        st.caption(MODE_HELP[mode])

        st.subheader("Settings")
        community_level = st.slider(
            "Community level", 0, max_level, min(2, max_level),
            help="Higher levels are smaller, more specific communities. Ignored by the vector baseline.",
            disabled=(mode == "basic"),
        )
        response_type = st.selectbox(
            "Response style",
            ["Multiple Paragraphs", "Single Paragraph", "List of 3-7 Points", "Short Answer"],
        )

        st.divider()
        st.subheader("Index")
        st.metric("Entities", f"{len(tables['entities']):,}")
        st.metric("Relationships", f"{len(tables['relationships']):,}")
        st.metric("Communities", f"{len(tables['communities']):,}")
        st.metric("Text chunks", f"{len(tables['text_units']):,}")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        for i, question in enumerate(EXAMPLE_QUESTIONS[mode]):
            if st.button(question, key=f"example_{mode}_{i}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("caption"):
                st.caption(message["caption"])

    prompt = st.chat_input("Ask about ISRO's missions...") or st.session_state.pending
    st.session_state.pending = None
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        collector = ContextCollector()
        factory = build_stream(mode, prompt, config, tables, community_level, response_type, collector)
        started = time.time()
        try:
            answer = st.write_stream(stream_sync(factory))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Search failed: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"_Search failed: {exc}_", "caption": ""}
            )
            return

        elapsed = time.time() - started
        caption = f"{mode_label} · {elapsed:.1f}s"
        st.caption(caption)
        render_citations(str(answer), tables)

    st.session_state.messages.append({"role": "assistant", "content": str(answer), "caption": caption})


if __name__ == "__main__":
    main()
