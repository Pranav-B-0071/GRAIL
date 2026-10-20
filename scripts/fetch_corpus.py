"""Download the GRAIL corpus: ISRO mission articles from Wikipedia, as plain text.

Wikipedia text is CC BY-SA 4.0. Each file keeps a header crediting the source article,
and the README credits Wikipedia.

Usage
-----
    python scripts/fetch_corpus.py              # download anything missing
    python scripts/fetch_corpus.py --force      # re-download everything
    python scripts/fetch_corpus.py --stats      # just report what is already on disk
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "GRAIL-corpus-builder/1.0 (educational GraphRAG project; https://github.com/Pranav-B-0071/GRAIL)"

# 25 articles: missions, launch vehicles, satellite programmes, institutions, people.
# Wide enough that "what patterns show up across ISRO's failures?" has a real answer,
# narrow enough that indexing stays cheap.
ARTICLES = [
    "Indian Space Research Organisation",
    "Chandrayaan-1",
    "Chandrayaan-2",
    "Chandrayaan-3",
    "Mars Orbiter Mission",
    "Aditya-L1",
    "AstroSat",
    "XPoSat",
    "Gaganyaan",
    "SpaDeX",
    "NISAR (satellite)",
    "Polar Satellite Launch Vehicle",
    "Geosynchronous Satellite Launch Vehicle",
    "LVM3",
    "Small Satellite Launch Vehicle",
    "RLV Technology Demonstration Programme",
    "Indian Regional Navigation Satellite System",
    "Indian National Satellite System",
    "GSAT",
    "RISAT",
    "Cartosat",
    "Satish Dhawan Space Centre",
    "Vikram Sarabhai Space Centre",
    "Vikram Sarabhai",
    "NewSpace India Limited",
]

TARGET_WORDS = 71_000  # corpus size the README quotes


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return re.sub(r"_+", "_", slug)


def fetch_extract(title: str, session: requests.Session, retries: int = 3) -> tuple[str, str] | None:
    """Return (resolved_title, plain_text) or None if the article does not exist."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    for attempt in range(retries):
        try:
            resp = session.get(API, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"    network error ({exc}), retrying...")
            time.sleep(2 * (attempt + 1))
            continue

        if resp.status_code in (429, 503):
            wait = 5 * (attempt + 1)
            print(f"    rate limited (HTTP {resp.status_code}), waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()

        pages = resp.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        page = pages[0]
        return page["title"], page.get("extract", "")
    msg = f"gave up fetching {title!r} after {retries} attempts"
    raise RuntimeError(msg)


def write_article(input_dir: Path, resolved_title: str, text: str) -> Path:
    """Write one article with a provenance header GraphRAG will index along with the body."""
    url = "https://en.wikipedia.org/wiki/" + resolved_title.replace(" ", "_")
    header = (
        f"# {resolved_title}\n"
        f"Source: {url}\n"
        f"License: CC BY-SA 4.0 (Wikipedia contributors)\n\n"
    )
    path = input_dir / f"{slugify(resolved_title)}.txt"
    path.write_text(header + text, encoding="utf-8")
    return path


def report(input_dir: Path) -> None:
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print("No articles in input/ yet.")
        return
    total = 0
    print(f"\n{'article':<52}{'words':>9}")
    print("-" * 61)
    for f in files:
        words = len(f.read_text(encoding="utf-8").split())
        total += words
        print(f"{f.stem[:50]:<52}{words:>9,}")
    print("-" * 61)
    print(f"{'TOTAL':<52}{total:>9,}")
    print(f"\n{len(files)} articles, ~{total:,} words (~{int(total * 1.35):,} tokens).")
    drift = (total - TARGET_WORDS) / TARGET_WORDS
    if abs(drift) > 0.25:
        direction = "more" if drift > 0 else "fewer"
        print(
            f"Note: that is {abs(drift):.0%} {direction} than the ~{TARGET_WORDS:,} words this project "
            f"was scoped around - Wikipedia articles change over time. Add or drop articles in ARTICLES."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download articles already on disk")
    parser.add_argument("--stats", action="store_true", help="only report on what is already downloaded")
    parser.add_argument("--input-dir", default="input", help="where to write the .txt files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        report(input_dir)
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    missing: list[str] = []
    for i, title in enumerate(ARTICLES, 1):
        target = input_dir / f"{slugify(title)}.txt"
        if target.exists() and not args.force:
            print(f"[{i:>2}/{len(ARTICLES)}] skip (already have) {title}")
            continue

        print(f"[{i:>2}/{len(ARTICLES)}] fetching {title}")
        result = fetch_extract(title, session)
        if result is None:
            print("    NOT FOUND - Wikipedia may have renamed it. Fix the title in ARTICLES.")
            missing.append(title)
            continue

        resolved_title, text = result
        if len(text.split()) < 200:
            print(f"    WARNING: only {len(text.split())} words - looks like a stub or disambiguation page.")
        path = write_article(input_dir, resolved_title, text)
        if resolved_title != title:
            print(f"    redirected to {resolved_title!r}")
        print(f"    wrote {path.name} ({len(text.split()):,} words)")
        time.sleep(0.4)  # be polite to the API

    report(input_dir)

    if missing:
        print("\nCould not fetch:")
        for title in missing:
            print(f"  - {title}")
        print("Search Wikipedia for the current title and update ARTICLES in this file.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
