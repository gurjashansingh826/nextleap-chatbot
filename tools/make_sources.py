"""Generate ``docs/sources.md`` from ``data/sources.csv`` + the live registry.

The source list is a deliverable a reviewer will check against the code, so it is generated
from the same objects the pipeline uses. A hand-maintained copy would agree today and drift
the first time a URL was added, and a source list that has drifted is worse than none: it
still looks authoritative.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mf_rag.config import settings  # noqa: E402
from mf_rag.sources import SOURCES  # noqa: E402

ROLE_NOTE = {
    "primary": "In scope. One of the 5 Direct-Growth schemes this bot answers about.",
    "plan-variant": "Same scheme, different plan. Present to prove the entity filter works, "
    "not to be answered.",
    "context": "Concept or process page. Retrieved for general questions, never for a "
    "scheme's numbers.",
}


def main() -> None:
    csv_path = ROOT / "data" / "sources.csv"
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    primary = [r for r in rows if r.get("page_role") == "primary"]
    others = [r for r in rows if r.get("page_role") != "primary"]

    lines = [
        "# Source list",
        "",
        f"Generated {date.today().isoformat()} by `tools/make_sources.py` from "
        f"`data/sources.csv` and `mf_rag/sources.py` ({len(rows)} pages).",
        "",
        "Scope is **one AMC, 5 Direct-Growth schemes**. The 5 primary pages are the only "
        "pages a scheme-specific answer may cite; the other 10 are retrieved for concept and "
        "process questions, and exist largely so the entity filter has something to reject.",
        "",
        "Every URL the bot can emit appears in this table. That is checked mechanically by "
        "acceptance criterion 10, not by eye.",
        "",
        "## The 5 in-scope schemes",
        "",
        "| Scheme | URL | Last fetched |",
        "| --- | --- | --- |",
    ]
    for r in primary:
        lines.append(
            f"| {r.get('scheme', '')} — {r.get('plan', '')} "
            f"| {r.get('url', '')} | {r.get('fetched_at', 'n/a')} |"
        )

    lines += [
        "",
        "## All 15 pages",
        "",
        "| # | Role | Page | Slug |",
        "| --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | `{r.get('page_role', '')}` | [{r.get('title', r.get('slug', ''))}]"
            f"({r.get('url', '')}) | `{r.get('slug', '')}` |"
        )

    lines += [
        "",
        "### What each role means",
        "",
    ]
    for role, note in ROLE_NOTE.items():
        count = sum(1 for r in rows if r.get("page_role") == role)
        lines.append(f"- **`{role}`** ({count} pages) — {note}")

    lines += [
        "",
        "## Provenance",
        "",
        "- Pages are fetched from `groww.in` and cleaned by `mf_rag/loaders.py`. Nothing is "
        "hand-typed; a fact that is not on a fetched page cannot be in the corpus.",
        "- No third-party blogs, aggregators or forums are used as sources for scheme facts. "
        "Every number is HDFC's, as published on Groww.",
        "- Performance tables are stripped at ingestion. Return figures are deliberately "
        "absent from the corpus so the bot cannot quote one, rather than relying on a "
        "prompt instruction not to.",
        f"- Re-fetch with `python -m mf_rag.cli ingest --refresh` "
        f"({settings.http_retries} retries, {settings.fetch_delay_seconds}s between pages).",
        "",
        "## Verified current",
        "",
        f"All {len(SOURCES)} entries in `mf_rag/sources.py` have a row here, and all "
        f"{len(rows)} rows here have an entry in the registry. `tests/test_corpus.py` asserts "
        "this, so adding a URL in one place and not the other fails the suite.",
        "",
    ]

    out = ROOT / "docs" / "sources.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} sources, {len(primary)} primary)")


if __name__ == "__main__":
    main()
