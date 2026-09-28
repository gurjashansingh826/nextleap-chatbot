"""Source registry — the ONLY place a corpus URL is defined. Not a pipeline stage.

15 public pages, 5 of them primary (the schemes the assistant answers about). Provenance rule
P1: no chunk exists without a ``source_url``, and every URL the bot emits must appear here
with ``http_status == 200``.

Adding an AMC later means adding rows to ``SOURCES`` and nothing else — chunking, guards,
prompts, UI and post-checks are all scheme-agnostic (architecture §17).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .config import ROOT

PageRole = Literal[
    "primary",      # a scheme page the assistant answers facts about
    "variant",      # plan variant (Regular / IDCW / Focused) — context only
    "amc",          # fund house page
    "category",     # category listing
    "tool",         # SIP calculator etc.
    "regulatory",   # SEBI riskometer
    "education",    # concepts only — never a source of scheme numbers (P3)
]

CSV_COLUMNS: tuple[str, ...] = (
    "slug",
    "url",
    "title",
    "scheme",
    "category",
    "page_role",
    "fetched_at",
    "http_status",
    "char_count",
)


@dataclass(frozen=True)
class SourceSpec:
    """One public source page in the corpus.

    Frozen so a URL cannot be mutated downstream after chunks are built against it.
    """

    slug: str
    url: str
    title: str
    scheme: str  # "" for non-scheme pages
    category: str  # "" for non-scheme pages
    page_role: PageRole


SOURCES: tuple[SourceSpec, ...] = (
    # ── Primary: the 5 in-scope schemes, all Direct-Growth ─────────────────
    SourceSpec(
        "hdfc-large-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "HDFC Large Cap Fund – Direct Growth",
        "HDFC Large Cap Fund",
        "large_cap",
        "primary",
    ),
    SourceSpec(
        "hdfc-equity-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        "HDFC Equity (Flexi Cap) Fund – Direct Growth",
        "HDFC Equity (Flexi Cap) Fund",
        "flexi_cap",
        "primary",
    ),
    SourceSpec(
        "hdfc-elss-tax-saver-fund-direct-plan-growth",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        "HDFC ELSS Tax Saver Fund – Direct Plan Growth",
        "HDFC ELSS Tax Saver Fund",
        "elss",
        "primary",
    ),
    SourceSpec(
        "hdfc-small-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "HDFC Small Cap Fund – Direct Growth",
        "HDFC Small Cap Fund",
        "small_cap",
        "primary",
    ),
    SourceSpec(
        "hdfc-balanced-advantage-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        "HDFC Balanced Advantage Fund – Direct Growth",
        "HDFC Balanced Advantage Fund",
        "hybrid",
        "primary",
    ),
    # ── Context: plan variants and disambiguation ─────────────────────────
    SourceSpec(
        "hdfc-large-cap-fund-regular-growth",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth",
        "HDFC Large Cap Fund – Regular Growth",
        "HDFC Large Cap Fund",
        "large_cap",
        "variant",
    ),
    SourceSpec(
        "hdfc-large-cap-fund-direct-idcw",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw",
        "HDFC Large Cap Fund – Direct IDCW",
        "HDFC Large Cap Fund",
        "large_cap",
        "variant",
    ),
    SourceSpec(
        "hdfc-focused-large-cap-direct-plan-growth",
        "https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth",
        "HDFC Focused Large Cap Fund – Direct Growth",
        "HDFC Focused Large Cap Fund",
        "large_cap",
        "variant",
    ),
    # ── Context: AMC, categories, tool, regulatory, education ─────────────
    SourceSpec(
        "hdfc-mutual-funds",
        "https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
        "HDFC Mutual Funds (AMC)",
        "",
        "",
        "amc",
    ),
    SourceSpec(
        "best-large-and-midcap-mutual-funds",
        "https://groww.in/mutual-funds/category/best-large-and-midcap-mutual-funds",
        "Best Large and Mid Cap Mutual Funds",
        "",
        "",
        "category",
    ),
    SourceSpec(
        "best-flexi-cap-mutual-funds",
        # NOTE: the URL in the original brief (…/best-flexi-cap-mutual-fund, singular) returns
        # HTTP 404. Verified 2026-09-28: the plural form is the live page. Corrected here so the
        # corpus URL is real; recorded in docs/sources.md as a deviation from the brief.
        "https://groww.in/mutual-funds/category/best-flexi-cap-mutual-funds",
        "Best Flexi Cap Mutual Funds",
        "",
        "",
        "category",
    ),
    SourceSpec(
        "sip-calculator",
        "https://groww.in/calculators/sip-calculator",
        "SIP Calculator",
        "",
        "",
        "tool",
    ),
    SourceSpec(
        "riskometer",
        "https://groww.in/p/riskometer",
        "Riskometer (SEBI)",
        "",
        "",
        "regulatory",
    ),
    SourceSpec(
        "what-is-the-difference-between-elss-and-sip",
        "https://groww.in/questions/what-is-the-difference-between-elss-and-sip",
        "What is the difference between ELSS and SIP?",
        "",
        "",
        "education",
    ),
    SourceSpec(
        "tax-on-mutual-funds",
        "https://groww.in/blog/tax-on-mutual-funds/",
        "Tax on mutual funds",
        "",
        "",
        "education",
    ),
)

# The 5 schemes the assistant is allowed to give scheme-level facts about.
PRIMARY_SCHEMES: tuple[SourceSpec, ...] = tuple(
    s for s in SOURCES if s.page_role == "primary"
)

SOURCES_CSV: Path = ROOT / "data" / "sources.csv"

EXPECTED_SOURCE_COUNT = 15
EXPECTED_PRIMARY_COUNT = 5

VALID_SLUG_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


# ── Registry invariants ──────────────────────────────────────────────────────
def validate_registry(specs: Iterable[SourceSpec] = SOURCES) -> None:
    """Assert the registry satisfies the hard requirements. Raises ValueError on violation.

    Called explicitly (by tests and the verification script) rather than at import time, so
    that legitimately extending the corpus later does not require editing this function.
    """
    specs = tuple(specs)
    problems: list[str] = []

    if len(specs) != EXPECTED_SOURCE_COUNT:
        problems.append(
            f"expected {EXPECTED_SOURCE_COUNT} sources, found {len(specs)}"
        )

    primary = [s for s in specs if s.page_role == "primary"]
    if len(primary) != EXPECTED_PRIMARY_COUNT:
        problems.append(
            f"expected {EXPECTED_PRIMARY_COUNT} primary sources, found {len(primary)}"
        )

    slugs = [s.slug for s in specs]
    if len(set(slugs)) != len(slugs):
        dupes = sorted({s for s in slugs if slugs.count(s) > 1})
        problems.append(f"duplicate slugs: {dupes}")

    urls = [s.url for s in specs]
    if len(set(urls)) != len(urls):
        dupes = sorted({u for u in urls if urls.count(u) > 1})
        problems.append(f"duplicate urls: {dupes}")

    for s in specs:
        bad = set(s.slug) - VALID_SLUG_CHARS
        if bad:
            problems.append(f"slug {s.slug!r} has non-URL-safe chars: {sorted(bad)}")
        if not s.url.startswith("https://"):
            problems.append(f"{s.slug}: url must be https, got {s.url!r}")
        if s.page_role not in ("primary", "variant", "amc", "category", "tool",
                               "regulatory", "education"):
            problems.append(f"{s.slug}: unknown page_role {s.page_role!r}")
        # A primary page must identify its scheme and category; a non-scheme page must not
        # claim one (page_role keeps education pages away from scheme facts — P3).
        if s.page_role == "primary" and not (s.scheme and s.category):
            problems.append(f"{s.slug}: primary page must set scheme and category")

    if problems:
        raise ValueError(
            "Source registry is invalid:\n  - " + "\n  - ".join(problems)
        )


def scheme_names() -> list[str]:
    """The 5 in-scope scheme names, for display and for the UI coverage list."""
    return [s.scheme for s in PRIMARY_SCHEMES]


def scheme_name_for_slug(slug: str) -> str:
    """Scheme name for a slug, or ``""`` if the slug is unknown or not a scheme page.

    Used by the conversational memory, which stores slugs (short, stable, no spaces) but has
    to build a human-readable retrieval query from them. A context page like ``riskometer``
    has no scheme name, so the empty string is a real answer and the caller must check it
    rather than assume the lookup succeeded.
    """
    for spec in SOURCES:
        if spec.slug == slug:
            return spec.scheme
    return ""


def all_schemes() -> list[str]:
    """Every distinct scheme mentioned anywhere in the corpus, in first-seen order."""
    seen: dict[str, None] = {}
    for s in SOURCES:
        if s.scheme:
            seen.setdefault(s.scheme, None)
    return list(seen)


def by_role(page_role: str) -> list[SourceSpec]:
    """Sources with a given page_role."""
    return [s for s in SOURCES if s.page_role == page_role]


# ── sources.csv I/O (written by STAGE 1, read by the answer post-check) ──────
def write_sources_csv(rows: list[dict], path: Path | None = None) -> Path:
    """Write ``data/sources.csv``. Any row missing a column is written as empty."""
    path = path or SOURCES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    return path


def load_sources_csv(path: Path | None = None) -> list[dict]:
    """Read ``data/sources.csv``. Returns ``[]`` if it does not exist yet (pre-ingest)."""
    path = path or SOURCES_CSV
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def valid_urls(path: Path | None = None) -> set[str]:
    """URLs that were actually fetched OK.

    This is the authority behind post-check 2 (architecture §12): a citation that is not in
    this set invalidates the answer. Before ingestion it is empty, which is correct — no
    answer should be possible before the corpus exists.
    """
    urls: set[str] = set()
    for row in load_sources_csv(path):
        try:
            status = int(row.get("http_status") or 0)
        except (TypeError, ValueError):
            continue
        if status == 200 and row.get("url"):
            urls.add(row["url"])
    return urls


def _summary_table() -> str:
    width = max(len(s.slug) for s in SOURCES)
    lines = [
        f"SOURCE REGISTRY — {len(SOURCES)} pages, {len(PRIMARY_SCHEMES)} primary",
        "-" * (width + 34),
        f"{'#':>2}  {'page_role':<10} {'slug':<{width}}  scheme / topic",
        "-" * (width + 34),
    ]
    for i, s in enumerate(SOURCES, start=1):
        topic = s.scheme or s.title
        lines.append(f"{i:>2}  {s.page_role:<10} {s.slug:<{width}}  {topic}")
    lines.append("-" * (width + 34))
    lines.append(f"   all_schemes(): {len(all_schemes())} distinct")
    return "\n".join(lines)


if __name__ == "__main__":
    # Phase 1 verification (implementation.md, Phase 1). Run with:
    #     python -m mf_rag.sources
    validate_registry()
    print("registry valid: 15 sources, 5 primary, all slugs unique and URL-safe")
    print()
    print(_summary_table())
    print()
    print(f"sources.csv present: {SOURCES_CSV.exists()}")
    print(f"valid_urls()        : {len(valid_urls())}  (0 is expected before ingestion)")
