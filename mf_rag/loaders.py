"""STAGE 1 — Loading: fetch public pages, extract main text, clean, write markdown + CSV.

Design notes that matter (architecture §5, ADR-7):

* **Cache-first.** ``data/raw/<slug>.html`` is reused unless ``refresh=True``, so re-running
  the build is instant and the corpus is reproducible offline (NFR-6).
* **Failure is data, not an exception** (FR-1). A page that will not fetch still gets a row in
  ``sources.csv`` with its real status, and the run continues. If *zero* pages succeed we
  raise, because that is a real failure worth stopping for.
* **``include_tables=True`` is load-bearing.** Groww renders expense ratio and exit load in
  tables; the default extractor discards them, and those are two of the graded facts.
* **Numbers are never rewritten.** ``clean_text`` normalises dashes and whitespace but will
  not reformat, round, or reflow a percentage — rule R2.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from trafilatura import extract as trafilatura_extract

from .config import settings
from .sources import SOURCES, SourceSpec, write_sources_csv

# ── Cleaning ─────────────────────────────────────────────────────────────────

#: Zero-width and bidi control characters that survive HTML extraction and poison embeddings.
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")

#: Unicode punctuation normalised to ASCII so "3-year" and "3–year" embed alike.
DASH_RE = re.compile(r"[\u2010-\u2015\u2212]")
QUOTE_RE = re.compile(r"[\u2018\u2019\u201a\u201b\u2032]")
DQUOTE_RE = re.compile(r"[\u201c\u201d\u201e\u201f\u2033]")
BULLET_RE = re.compile(r"^[\u2022\u25cf\u25aa\u25e6\u2043\u2219\u00b7]\s*")

#: Case-insensitive substrings. Any line containing one of these is dropped whole. Keeping
#: these out of the corpus entirely is cheaper than filtering them at retrieval time.
BOILERPLATE_PATTERNS: tuple[str, ...] = (
    "groww app download",
    "download the groww app",
    "trusted by",
    "referral",
    "cookie",
    "privacy policy",
    "terms of use",
    "track your investment",
    "app store",
    "google play",
    "open in app",
    "log in",
    "sign up",
    "disclaimer: mutual fund investments are subject to market risks",
    "mutual fund investments are subject to market risks",
    "read all scheme related documents carefully",
    "investors should read all scheme",
    "get started with zero brokerage",
    "open a free account in mins",
    "start a sip in mins",
    "download our app",
    "disclaimer",
)

#: HTML elements that never contain scheme facts.
_CHROME_TAGS = ("nav", "header", "footer", "aside", "script", "style", "noscript", "form", "svg")

#: Cookie/consent and app-promotion containers, matched by id/class/aria attribute.
_CHROME_ATTR_RE = re.compile(
    r"(cookie|consent|banner|popup|modal|interstitial|app-?download|appdownload|"
    r"onetrust|nexage|advert|ad-slot|sticky-?bar)",
    re.IGNORECASE,
)


def is_boilerplate(line: str) -> bool:
    """True if a line should be dropped as site chrome rather than scheme content."""
    low = line.lower()
    return any(p in low for p in BOILERPLATE_PATTERNS)


def clean_text(text: str) -> str:
    """Normalise extracted text without altering any numeric value.

    Steps: strip zero-width chars, normalise dashes/quotes/bullets, drop boilerplate lines,
    collapse intra-line whitespace, collapse runs of blank lines, trim.

    Deliberately absent: any step that could reformat a number. ``0.63%`` and ``1.25`` reach
    the corpus byte-for-byte as they appeared on the page (rule R2).
    """
    if not text:
        return ""

    text = ZERO_WIDTH_RE.sub("", text)
    text = DASH_RE.sub("-", text)
    text = QUOTE_RE.sub("'", text)
    text = DQUOTE_RE.sub('"', text)

    kept: list[str] = []
    for raw in text.splitlines():
        line = BULLET_RE.sub("", raw).strip()
        if not line:
            # Preserve at most one blank line to keep paragraph breaks in the markdown.
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if is_boilerplate(line):
            continue
        # Collapse runs of spaces/tabs but never touch digits, units or '%'.
        kept.append(re.sub(r"[ \t ]+", " ", line))

    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()

    return "\n".join(kept)


#: Markdown-table headers that mark a block as performance or holdings data. trafilatura emits
#: Groww's returns, peer-comparison and holdings tables verbatim, and those are exactly the
#: tables this assistant must never cite (PRD §10.3 — no performance claims). They are dropped
#: here, at ingestion, rather than left for the intent guard to catch at query time.
PERFORMANCE_TABLE_MARKERS: tuple[str, ...] = (
    "would've become", "total investment", "fund returns", "category average",
    "fund size", "1y", "3y", "5y", "10y", "ytd", "since inception", "cagr",
    "rank", "returns", "return", "performance", "peer", "holdings", "portfolio",
    "sector", "asset allocation", "invested", "growth", "risk", "volatility",
    "sharpe", "sortino", "alpha", "beta",
)

#: Section headings whose entire body is performance data.
PERFORMANCE_SECTION_MARKERS: tuple[str, ...] = (
    "returns and rankings", "fund performance", "performance", "returns",
    "portfolio", "holdings", "peers", "ranking",
)


def _is_performance_header(row: str) -> bool:
    low = row.lower()
    return any(marker in low for marker in PERFORMANCE_TABLE_MARKERS)


def strip_performance_tables(text: str) -> str:
    """Remove markdown tables and sections that carry returns, rankings, or holdings.

    Conservative by construction: a table is dropped only when its own header row matches a
    performance marker. Fee data on Groww lives outside these tables, so nothing graded is
    lost — and the values that matter are additionally captured structurally by
    ``scheme_facts``, so this is defence in depth rather than the only defence.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    dropped_tables = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|"):
            # Collect the whole contiguous table block.
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            header = block[0] if block else ""
            if _is_performance_header(header):
                dropped_tables += 1
                continue
            out.extend(block)
            continue
        low = line.lower().lstrip("# ").strip()
        if low and any(low.startswith(m) for m in PERFORMANCE_SECTION_MARKERS):
            # Skip this heading and its body until the next heading of any level.
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.lstrip().startswith("#"):
                    break
                if nxt.strip() == "" and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("#"):
                    break
                i += 1
            dropped_tables += 1
            continue
        out.append(line)
        i += 1

    # Collapse any run of 3+ blank lines left behind.
    cleaned: list[str] = []
    for ln in out:
        if ln.strip() == "" and len(cleaned) >= 2 and cleaned[-1].strip() == "" and cleaned[-2].strip() == "":
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def _strip_chrome(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove navigation, scripts, and consent/app-promo containers in place."""
    for tag in soup.find_all(_CHROME_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        attrs = " ".join(
            str(v) for k, v in tag.attrs.items() if k in ("id", "class", "aria-label")
        )
        if attrs and _CHROME_ATTR_RE.search(attrs):
            tag.decompose()
    return soup


def _bs4_extract(html: str) -> str:
    """Fallback extractor: strip chrome, then take the densest remaining block of text.

    "Densest" beats "first" because financial pages bury the fee table below hero sections.
    """
    soup = _strip_chrome(BeautifulSoup(html, "lxml"))
    best, best_len = "", 0
    for tag in soup.find_all(
        ["main", "article", "section", "div", "body"]
    ):
        # Skip containers that are mostly links — that is a nav list, not content.
        text = tag.get_text("\n", strip=True)
        links = len(tag.find_all("a"))
        if links and links * 60 > len(text):
            continue
        if len(text) > best_len:
            best, best_len = text, len(text)
    if not best:
        best = soup.get_text("\n", strip=True)
    return best


def extract_main_text(html: str, url: str) -> str:
    """Extract the main content of a page. trafilatura first, BeautifulSoup as fallback."""
    text = ""
    try:
        text = trafilatura_extract(
            html,
            url=url,
            include_tables=True,
            include_comments=False,
            include_formatting=True,
            favor_precision=True,
        ) or ""
    except Exception:  # noqa: BLE001 - extraction must never abort the run
        text = ""
    if len(text) < 200:
        fallback = _bs4_extract(html)
        if len(fallback) > len(text):
            text = fallback
    return text


# ── Fetching ────────────────────────────────────────────────────────────────


@dataclass
class FetchResult:
    """Outcome of fetching one source page."""

    spec: SourceSpec
    ok: bool
    http_status: int
    html: str
    from_cache: bool = False
    error: str = ""


@dataclass
class IngestSummary:
    """Aggregate outcome of an ingestion run (FR-15 stage line)."""

    requested: int = 0
    fetched: int = 0
    cached: int = 0
    failed: int = 0
    low_text: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"STAGE 1 · pages requested {self.requested} · fetched {self.fetched} "
            f"· cached {self.cached} · failed {self.failed} · low_text {len(self.low_text)}"
        )


def raw_path(spec: SourceSpec) -> Path:
    return settings.raw_dir / f"{spec.slug}.html"


def processed_path(spec: SourceSpec) -> Path:
    return settings.processed_dir / f"{spec.slug}.md"


def fetch_page(spec: SourceSpec, refresh: bool = False) -> FetchResult:
    """Fetch one page, using the local cache when present (ADR-7).

    Retries with exponential backoff on transport errors and 5xx/429. A 4xx other than 429
    is not retried — the server has made its decision.
    """
    import httpx

    cache = raw_path(spec)
    if cache.exists() and cache.stat().st_size > 0 and not refresh:
        return FetchResult(
            spec=spec,
            ok=True,
            http_status=200,
            html=cache.read_text(encoding="utf-8", errors="replace"),
            from_cache=True,
        )

    delay = 1.0
    last_error = ""
    status = 0
    with httpx.Client(
        timeout=settings.http_timeout,
        follow_redirects=True,
        headers={
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-IN,en;q=0.9",
        },
    ) as client:
        for attempt in range(1, settings.http_retries + 1):
            try:
                resp = client.get(spec.url)
                status = resp.status_code
                if status == 200 and resp.text.strip():
                    cache.write_text(resp.text, encoding="utf-8")
                    return FetchResult(spec, True, 200, resp.text, False)
                last_error = f"HTTP {status}"
                if status < 500 and status != 429:
                    break
            except Exception as exc:  # noqa: BLE001 - transport errors are expected
                last_error = type(exc).__name__
            if attempt < settings.http_retries:
                time.sleep(delay)
                delay *= 2

    return FetchResult(spec, False, status, "", False, last_error or "unknown error")


# ── Markdown output ─────────────────────────────────────────────────────────

_FRONT_MATTER_ORDER = ("title", "url", "scheme", "category", "page_role", "fetched_at")


def process_page(spec: SourceSpec, html: str, fetched_at: str | None = None) -> str:
    """Extract, clean, and render one page as front-matter markdown.

    Layout is deliberate, because Stage 2 chunks on headings:

        ---
        front-matter
        ---
        ## Key facts              <- structured, labelled, deterministic (scheme pages only)
        ## Overview               <- the scheme objective
        <narrative>               <- visible text, performance tables removed

    Putting the fact block first and under its own ``##`` heading means the structure-aware
    chunker keeps it as one intact chunk, so "what is the expense ratio" retrieves the whole
    block rather than a fragment of a split table.
    """
    from .scheme_facts import render_fact_block, scheme_payload

    payload = scheme_payload(html)
    sections: list[str] = []

    if payload:
        facts = render_fact_block(payload)
        if facts:
            sections.append(facts.rstrip())

    narrative = clean_text(extract_main_text(html, spec.url))
    narrative = strip_performance_tables(narrative)
    if narrative:
        sections.append("## Overview" + ("\n\n" + narrative if narrative else ""))

    body = "\n\n".join(s for s in sections if s.strip()).strip()

    stamp = fetched_at or date.today().isoformat()
    lines = ["---"]
    for key in _FRONT_MATTER_ORDER:
        value = stamp if key == "fetched_at" else getattr(spec, key)
        # Quote anything that could break the YAML block.
        lines.append(f'{key}: "{value}"')
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def ingest_all(refresh: bool = False, verbose: bool = True) -> IngestSummary:
    """Fetch, extract, and clean every registered source. Never raises for a single page."""
    summary = IngestSummary(requested=len(SOURCES))
    stamp = date.today().isoformat()
    fetched_any = False

    for i, spec in enumerate(SOURCES, start=1):
        result = fetch_page(spec, refresh=refresh)
        row = {
            "slug": spec.slug,
            "url": spec.url,
            "title": spec.title,
            "scheme": spec.scheme,
            "category": spec.category,
            "page_role": spec.page_role,
            "fetched_at": stamp,
            "http_status": result.http_status,
            "char_count": 0,
        }

        if result.ok and result.html:
            markdown = process_page(spec, result.html, stamp)
            # Measure the body only — front-matter is metadata, already counted in the CSV
            # by other columns, and would inflate char_count against the low_text threshold.
            body = markdown.split("---", 2)[-1].lstrip("\n")
            char_count = len(body)
            row["char_count"] = char_count

            if result.from_cache:
                summary.cached += 1
            else:
                summary.fetched += 1
                time.sleep(settings.fetch_delay_seconds)
            fetched_any = True

            if char_count < settings.low_text_threshold:
                summary.low_text.append(spec.slug)

            processed_path(spec).write_text(markdown, encoding="utf-8")
        else:
            summary.failed += 1
            # Leave any stale processed file in place but make the failure impossible to miss.
            if verbose:
                print(f"  [{i:>2}/{len(SOURCES)}] FAIL {spec.slug}: {result.error}")

        if verbose:
            tag = "cache" if result.from_cache else ("ok   " if result.ok else "FAIL ")
            print(
                f"  [{i:>2}/{len(SOURCES)}] {tag} {spec.slug:<44} "
                f"{row['char_count']:>6} chars"
            )
        summary.rows.append(row)

    if not fetched_any:
        raise RuntimeError(
            f"All {len(SOURCES)} source pages failed to fetch. Check network access and "
            "settings.user_agent, then retry with --refresh."
        )

    write_sources_csv(summary.rows)
    return summary


@dataclass
class ProcessedDoc:
    """One ingested page, ready for STAGE 2."""

    slug: str
    url: str
    title: str
    scheme: str
    category: str
    page_role: str
    fetched_at: str
    body: str


def load_processed_docs() -> list[ProcessedDoc]:
    """Read every processed markdown file as a :class:`ProcessedDoc`.

    STAGE 2's input. Pages that failed ingestion are simply absent — a missing page must not
    silently become an empty chunk.
    """
    docs: list[ProcessedDoc] = []
    for spec in SOURCES:
        path = processed_path(spec)
        if not path.exists():
            continue
        markdown = path.read_text(encoding="utf-8")
        parts = markdown.split("---", 2)
        body = parts[2].lstrip("\n") if len(parts) >= 3 else markdown
        stamp = ""
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if line.startswith("fetched_at:"):
                    stamp = line.split(":", 1)[1].strip().strip('"')
        docs.append(
            ProcessedDoc(
                slug=spec.slug,
                url=spec.url,
                title=spec.title,
                scheme=spec.scheme,
                category=spec.category,
                page_role=spec.page_role,
                fetched_at=stamp,
                body=body,
            )
        )
    return docs


if __name__ == "__main__":
    print(ingest_all(refresh="--refresh" in __import__("sys").argv))
