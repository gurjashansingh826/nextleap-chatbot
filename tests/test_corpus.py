"""Corpus gates for STAGE 1 and STAGE 2.

These are the two checks that decide whether the corpus is fit to build a RAG system on. They
live in ``tests/`` rather than in a script so they run in CI and fail loudly if a future change
to the loader, the fact extractor, or the chunker breaks them.

Run with::

    python -m pytest tests -q

Both gates are *derived from the real corpus*, not from a hand-written expectation list, so
they keep working as the source pages change.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mf_rag.chunkers import build_chunks
from mf_rag.config import settings
from mf_rag.loaders import clean_text, extract_main_text, raw_path, processed_path
from mf_rag.scheme_facts import (
    GRADED_FACT_LABELS,
    extract_facts,
    scheme_payload,
)
from mf_rag.sources import SOURCES

PRIMARY = [s for s in SOURCES if s.page_role == "primary"]
SCHEME_PAGES = [s for s in SOURCES if s.page_role in ("primary", "variant")]

#: Lock-in is only meaningful for a tax-saving scheme, so its absence elsewhere is correct.
LOCK_IN_RELEVANT = lambda slug: "elss" in slug  # noqa: E731


@pytest.fixture(scope="module")
def chunks():
    built, stats = build_chunks()
    return built, stats


# ── Gate 1 · data inspection ─────────────────────────────────────────────────


@pytest.mark.parametrize("spec", PRIMARY, ids=lambda s: s.slug)
def test_gate1_every_primary_page_ingested(spec):
    """A primary scheme page must have produced a processed document."""
    assert processed_path(spec).exists(), f"{spec.slug} was never ingested"
    assert raw_path(spec).exists(), f"{spec.slug} has no cached HTML"


@pytest.mark.parametrize("spec", PRIMARY, ids=lambda s: s.slug)
def test_gate1_primary_carries_every_applicable_graded_fact(spec):
    """The seven graded facts must be present for all five in-scope schemes.

    This is the check that closes risk F4. A text-only extractor reported every one of these
    as absent, because Groww renders them only inside the ``__NEXT_DATA__`` payload.
    """
    payload = scheme_payload(
        raw_path(spec).read_text(encoding="utf-8", errors="replace")
    )
    assert payload, f"{spec.slug}: no __NEXT_DATA__ payload found"
    present = {label for label, _ in extract_facts(payload)}

    missing = []
    for label in GRADED_FACT_LABELS:
        if label in present:
            continue
        if label == "Lock-in period" and not LOCK_IN_RELEVANT(spec.slug):
            continue  # not applicable to a non-ELSS scheme
        missing.append(label)
    assert not missing, f"{spec.slug} is missing graded facts: {missing}"


def test_gate1_no_page_is_silent():
    """Every registered page with an HTTP 200 must have produced text."""
    import csv

    with settings.data_dir.joinpath("sources.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ok_rows = [r for r in rows if r["http_status"] == "200"]
    assert ok_rows, "sources.csv records no successful fetches"
    empty = [r["slug"] for r in ok_rows if int(r["char_count"]) < settings.low_text_threshold]
    assert not empty, f"pages fetched OK but yielded no usable text: {empty}"


def test_gate1_numbers_are_not_rewritten():
    """clean_text must never reformat a numeric value (operating rule R2).

    The regex substitutions in clean_text target punctuation and whitespace. If one of them
    ever grew a digit, a fee would silently change. This asserts the invariant directly.
    """
    sample = (
        "Expense ratio 1.03% and 0.84%; min SIP \u20b9100 \u2013 1,00,000; "
        "exit load 1% within 1 year\u2014total 5.25%."
    )
    out = clean_text(sample)
    for token in ("1.03%", "0.84%", "\u20b9100", "1,00,000", "1%", "5.25%"):
        assert token in out, f"clean_text altered or dropped {token!r}: {out!r}"
    # The unicode dash is normalised to ASCII, which is the intended behaviour.
    assert "\u2013" not in out and "-" in out


# ── Gate 2 · chunking ────────────────────────────────────────────────────────


def test_gate2_every_chunk_has_provenance(chunks):
    """P1 / FR-4: no chunk exists without a source URL."""
    built, _ = chunks
    orphans = [c.chunk_id for c in built if not c.source_url]
    assert not orphans, f"chunks without a source_url: {orphans}"


def test_gate2_chunk_ids_unique(chunks):
    built, _ = chunks
    ids = [c.chunk_id for c in built]
    assert len(set(ids)) == len(ids), "chunk_ids are not unique"


def test_gate2_chunks_are_self_describing(chunks):
    """Every chunk opens with ``scheme — section`` so it is citable on its own (P5)."""
    built, _ = chunks
    bad = [c.chunk_id for c in built if " \u2014 " not in c.text.split("\n")[0]]
    assert not bad, f"chunks missing the scheme/section prefix: {bad}"


def test_gate2_no_chunk_exceeds_hard_ceiling(chunks):
    """A chunk far past ``chunk_size`` signals a failed split rather than a long section."""
    built, _ = chunks
    limit = settings.chunk_size * 2
    over = [(c.chunk_id, c.char_len) for c in built if c.char_len > limit]
    assert not over, f"chunks exceed {limit} chars, splitting likely failed: {over}"


def test_gate2_fact_groups_survive_the_length_filter(chunks):
    """Regression guard.

    A labelled fact group is a complete answer to a question class. An earlier version applied
    the prose minimum-length rule to fact groups as well, which silently discarded Minimum
    investments, Risk and benchmark, NAV and fund size, and Scheme objective — i.e. the
    minimum SIP, the riskometer and the benchmark, three of the graded facts.
    """
    built, _ = chunks
    sections = {c.section for c in built if c.kind == "facts"}
    for required in (
        "Key facts: Fees and charges",
        "Key facts: Minimum investments",
        "Key facts: Risk and benchmark",
        "Key facts: Scheme identity",
        "Key facts: NAV and fund size",
    ):
        assert required in sections, f"fact group {required!r} was dropped by the filter"


@pytest.mark.parametrize("spec", PRIMARY, ids=lambda s: s.slug)
def test_gate2_graded_facts_are_retrievable_from_chunks(spec, chunks):
    """Each graded fact must be present in some chunk of its own page.

    Measured on the emitted chunks rather than the raw payload, so it also proves the
    chunker did not drop or mangle the fact.
    """
    built, _ = chunks
    blob = "\n".join(c.text for c in built if c.source_url == spec.url)
    assert blob, f"{spec.slug} produced no chunks"
    for label in GRADED_FACT_LABELS:
        if label == "Lock-in period" and not LOCK_IN_RELEVANT(spec.slug):
            continue
        assert f"{label}:" in blob, f"{spec.slug}: {label!r} not retrievable from any chunk"


def test_gate2_build_is_deterministic(chunks):
    """Re-chunking the same processed files must give a byte-identical JSONL."""
    built, _ = chunks
    first = (settings.chunks_dir / "chunks.jsonl").read_bytes()
    build_chunks()
    second = (settings.chunks_dir / "chunks.jsonl").read_bytes()
    assert first == second, "chunking is not deterministic across runs"


def test_gate2_jsonl_matches_dataclass(chunks):
    built, stats = chunks
    rows = [
        json.loads(line)
        for line in (settings.chunks_dir / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == len(built) == stats["chunks"]
    required = {
        "chunk_id", "text", "source_url", "title", "scheme", "category",
        "page_role", "section", "char_len", "fetched_at", "also_seen_at",
        "strategy", "kind",
    }
    for row in rows:
        assert required <= set(row), f"row missing keys: {required - set(row)}"


# ── The project's hardest constraint · no performance claims ──────────────────

#: Contexts in which a nearby percentage is a FEE, not a RETURN.
FEE_CONTEXT = re.compile(
    r"exit\s*load|entry\s*load|load\s*of|redeem|redemption|expense|\bter\b|"
    r"stamp\s*duty|charge|fee|turnover|lock[-\s]?in|multiplier",
    re.I,
)
#: Official index names that legitimately contain the word "Return".
INDEX_NAME = re.compile(
    r"(total\s+return(\s+index)?\b|\btri\b|price\s+return|index\s+return|"
    r"nifty\s+\d+\s+(total\s+return|price\s+return))",
    re.I,
)
PERIOD_PCT = re.compile(
    r"(?:1\s*(?:y\b|year)|3\s*(?:y\b|year)|5\s*(?:y\b|year)|10\s*(?:y\b|year)|ytd\b|"
    r"since\s+inception)[^\n]{0,40}?[+\-\u2212]?\d+(?:\.\d+)?\s*%",
    re.I,
)
ABS_FIGURE = re.compile(
    r"\b(?:cagr|annualised|annualized)\b[^\n]{0,30}?\d+(?:\.\d+)?\s*%"
    r"|\breturns?\s*\([^)]*\)[^\n]{0,30}?\d"
    r"|\brank(?:ed|ing)?\b[^\n]{0,30}?\d+\s*(?:st|nd|rd|th)\b",
    re.I,
)
TABLE_HEADER = re.compile(r"^\s*\|.*\b(?:returns?|cagr|ytd)\b.*\|\s*$", re.I | re.M)


def perf_leaks(text: str) -> list[str]:
    """Report performance *figures* in `text`. The word "returns" alone is not a leak."""
    reasons = []
    for pattern, label in ((PERIOD_PCT, "period-bound return figure"),
                           (ABS_FIGURE, "CAGR / ranking figure")):
        for m in pattern.finditer(text):
            window = text[max(0, m.start() - 60):m.end() + 60]
            if FEE_CONTEXT.search(window) or INDEX_NAME.search(window):
                continue
            reasons.append(label)
    if TABLE_HEADER.search(text):
        reasons.append("returns table header")
    return reasons


@pytest.mark.parametrize("sample", [
    "| 3 years | \u20b91,80,000 | \u20b91,83,700 | | +2.06% |",
    "The 3-year CAGR of the fund was 12.4% as on 25-Sep-2026.",
    "This fund was ranked 3rd among its peers.",
    "YTD returns stood at +8.12% for the period.",
])
def test_perf_check_catches_a_real_leak(sample):
    """Positive control — without this the negative tests below prove nothing."""
    assert perf_leaks(sample), f"perf check failed to detect a leak in {sample!r}"


@pytest.mark.parametrize("sample", [
    "- Exit load: 3 years: Exit load of 3% if redeemed within 1 year, 2% if after",
    "- Benchmark: NIFTY 100 Total Return Index (NIFTY 100 TRI)",
    "- Expense ratio (TER, direct plan): 1.03%",
    "- Lock-in period: 3 years",
])
def test_perf_check_spares_graded_facts(sample):
    """Negative control — a leak check that flags real facts destroys graded data."""
    assert not perf_leaks(sample), f"perf check wrongly flagged {sample!r}"


def test_no_chunk_states_a_performance_figure(chunks):
    built, _ = chunks
    leaks = [(c.chunk_id, perf_leaks(c.text)) for c in built]
    leaks = [(cid, r) for cid, r in leaks if r]
    assert not leaks, f"performance data reached the corpus: {leaks}"


def test_scheme_pages_carry_no_performance_vocabulary(chunks):
    """After removing official index names, a scheme chunk must not mention returns."""
    built, _ = chunks
    vocab = re.compile(r"\b(returns?|cagr|peers?|rank(?:ed|ing)?|ytd)\b", re.I)
    offenders = []
    for c in built:
        if c.page_role not in ("primary", "variant"):
            continue
        scrubbed = INDEX_NAME.sub("", c.text)
        if vocab.search(scrubbed):
            offenders.append(c.chunk_id)
    assert not offenders, f"scheme chunks mention performance data: {offenders}"


# ── Stage 1 extraction behaviour ─────────────────────────────────────────────


def test_extraction_is_deterministic():
    """The same cached HTML must always yield the same text (NFR-6)."""
    spec = PRIMARY[0]
    html = raw_path(spec).read_text(encoding="utf-8", errors="replace")
    runs = {clean_text(extract_main_text(html, spec.url)) for _ in range(3)}
    assert len(runs) == 1, "extraction is non-deterministic for identical input"


def test_cache_first_ingestion_does_not_refetch():
    """ADR-7: with a warm cache, ingestion must not hit the network."""
    from mf_rag.loaders import fetch_page

    spec = PRIMARY[0]
    result = fetch_page(spec, refresh=False)
    assert result.from_cache, "cache was not used"
    assert result.ok and result.html
