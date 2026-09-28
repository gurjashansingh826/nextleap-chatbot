"""Gate 2 — eval-set integrity.

The scorecards in ``docs/eval_report.md`` are only meaningful if the labels they are computed
from are true. An eval set is hand-written, and hand-written labels rot: this repo already
caught two of them, where a ``₹`` read as ``?`` because the file was read without an explicit
UTF-8 encoding and the label silently became a different number than the corpus contains.

A wrong label does not announce itself. It lowers recall, the strategy is "fixed" to
compensate, and the real defect ships. So every in-scope ``key_fact`` is asserted against the
processed document it claims to come from. This is the test that would have caught that bug
at write time instead of three phases later.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "chunking_eval.json"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Every rubric string is compared after this normalisation. The corpus is written by hand from
# HTML and mixes bullet characters, non-breaking spaces and rupee signs; the labels must not
# depend on those.
def _norm(text: str) -> str:
    for ch in ("\u00a0", "\u2022", "\u2013", "\u2014", "\u2192", "\n", "\r", "\t"):
        text = text.replace(ch, " ")
    return " ".join(text.split()).lower()


def _load_rows() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def _doc_for(url: str) -> Path:
    slug = Path(urlparse(url).path).name
    return PROCESSED / f"{slug}.md"


IN_SCOPE = [r for r in _load_rows() if not r["id"].startswith("x")]
OUT_OF_SCOPE = [r for r in _load_rows() if r["id"].startswith("x")]


def test_eval_set_exists_and_is_not_empty() -> None:
    assert EVAL_PATH.exists(), "eval/chunking_eval.json is missing"
    assert len(_load_rows()) >= 30, "the eval set is smaller than the Gate 2 design intends"


def test_the_six_out_of_scope_probes_exist() -> None:
    """Threshold calibration needs a negative set; without it min_score is unfalsifiable."""
    assert len(OUT_OF_SCOPE) >= 6, "expected at least 6 out-of-scope probes"
    for row in OUT_OF_SCOPE:
        assert not row.get("expect_url"), f"{row['id']} is an out-of-scope probe but names a URL"


@pytest.mark.parametrize("row", IN_SCOPE, ids=[r["id"] for r in IN_SCOPE])
def test_key_fact_is_present_in_its_own_source_document(row: dict) -> None:
    """The label-drift gate. Every graded fact must be verifiable in the page it is graded against."""
    url = row.get("expect_url")
    assert url, f"{row['id']} is in scope but has no expect_url"

    doc = _doc_for(url)
    assert doc.exists(), f"{row['id']}: no processed document at {doc.name}"

    body = _norm(doc.read_text(encoding="utf-8"))
    needle = _norm(row["key_fact"])
    assert needle, f"{row['id']}: empty key_fact"
    assert needle in body, (
        f"{row['id']}: key_fact {row['key_fact']!r} is not in {doc.name}. "
        "Either the label is wrong or the corpus changed — re-verify before trusting scores."
    )


@pytest.mark.parametrize("row", IN_SCOPE, ids=[r["id"] for r in IN_SCOPE])
def test_every_in_scope_row_is_fully_specified(row: dict) -> None:
    for field in ("id", "query", "key_fact", "expect_url", "fact_type"):
        assert row.get(field), f"{row.get('id')}: missing {field}"
    assert row["fact_type"] in {"numeric", "attribute", "concept", "procedural"}, (
        f"{row['id']}: unknown fact_type {row['fact_type']!r}"
    )
    assert _doc_for(row["expect_url"]).exists(), f"{row['id']}: expect_url is not a corpus page"


def test_fact_types_cover_the_taxonomy() -> None:
    """A set made of one fact type would make recall@5 look better than the bot is."""
    seen = {r["fact_type"] for r in IN_SCOPE}
    assert seen == {"numeric", "attribute", "concept", "procedural"}, (
        f"fact_type coverage is incomplete: {sorted(seen)}"
    )


def test_ids_are_unique() -> None:
    ids = [r["id"] for r in _load_rows()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate eval ids: {sorted(dupes)}"


def test_no_performance_questions_in_the_eval_set() -> None:
    """FR: the bot must never assert returns, so the eval set must not reward answering with them.

    Only the ``query`` is checked, and only for phrasings that *ask for* a return figure.
    Matching the bare word "return" over the whole row would flag q03, whose graded fact is
    "Benchmark: NIFTY 100 Total Return Index" — the name of a published index, which is
    exactly the kind of fact the bot is supposed to be able to state.
    """
    banned = (
        "cagr",
        "best performing",
        "highest return",
        "lowest return",
        "returns last year",
        "return last year",
        "% return",
        "performance of",
        "how much did",
    )
    for row in _load_rows():
        query = row.get("query", "").lower()
        for phrase in banned:
            assert phrase not in query, (
                f"{row['id']}: asks about {phrase!r}. Return figures are out of scope by design."
            )


def test_published_index_names_are_not_treated_as_performance() -> None:
    """The negative control for the check above, so the guard is not later 'fixed' too hard."""
    rows = [r for r in _load_rows() if "Total Return Index" in r.get("key_fact", "")]
    assert rows, "expected at least one benchmark row naming a Total Return Index"
    for row in rows:
        assert row["fact_type"] == "attribute"
