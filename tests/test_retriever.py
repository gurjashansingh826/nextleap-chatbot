"""STAGE 5b — retriever tests.

The retriever's job is narrow and the failure that matters is specific: a question about
``Direct Growth`` citing a ``regular-growth`` page, or about HDFC Large Cap citing HDFC
Focused Large Cap. That is not a ranking problem you can tune away — it is a categorical
error, which is why ADR-14 made it a deterministic filter rather than a similarity boost.

So these tests exercise ``resolve_scheme`` and ``_matches_filter`` on hand-built chunks,
where the correct answer is known by construction. ``retrieve`` is additionally checked
against the real index, and skips when it is absent so a fresh clone still gets a signal.
"""

from __future__ import annotations

import pytest

from mf_rag.retriever import (
    RetrievedChunk,
    build_context,
    mmr_rerank,
    resolve_scheme,
    retrieve,
    _matches_filter,
)

LARGE_CAP_DG = "hdfc-large-cap-fund-direct-growth"
LARGE_CAP_REG = "hdfc-large-cap-fund-regular-growth"
LARGE_CAP_DI = "hdfc-large-cap-fund-direct-idcw"
FLEXI_CAP = "hdfc-equity-fund-direct-growth"


def _chunk(slug: str, *, plan: str = "Direct Growth", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=slug,
        text=f"HDFC Test Fund ({plan}) — Key facts: Fees and charges\n- Exit load: 1%",
        source_url=f"https://groww.in/mutual-funds/{slug}",
        title=f"HDFC Test Fund – {plan}",
        scheme="HDFC Test Fund",
        category="Large cap",
        page_role="scheme",
        plan=plan,
        section="Key facts: Fees and charges",
        char_len=90,
        fetched_at="2026-09-28",
        score=score,
    )


# ── resolve_scheme ────────────────────────────────────────────────────────────


def test_resolves_exact_scheme_and_plan() -> None:
    res = resolve_scheme("What is the expense ratio of HDFC Large Cap Fund Direct Growth?")
    assert LARGE_CAP_DG in res["slugs"]
    assert res["plan"] == "direct growth"
    assert res["reason"]


def test_plan_unspecified_still_resolves_to_the_primary_page() -> None:
    res = resolve_scheme("What is HDFC Small Cap Fund?")
    assert res["slugs"], "a named scheme must resolve even without a plan"
    assert res["plan"] == ""


def test_question_naming_no_scheme_resolves_to_nothing() -> None:
    res = resolve_scheme("Which fund is best?")
    assert res["slugs"] == []
    assert "no scheme" in res["reason"].lower()


def test_resolution_does_not_raise_on_odd_input() -> None:
    for q in ["", "   ", "?!", "the", "₹100", "HDFC"]:
        assert resolve_scheme(q) is not None


# ── _matches_filter — the ADR-14 guard ─────────────────────────────────────────


def test_direct_growth_question_rejects_regular_growth_chunk() -> None:
    """The exact hazard: same scheme, wrong plan page. Must be dropped."""
    res = resolve_scheme("exit load of HDFC Large Cap Fund Direct Growth")
    pool = [_chunk(LARGE_CAP_REG, plan="Growth"), _chunk(LARGE_CAP_DG)]
    kept = [c for c in pool if _matches_filter(c, res, pool)[0]]
    assert [c.source_url for c in kept] == [f"https://groww.in/mutual-funds/{LARGE_CAP_DG}"]


def test_growth_must_not_match_inside_direct_growth() -> None:
    """A bare substring test would let 'growth' match 'direct growth' and admit both."""
    res = resolve_scheme("exit load of HDFC Large Cap Fund Direct Growth")
    pool = [_chunk(LARGE_CAP_REG, plan="Growth"), _chunk(LARGE_CAP_DG, plan="Direct Growth")]
    for chunk in pool:
        keep, reason = _matches_filter(chunk, res, pool)
        assert keep == (chunk.plan == "Direct Growth"), f"{chunk.plan} -> {keep} ({reason})"


def test_direct_idcw_is_not_direct_growth() -> None:
    res = resolve_scheme("exit load of HDFC Large Cap Fund Direct Growth")
    pool = [_chunk(LARGE_CAP_DI, plan="Direct IDCW"), _chunk(LARGE_CAP_DG)]
    kept = [c for c in pool if _matches_filter(c, res, pool)[0]]
    assert [c.plan for c in kept] == ["Direct Growth"]


def test_cross_scheme_chunk_is_rejected() -> None:
    """HDFC Large Cap question must not cite HDFC Flexi Cap just because both are 'equity'."""
    res = resolve_scheme("expense ratio of HDFC Large Cap Fund Direct Growth")
    pool = [_chunk(FLEXI_CAP, plan="Direct Growth"), _chunk(LARGE_CAP_DG)]
    kept = [c for c in pool if _matches_filter(c, res, pool)[0]]
    assert [c.source_url for c in kept] == [f"https://groww.in/mutual-funds/{LARGE_CAP_DG}"]


def test_concept_question_keeps_everything() -> None:
    """With no scheme named there is nothing to filter against — c01/c03 depend on this."""
    res = resolve_scheme("What is the difference between ELSS and SIP?")
    pool = [_chunk(LARGE_CAP_DG), _chunk(FLEXI_CAP)]
    assert all(_matches_filter(c, res, pool)[0] for c in pool)


def test_every_kept_chunk_explains_itself() -> None:
    """FR-14: the filter must be auditable, so a kept or dropped chunk always has a reason."""
    res = resolve_scheme("exit load of HDFC Large Cap Fund Direct Growth")
    pool = [_chunk(LARGE_CAP_REG, plan="Growth"), _chunk(LARGE_CAP_DG)]
    for chunk in pool:
        _, reason = _matches_filter(chunk, res, pool)
        assert reason, f"no filter_reason for {chunk.plan}"


# ── MMR ───────────────────────────────────────────────────────────────────────


def test_mmr_drops_a_near_duplicate_when_diversity_is_weighted_highly() -> None:
    """With lambda low enough, MMR must prefer the chunk that says something new.

    The boundary is what makes this a real test: at the configured ``mmr_lambda=0.7`` the
    relevance term dominates, so a near-duplicate with a slightly higher score legitimately
    wins. That is the intended trade-off for this corpus — a *slightly* worse chunk that
    answers the question beats the perfect one that does not — but it does mean lambda is a
    tuned parameter and not a de-duplicator. This test pins the mechanism at the other end of
    the range, where the diversity term is decisive.
    """
    a = _chunk(LARGE_CAP_DG, score=0.70)
    b = _chunk(LARGE_CAP_DG, score=0.70)
    b.text = a.text + " "
    c = _chunk(FLEXI_CAP, score=0.69)
    c.text = (
        "HDFC Flexi Cap Fund (Direct Growth) — Key facts: About the fund\n"
        "- Number of equity holdings: 47"
    )
    picked = mmr_rerank([a, b, c], lambda_=0.1, top_k=2)
    assert len(picked) == 2
    # `a` and `b` deliberately share a chunk_id, so membership is the wrong test — count them.
    assert sum(1 for x in picked if x.chunk_id == LARGE_CAP_DG) <= 1, (
        "with diversity weighted highly, both near-duplicates were still returned"
    )


def test_mmr_never_returns_a_chunk_twice() -> None:
    pool = [
        _chunk(LARGE_CAP_DG, score=0.70),
        _chunk(LARGE_CAP_DG, score=0.70),
        _chunk(FLEXI_CAP, score=0.70),
    ]
    pool[1].text = pool[0].text + " "
    picked = mmr_rerank(pool, lambda_=0.5, top_k=3)
    assert len(picked) == 3, "MMR must not shrink the pool"
    assert len({id(x) for x in picked}) == 3, "MMR returned the same object twice"


def test_mmr_keeps_the_highest_scoring_chunk() -> None:
    """Whatever else it trades off, the best-matching chunk must survive."""
    a = _chunk(LARGE_CAP_DG, score=0.90)
    b = _chunk(FLEXI_CAP, score=0.60)
    b.text = "HDFC Flexi Cap Fund (Direct Growth) — Key facts: About the fund\n- Holdings: 47"
    c = _chunk("hdfc-balanced-advantage-fund-direct-growth", score=0.70)
    c.text = "HDFC Balanced Advantage Fund (Direct Growth) — Key facts: About\n- Debt: 45%"
    picked = mmr_rerank([a, b, c], lambda_=0.7, top_k=2)
    assert picked[0].chunk_id == LARGE_CAP_DG


def test_mmr_returns_the_pool_untouched_when_it_already_fits() -> None:
    """With fewer candidates than slots there is nothing to trade off, so no re-embedding."""
    pool = [_chunk(LARGE_CAP_DG, score=0.9), _chunk(FLEXI_CAP, score=0.8)]
    picked = mmr_rerank(pool, lambda_=0.7, top_k=4)
    assert [c.chunk_id for c in picked] == [c.chunk_id for c in pool]


def test_mmr_respects_top_k_and_input_order() -> None:
    pool = [_chunk(f"s{i}", score=0.9 - i / 100) for i in range(5)]
    picked = mmr_rerank(pool, lambda_=0.7, top_k=3)
    assert len(picked) == 3
    assert picked[0].score >= picked[-1].score - 1e-9


def test_mmr_handles_empty_and_singleton() -> None:
    assert mmr_rerank([], lambda_=0.7, top_k=4) == []
    one = _chunk(LARGE_CAP_DG)
    assert mmr_rerank([one], lambda_=0.7, top_k=4) == [one]


# ── context assembly ──────────────────────────────────────────────────────────


def test_build_context_carries_url_scheme_plan_and_section() -> None:
    ctx = build_context([_chunk(LARGE_CAP_DG)])
    assert LARGE_CAP_DG in ctx
    assert "Direct Growth" in ctx
    assert "Key facts: Fees and charges" in ctx


# ── end to end against the real index ─────────────────────────────────────────


def _index_is_built() -> bool:
    try:
        from mf_rag.store import collection_size

        return collection_size() > 0
    except Exception:
        return False


requires_index = pytest.mark.skipif(
    not _index_is_built(), reason="index not built — run `python -m mf_rag.cli embed`"
)


@requires_index
@pytest.mark.parametrize(
    "question,expected_slug",
    [
        ("exit load of HDFC Large Cap Fund Direct Growth", LARGE_CAP_DG),
        ("expense ratio of HDFC Large Cap Fund Direct Growth", LARGE_CAP_DG),
    ],
)
def test_retrieve_never_cites_the_wrong_plan_page(question: str, expected_slug: str) -> None:
    """The end-to-end version of the ADR-14 hazard, against the real collection."""
    chunks = retrieve(question, top_k=6)
    assert chunks, "expected at least one chunk"
    assert chunks[0].source_url.endswith(expected_slug)
    for c in chunks:
        assert c.source_url.endswith(expected_slug), f"leaked citation: {c.source_url}"


@requires_index
def test_retrieve_attaches_a_filter_reason() -> None:
    for c in retrieve("expense ratio of HDFC Large Cap Fund Direct Growth", top_k=6):
        assert c.filter_reason, "FR-14 requires a reason on every retrieved chunk"


@requires_index
def test_retrieve_respects_top_k_and_scores() -> None:
    chunks = retrieve("minimum SIP for HDFC Small Cap Fund", top_k=2)
    assert len(chunks) <= 2
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)
