"""STAGE 6 — answerer tests.

The answerer holds the last line of defence. Even with an LLM in the loop, an answer is only
allowed out if it survives seven checks, so these tests feed it text that *should* be rejected
and text that must be allowed through.

Each post-check gets a positive control (the bad thing is caught) and a negative control (the
good thing is not collateral damage). The negative controls are the ones that usually go
missing: a pipeline tuned to reject everything passes every positive test and is useless.
"""

from __future__ import annotations

import pytest

from mf_rag import answerer as answerer_mod
from mf_rag.answerer import (
    _ADVICE_RE,
    answer_question,
    extractive_fallback,
    post_check,
)
from mf_rag.config import settings
from mf_rag.retriever import RetrievedChunk

URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
AS_OF = "2026-09-28"


# ── the one-shot widening ─────────────────────────────────────────────────────


def _pool_chunk(i: int, text: str) -> RetrievedChunk:
    """A chunk for the widening tests. Named apart from the two ``_chunk`` helpers below."""
    return RetrievedChunk(
        chunk_id=f"c{i}", text=text, source_url=URL, title="HDFC Large Cap Fund",
        scheme="hdfc-large-cap-fund-direct-growth", category="Large cap",
        page_role="primary", plan="Direct Growth", section="Key facts: NAV and fund size",
        char_len=len(text), fetched_at=AS_OF, score=1.0 - i / 100,
    )


def test_declining_widens_the_pool_once_before_giving_up(monkeypatch):
    """A field at rank 13 must not alternate between answered and declined.

    Rank in an approximate index is a property of the graph, not of the question, so a fact
    the corpus plainly contains can fall in and out of a fixed-size pool. One extra retrieval
    removes that whole class of non-determinism.
    """
    shallow = [_pool_chunk(i, "- Minimum SIP investment: ₹100") for i in range(12)]
    deep = shallow + [_pool_chunk(12, "- Fund manager: Prashant Jain")]

    depths = []

    def fake_retrieve(query, top_k=None, **kwargs):
        depths.append(top_k)
        return deep if top_k and top_k >= answerer_mod._MAX_POOL else shallow

    monkeypatch.setattr(answerer_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(answerer_mod, "call_groq", lambda *a, **k: None)

    answer = answer_question("Who manages HDFC Large Cap Fund?")
    assert "Prashant" in answer.text
    assert any("widened the search" in n for n in answer.notes)
    assert depths == [max(settings.top_k, settings.extractive_pool), answerer_mod._MAX_POOL]


def test_widening_stops_when_nothing_matches_anywhere(monkeypatch):
    """A genuinely absent field must still decline, and must not widen forever."""
    calls = []

    def fake_retrieve(query, top_k=None, **kwargs):
        calls.append(top_k)
        return [_pool_chunk(i, "- Minimum SIP investment: ₹100") for i in range(30)]

    monkeypatch.setattr(answerer_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(answerer_mod, "call_groq", lambda *a, **k: None)

    answer = answer_question("What is the Sharpe ratio of HDFC Large Cap Fund?")
    assert answer.mode == "not_in_sources"
    assert answer.citations == []
    assert len(calls) == 2, "widening must happen at most once"

CHUNK_BODY = (
    "HDFC Large Cap Fund (Direct Growth) — Key facts: Fees and charges\n"
    "- Expense ratio (TER): 1.03%\n"
    "- Base expense ratio (excl. additional fund expenses): 0.84%\n"
    "- Exit load: Exit load of 1% if redeemed within 1 year\n"
)


def _chunk(**over) -> RetrievedChunk:
    base = dict(
        chunk_id="k1",
        text=CHUNK_BODY,
        source_url=URL,
        title="HDFC Large Cap Fund – Direct Growth",
        scheme="HDFC Large Cap Fund",
        category="Large cap",
        page_role="scheme",
        plan="Direct Growth",
        section="Key facts: Fees and charges",
        char_len=len(CHUNK_BODY),
        fetched_at=AS_OF,
        score=0.86,
        filter_reason="scheme + plan matched (Direct Growth)",
    )
    base.update(over)
    return RetrievedChunk(**base)


# ── check 0: empty output ──────────────────────────────────────────────────────


def test_empty_text_is_rejected() -> None:
    assert post_check("", [_chunk()], "q", AS_OF) is None
    assert post_check("   ", [_chunk()], "q", AS_OF) is None


# ── check 1: a citation is required ───────────────────────────────────────────


def test_answer_without_a_citation_is_rejected() -> None:
    result = post_check("The exit load is 1% within 1 year.", [_chunk()], "q", AS_OF)
    assert result is None, "an uncited factual answer must never be returned"


# ── check 2: the citation must be real ────────────────────────────────────────


def test_invented_source_is_rejected() -> None:
    result = post_check(
        "The exit load is 1%. Source: https://example.com/made-up-fund",
        [_chunk()],
        "q",
        AS_OF,
    )
    assert result is None, "a citation outside sources.csv must be rejected"


def test_second_citation_is_dropped() -> None:
    """Rule 3 is one source link per answer, so a second must be trimmed rather than kept."""
    other = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
    result = post_check(
        f"Exit load is 1%. Source: {URL}\nAlso see {other}",
        [_chunk()],
        "q",
        AS_OF,
    )
    assert result is not None
    assert result.citations == [URL]
    assert other not in result.text, "the second citation should have been stripped"


# ── check 3: at most three sentences ───────────────────────────────────────────


def test_long_answer_is_truncated_and_keeps_its_citation() -> None:
    long_text = (
        f"Expense ratio is 1.03%. Exit load is 1%. Minimum SIP is ₹100. "
        f"Benchmark is NIFTY 50. Source: {URL}"
    )
    result = post_check(long_text, [_chunk()], "q", AS_OF)
    assert result is not None
    assert len(_sentence_count(_body_of(result.text))) <= settings.max_sentences
    assert URL in result.text, "truncation must not drop the citation"
    assert any("check 3" in n for n in result.notes), "truncation should be recorded in notes"


def _body_of(text: str) -> str:
    """The answer proper, without the source line, footer and disclaimer.

    Check 3 bounds the answer, not the boilerplate bolted underneath it, so measuring the
    rendered text would fail on a correct implementation.
    """
    for marker in ("\nSource:", "\nLast updated from sources:"):
        if marker in text:
            text = text.split(marker)[0]
    return text


def _sentence_count(text: str) -> list[str]:
    from mf_rag.answerer import _sentences

    return _sentences(text)


# ── check 4: advice language ───────────────────────────────────────────────────


def test_advice_in_the_answer_becomes_a_refusal() -> None:
    result = post_check(
        f"You should invest in this fund because it suits you. Source: {URL}",
        [_chunk()],
        "q",
        AS_OF,
    )
    assert result is not None
    assert result.mode == "refusal"
    assert result.guard == "advice"
    assert not result.citations, "a refusal must not carry a citation"


# ── check 5: performance claims ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "claim",
    [
        "HDFC Large Cap Fund returned 22.4% in the last year.",
        "Its 5-year CAGR is 18.2%.",
        "This fund has delivered 30% returns since launch.",
    ],
)
def test_performance_claim_in_the_answer_becomes_a_refusal(claim: str) -> None:
    result = post_check(f"{claim} Source: {URL}", [_chunk()], "q", AS_OF)
    assert result is not None
    assert result.mode == "refusal"
    assert result.guard == "returns"


def test_published_fees_are_not_mistaken_for_performance() -> None:
    """The negative control for check 5, and the most important one.

    Every factual answer in this system is a number, and a naive return-detector that fires on
    "1.03%" would refuse every question the bot exists to answer. Fees, ratios, lock-ins and
    minimums must pass.
    """
    result = post_check(
        f"The expense ratio (TER) is 1.03% and the exit load is 1% within 1 year. "
        f"The lock-in period is 3 years. Source: {URL}",
        [_chunk()],
        "q",
        AS_OF,
    )
    assert result is not None, "published fee facts were wrongly refused"
    assert result.mode == "llm"


# ── checks 6 and 7: footer and disclaimer ──────────────────────────────────────


def test_footer_and_as_of_are_present_on_every_accepted_answer() -> None:
    result = post_check(f"Exit load is 1%. Source: {URL}", [_chunk()], "q", AS_OF)
    assert result is not None
    assert f"Last updated from sources: {AS_OF}" in result.text
    assert result.as_of == AS_OF


# ── the deterministic fallback ─────────────────────────────────────────────────


def test_fallback_quotes_a_real_chunk_and_cites_it() -> None:
    answer = extractive_fallback([_chunk()], "exit load of HDFC Large Cap Fund")
    assert answer.mode == "extractive"
    assert answer.citations == [URL]
    assert "1%" in answer.text
    assert f"Last updated from sources: {AS_OF}" in answer.text


def test_fallback_answers_the_field_that_was_asked_about() -> None:
    """The label tie-break, tested because the failure is silent and looks like a fact.

    The page publishes both "Expense ratio (TER): 1.03%" and "Base expense ratio
    (excl. additional fund expenses): 0.84%". Answering the first with the second is not a
    crash — it is a wrong number, correctly formatted, with a real citation. That is the
    hardest class of bug in this project to catch, so it gets an explicit test.
    """
    answer = extractive_fallback([_chunk()], "what is the expense ratio")
    assert "Expense ratio (TER): 1.03%" in answer.text
    assert "Base expense ratio" not in answer.text


def test_fallback_does_not_drag_in_a_matching_line() -> None:
    answer = extractive_fallback([_chunk()], "exit load")
    assert "Exit load" in answer.text
    assert "expense ratio" not in answer.text.lower()


def test_fallback_is_bounded_by_max_sentences() -> None:
    answer = extractive_fallback([_chunk()], "expense ratio exit load")
    assert len(_sentence_count(_body_of(answer.text))) <= settings.max_sentences


def test_a_quoted_regulation_is_not_mistaken_for_advice() -> None:
    """SEBI's 35/35 rule is a published fact, and the model quotes it in that exact form.

    A bare ``must invest`` rejected this, so a correct answer became a refusal — the guard
    discarded real evidence because the corpus happens to contain the word "invest". Advice is
    addressed to the reader ("you must invest"); a regulation is addressed to the fund ("these
    funds must invest"), and conflating them is what made the guard reject the fact.
    """
    regulation = (
        "According to SEBI guidelines, these funds must invest at least 35% of their "
        "total assets each in large-cap and mid-cap stocks."
    )
    assert not _ADVICE_RE.search(regulation), "a published regulation was read as advice"

    # The negative control: advice in every shape the guard exists to catch must still fire.
    for genuine_advice in (
        "You should invest in this fund",
        "You must invest in a large and mid cap fund",
        "I recommend this scheme",
        "This is advisable for you",
    ):
        assert _ADVICE_RE.search(genuine_advice), genuine_advice


def test_fallback_with_no_chunks_declines_rather_than_guessing() -> None:
    answer = extractive_fallback([], "what is the expense ratio")
    assert answer.mode == "not_in_sources"
    assert not answer.citations
    assert answer.text.strip()


# ── guards run before anything else ────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "Should I buy HDFC Small Cap Fund?",
        "Which HDFC fund has the highest returns?",
        "my PAN is ABCDE1234F",
    ],
)
def test_answer_question_refuses_without_calling_the_llm(question: str) -> None:
    """A guard must short-circuit before retrieval, so this needs no index to pass."""
    answer = answer_question(question)
    assert answer.guard is not None, f"no guard fired for: {question!r}"
    assert answer.mode.endswith("refusal"), f"mode was {answer.mode!r}"
    assert not answer.citations, "a refusal must not present a source as if it were a fact"
