"""The generic-field-word failure, and the facts that must survive fixing it.

Every test here exists because a wrong fact was returned under a correct citation, which
is the failure mode this project is built to make impossible. A wrong fact with a
legitimate-looking source link is worse than an unhelpful one, because the user has no
way to tell — so these are the tests that matter most in the suite.
"""

from __future__ import annotations

import pytest

from mf_rag.answerer import _GENERIC_FIELD_STEMS, _pick_lines, _stems, extractive_fallback
from mf_rag.prompts import not_in_sources_message
from mf_rag.retriever import RetrievedChunk, retrieve

#: The decline text, stripped out below so a test can never pass because the expected
#: string appeared in a decline message rather than in a quoted fact.
_CORPUS_NOTE = not_in_sources_message("").split(".")[0]


def _lines_for(question: str) -> list[str]:
    """What the extractive path would quote for ``question``, or ``[]`` if it declines.

    Only the quoted lines count. The citation, the as-of footer and the out-of-scope notice
    are stripped, so a test can never pass because the expected string appeared in a URL or
    in a decline message rather than in a quoted fact.

    The pool depth mirrors ``answer_question`` (``extractive_pool``, not ``top_k``). Using
    ``top_k`` here makes the fund-manager questions decline, because the "Scheme identity"
    group sits around 13th — which is the very reason the deeper pool exists, so a test that
    ignored it would be testing a configuration the app never runs.
    """
    from mf_rag.config import settings

    depth = max(settings.top_k, settings.extractive_pool)
    answer = extractive_fallback(retrieve(question, top_k=depth), question)
    quoted = []
    for line in answer.text.splitlines():
        line = line.strip()
        if not line or line.startswith(("Source:", "Last updated from sources:")):
            continue
        if line == _CORPUS_NOTE:
            continue
        quoted.append(line.lstrip("- ").strip())
    return quoted


# ── the two live failures ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question, wrong_answer",
    [
        # "ratio" must not be enough to reach the expense-ratio line.
        ("What is the Sharpe ratio of HDFC Large Cap Fund?", "expense ratio"),
        # "plan" must not be enough to reach the plan-type line.
        ("Who manages HDFC Focused Large Cap Direct Plan?", "plan type"),
    ],
)
def test_generic_field_word_does_not_produce_a_wrong_fact(question, wrong_answer):
    quoted = " | ".join(_lines_for(question)).lower()
    assert wrong_answer not in quoted, (
        f"answered with an unrelated field for {question!r}: {quoted!r}"
    )


def test_a_question_naming_no_field_declines():
    """Every query term is generic, so nothing in the chunk can be shown to be on-topic."""
    from mf_rag.answerer import _stems

    assert _stems("what is the ratio") - _GENERIC_FIELD_STEMS == set()


# ── partial overlap is a coincidence, not an answer ───────────────────────────


@pytest.mark.parametrize(
    "question, wrong_answer",
    [
        # "SIP" alone matched "- SIP available: Yes" while the page that answers the
        # question ranked 1st at 0.841 and the quoted scheme page ranked 25th.
        ("what is the difference between ELSS and SIP", "sip available"),
        # "Mutual Fund" alone matched "- Fund house: HDFC Mutual Fund".
        ("what are the steps to invest in HDFC Mutual Funds on Groww", "fund house"),
    ],
)
def test_partial_term_overlap_does_not_produce_a_wrong_fact(question, wrong_answer):
    """A line must account for *every* content term, not just one of them.

    Both of these are the worst kind of failure for a facts-only bot: a real field, quoted
    verbatim, under a real citation, with nothing visible on screen to show it is wrong.
    The single shared word ("SIP", "Mutual Fund") is a common word in field labels, not
    evidence that the line answers the question.
    """
    quoted = " | ".join(_lines_for(question)).lower()
    assert wrong_answer not in quoted, (
        f"a one-word overlap produced an unrelated field for {question!r}: {quoted!r}"
    )


def test_partial_overlap_rule_still_admits_multi_term_answers():
    """The negative control: full coverage must keep working, or the rule is a blanket ban.

    "exit load" and "minimum SIP" are both two-term questions whose answer legitimately
    covers both terms, so requiring complete coverage cannot be a way of making the bot
    decline everything.
    """
    assert _lines_for("What is the exit load of HDFC Large Cap Fund?")
    assert _lines_for("What is the minimum SIP for HDFC ELSS Tax Saver Fund?")


# ── the facts the fix must not break ──────────────────────────────────────────


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is the expense ratio of HDFC Large Cap Fund Direct Growth?", "1.03"),
        ("Who manages HDFC Large Cap Fund?", "prashant jain"),
        ("Who manages HDFC Small Cap Fund?", "chirag setalvad"),
        ("Who manages HDFC ELSS Tax Saver Fund?", "vinay kulkarni"),
        ("Is HDFC ELSS Tax Saver Fund locked in?", "3 years"),
        ("What is the exit load of HDFC Large Cap Fund?", "1%"),
        ("What is the NAV of HDFC Equity Fund?", "nav"),
        ("What is the benchmark of HDFC Small Cap Fund?", "bse 250"),
        ("What is the riskometer level of HDFC Large Cap Fund?", "moderately high"),
        ("What is the minimum SIP for HDFC ELSS Tax Saver Fund?", "500"),
        ("What is the minimum lump sum for HDFC Equity Fund?", "100"),
    ],
)
def test_distinguishing_field_words_still_match(question, expected):
    quoted = " | ".join(_lines_for(question)).lower()
    assert expected in quoted, f"{question!r} lost its field: {quoted!r}"


def test_every_labelled_field_on_a_scheme_page_is_reachable():
    """A guard against over-pruning: no distinguishing stem in the generic set.

    If a field's own name were listed as generic, that field would become unanswerable and
    the failure would be silent — a decline, which looks like the system being careful.

    "growth" is deliberately absent from this list: it is a plan descriptor, not a field.
    Which plan is in scope is the entity filter's job (ADR-14), and suppressing the word
    keeps "HDFC Large Cap Fund Direct **Growth**" from being treated as a field request.
    """
    # The words that identify a specific field on a scheme page. None may be suppressed.
    for field in (
        "expense", "benchmark", "lock", "exit", "manager", "sip", "lump", "turnover",
        "riskometer", "nav", "sharpe", "minimum", "sebi", "aum", "identity", "objective",
    ):
        stems = _stems(field)
        assert stems, field
        assert not (stems & _GENERIC_FIELD_STEMS), (
            f"{field!r} is suppressed as a generic field word, so questions about it "
            f"would silently decline"
        )


def test_pick_lines_returns_none_rather_than_a_weak_match():
    """The unit-level contract: no distinguishing term in the chunk means None, not a guess."""
    chunk = RetrievedChunk(
        chunk_id="t1",
        text="- Expense ratio (TER): 1.03%\n- Plan type: Direct",
        source_url="https://groww.in/x",
        title="HDFC Large Cap Fund - Direct Growth",
        scheme="hdfc-large-cap-fund-direct-growth",
        category="Large cap",
        page_role="primary",
        plan="Direct Growth",
        section="Key facts: Fees and charges",
        char_len=52,
        fetched_at="2026-09-28",
        score=0.9,
    )
    assert _pick_lines(chunk, "What is the Sharpe ratio of HDFC Large Cap Fund?") is None
    assert _pick_lines(chunk, "Who manages HDFC Focused Large Cap Direct Plan?") is None
    # and the real question still works
    assert _pick_lines(chunk, "What is the expense ratio?")
