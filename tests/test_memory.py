"""Tests for STAGE 5c — conversational memory.

The memory is the one component whose job is to hold on to text, so most of these tests are
about what it must *not* do: never change the guards, never persist, never keep a refusal or
anything identifiable.
"""

from __future__ import annotations

import pytest

from mf_rag.answerer import answer_question
from mf_rag.config import settings
from mf_rag.memory import ConversationMemory, Turn
from mf_rag.retriever import resolve_scheme


def _turn(slug: str = "hdfc-large-cap-fund-direct-growth", mode: str = "llm") -> Turn:
    return Turn(
        question="expense ratio of HDFC Large Cap Fund Direct Growth",
        scheme=slug,
        as_of="2026-09-28",
        mode=mode,
    )


# ── the window is bounded ────────────────────────────────────────────────────


def test_maxlen_defaults_to_configured_window():
    assert ConversationMemory().maxlen == settings.memory_turns == 10


def test_oldest_turn_is_dropped_not_summarised():
    """A summary would be generated text about what the bot said — where memories invent."""
    mem = ConversationMemory(maxlen=3)
    for i in range(5):
        mem.add(Turn(question=f"q{i}", scheme="hdfc-small-cap-fund-direct-growth", as_of="", mode="llm"))
    assert len(mem) == 3
    assert [t.question for t in mem.recent()] == ["q2", "q3", "q4"]


def test_maxlen_zero_means_no_memory_at_all():
    mem = ConversationMemory(maxlen=0)
    assert mem.add(_turn()) is False
    assert len(mem) == 0
    # And a zero window must not rewrite anything.
    assert mem.contextualise("what about its exit load?") == "what about its exit load?"


# ── what may not enter the window ────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["refusal", "pii_refusal", "not_in_sources"])
def test_refusals_are_not_recorded(mode):
    """A refusal has no subject to lend, and inheriting one is how advice leaks in.

    If "Should I buy HDFC Small Cap?" set the context, the next "what about its exit load?"
    would quietly answer about a fund the bot just refused to discuss.
    """
    mem = ConversationMemory()
    assert mem.add(_turn(mode=mode)) is False
    assert mem.last_scheme() == ""


def test_pii_is_never_recorded_even_if_the_caller_forgets_to_guard():
    mem = ConversationMemory()
    leak = Turn(question="my PAN is ABCDE1234F", scheme="hdfc-small-cap-fund-direct-growth", as_of="", mode="llm")
    assert mem.add(leak) is False
    assert len(mem) == 0


def test_add_from_answer_uses_the_redacted_question():
    mem = ConversationMemory()
    refused = answer_question("Should I buy HDFC Small Cap Fund?")
    assert refused.mode == "refusal"
    assert mem.add_from_answer(refused) is False


# ── the rewrite itself ───────────────────────────────────────────────────────


def _primed(slug: str = "hdfc-large-cap-fund-direct-growth") -> ConversationMemory:
    mem = ConversationMemory()
    mem.add(_turn(slug=slug))
    return mem


@pytest.mark.parametrize(
    "question",
    ["what about its exit load?", "and the minimum SIP", "who manages it", "its lock-in period"],
)
def test_pronominal_followups_are_resolved(question):
    mem = _primed()
    rewritten = mem.contextualise(question)
    assert rewritten != question
    assert "HDFC Large Cap Fund" in rewritten
    # And the rewrite must actually resolve to the fund it borrowed.
    assert "hdfc-large-cap-fund-direct-growth" in resolve_scheme(rewritten)["slugs"]


def test_bare_field_lookup_is_resolved():
    """"what is the benchmark" names a field and nothing else, so the subject is inherited."""
    mem = _primed()
    rewritten = mem.contextualise("what is the benchmark")
    assert "HDFC Large Cap Fund" in rewritten


@pytest.mark.parametrize(
    "question",
    [
        "What is the difference between ELSS and SIP?",  # concept
        "what is a SIP?",  # concept
        "Compare HDFC Large Cap Fund and HDFC Small Cap Fund",  # comparative
    ],
)
def test_concept_questions_keep_their_independence(question):
    """A definition must never be answered with a scheme figure borrowed from last turn."""
    mem = _primed()
    assert mem.contextualise(question) == question


def test_a_question_that_names_its_own_scheme_is_never_rewritten():
    mem = _primed()
    q = "expense ratio of HDFC Small Cap Fund"
    assert mem.contextualise(q) == q


def test_long_question_without_a_subject_is_not_a_followup():
    mem = _primed()
    q = "can you tell me about the expense ratio and the exit load and the benchmark"
    assert mem.contextualise(q) == q


def test_empty_window_rewrites_nothing():
    assert ConversationMemory().contextualise("what about its exit load?") == (
        "what about its exit load?"
    )


def test_scheme_name_is_appended_not_prefixed():
    """Prefixing embeds worse and reads like a label rather than a typed question."""
    mem = _primed()
    assert mem.contextualise("what about its exit load?").startswith("what about its exit load")


# ── end to end ───────────────────────────────────────────────────────────────


def test_followup_cites_the_same_page_as_the_original_question():
    mem = ConversationMemory()
    first = answer_question("expense ratio of HDFC Large Cap Fund Direct Growth")
    mem.add_from_answer(first)
    second = answer_question("what about its exit load?", memory=mem)
    assert second.citations == first.citations


def test_guards_see_the_typed_question_not_the_rewrite():
    """Memory rewrites the query; it must never be able to rewrite the rules.

    If guards saw the rewritten string, a follow-up could borrow a subject in a way that
    slipped past a pattern the original question would have matched.
    """
    mem = _primed()
    # "its" would resolve to a fund, but the question is still an advice question.
    advice = answer_question("should I buy it?", memory=mem)
    assert advice.mode == "refusal"
    assert advice.guard == "advice"
    assert advice.citations == []


def test_rewrite_is_reported_in_notes():
    mem = _primed()
    ans = answer_question("what about its exit load?", memory=mem)
    assert any("follow-up resolved to" in n for n in ans.notes)


def test_memory_does_not_change_a_factual_answer():
    """Same question, with and without memory, must give the same fact."""
    bare = answer_question("What is the NAV of HDFC Equity Fund?")
    mem = _primed()
    with_mem = answer_question("What is the NAV of HDFC Equity Fund?", memory=mem)
    assert with_mem.citations == bare.citations
