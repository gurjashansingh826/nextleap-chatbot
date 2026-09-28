"""The answer-accuracy grader must not grade the system wrongly.

The first version of ``tools/answer_eval.py`` reported 13.3% end-to-end accuracy. Every one
of its 30 "misses" was a correct answer: it tested whether the answer contained the eval
row's ``key_fact`` label verbatim, and an LLM asked the same thing answers in its own words.

These tests pin the paraphrases that were misgraded, and — just as important — the negatives
that a lazy grader would wave through. A grader that only produces passes is as useless as
one that only produces failures, and it is the failure direction that publishes a wrong
number in the eval report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from answer_eval import grade, probes_for, squeeze  # noqa: E402


# ── paraphrases that must be graded as correct ───────────────────────────────


@pytest.mark.parametrize(
    "key_fact, answer",
    [
        # the LLM says "Total Expense Ratio", the label says "TER"
        ("Expense ratio (TER): 1.03%", "The expense ratio (Total Expense Ratio) is 1.03%."),
        # number spelled out
        ("Further, these schemes have a mandatory lock-in period of 3 years.",
         "The lock-in period for ELSS funds under Section 80C is three years."),
        # "Nil" rendered as a negation
        ("Exit load: Nil", "There is no exit load on the HDFC ELSS Tax Saver Fund."),
        # no colon anywhere in the label, so there is no value half to take a figure from
        ("Exit Load for units in excess of 15% of the investment,1% will be charged for "
         "redemption within 1 year.",
         "The fund charges a 1% exit load on redemption of units within 1 year if the "
         "redemption exceeds 1% of the investment."),
        ("According to SEBI guidelines, these funds are required to invest at least 35% of "
         "their total assets each in large-cap and mid-cap stocks.",
         "Large- and mid-cap funds must allocate at least 35% of their total assets."),
        # the leading article must be stripped, or no prefix of the label matches
        ("The five levels of risks in a mutual fund riskometer are explained below -",
         "The SEBI riskometer classifies mutual fund risk into five levels."),
        # a label that repeats itself, so the tail can never match in full
        ("Riskometer level (as published on the page): Moderately High Riskometer",
         "The published riskometer level is Moderately High."),
    ],
)
def test_correct_paraphrase_is_graded_correct(key_fact, answer):
    passed, _ = grade({"key_fact": key_fact}, answer)
    assert passed, f"correct answer misgraded: {key_fact!r} -> {answer!r}"


# ── wrong answers that must be graded wrong ───────────────────────────────────


@pytest.mark.parametrize(
    "key_fact, answer",
    [
        # the sibling-plan hazard: right field, wrong plan page's number
        ("Expense ratio (TER): 0.78%", "The expense ratio is 1.03%."),
        # base vs total expense ratio, which the corpus publishes as two separate fields
        ("Expense ratio (TER): 1.03%", "The base expense ratio is 0.84%."),
        # the wrong fund manager
        ("Fund manager: Prashant Jain", "The fund is managed by Chirag Setalvad."),
        # the wrong minimum
        ("Minimum SIP investment: \u20b9500", "The minimum SIP is \u20b9100."),
        ("Minimum SIP investment: \u20b9100", "The minimum SIP is \u20b9500."),
        # a near-miss on a different index
        ("Benchmark: NIFTY 500 Total Return Index (NIFTY 500 TRI)",
         "The benchmark is the NIFTY 100 Total Return Index."),
        # denying a fact that exists
        ("Further, these schemes have a mandatory lock-in period of 3 years.",
         "There is no lock-in requirement for these funds."),
        # an unrelated field entirely
        ("Lock-in period: 3 years", "The exit load is 1% if redeemed within 1 year."),
    ],
)
def test_wrong_answer_is_graded_wrong(key_fact, answer):
    passed, _ = grade({"key_fact": key_fact}, answer)
    assert not passed, f"wrong answer passed: {key_fact!r} -> {answer!r}"


# ── the grader's own invariants ───────────────────────────────────────────────


def test_a_bare_digit_is_never_used_as_a_probe_on_a_colonless_label():
    """Grading on "1" would pass almost anything, since almost any answer contains one."""
    for kind, probe in probes_for("Exit load of 1% applies for redemption within 1 year."):
        if kind == "value-digit":
            assert len(probe) >= 2, f"single-digit probe {probe!r} is too weak to be evidence"


def test_number_words_fold_to_digits_on_both_sides():
    assert squeeze("three years") == squeeze("3 years")
    assert squeeze("Five levels") == "5levels"


def test_empty_key_fact_produces_no_probes():
    assert probes_for("") == []
    assert grade({"key_fact": ""}, "anything at all") == (False, "-")
