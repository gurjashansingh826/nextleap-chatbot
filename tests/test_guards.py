"""STAGE 5a — guard tests.

The guards are the only thing standing between this bot and a compliance incident, so they
get the most adversarial tests in the repo. Two failure modes matter and they pull in
opposite directions:

* **False negatives** — a PAN or OTP slips through and gets logged or sent to an LLM.
* **False positives** — an ordinary question is refused, which is its own kind of failure:
  the bot becomes useless and nobody notices because the refusal looks polite.

Every PII test is therefore paired with a near-miss control that must NOT fire. A test that
only proves the guard fires proves very little.
"""

from __future__ import annotations

import pytest

from mf_rag.guards import FACT_ONLY, check_guards, detect_pii, scrub_for_log


def test_the_refusal_is_one_sentence_and_states_no_opinion() -> None:
    """The bot answers published facts. Its refusal must not become guidance by other means.

    Asserted on the string itself rather than on any one call site, because the failure mode
    is cumulative: a longer "helpful" refusal, or one that suggests a better question, is the
    easiest way for a facts-only bot to start advising.
    """
    assert FACT_ONLY == "I can help you with the factual details."
    assert FACT_ONLY.count(".") == 1, "the refusal must be a single sentence"
    for banned in ("should", "recommend", "instead", "try", "https", "consider", "you can"):
        assert banned not in FACT_ONLY.lower(), f"the refusal offers guidance: {banned!r}"


def test_every_non_factual_kind_shares_one_message() -> None:
    """Advice, returns and comparative must be indistinguishable to the user.

    Naming the category would tell someone who asked for guidance that their question was
    "about returns" — a distinction they did not make and that does not help them.
    """
    for question in [
        "Should I buy HDFC Small Cap Fund?",
        "Which HDFC fund has the highest returns?",
        "Is HDFC Large Cap better than HDFC Small Cap?",
    ]:
        verdict = check_guards(question)
        assert verdict is not None, f"no guard fired for: {question!r}"
        assert verdict.message == FACT_ONLY, f"{question!r} used a different message"
        assert verdict.link is None

# ── PII detection ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "my PAN is ABCDE1234F",  # explicit PAN label
        "PAN ABCDE1234F",  # label as a bare token
        "my aadhaar number is 1234 5678 9012",  # spaced Aadhaar
        "aadhaar 123456789012",  # unspaced
        "account number 50100234567890",  # 14-digit account + context
        "my otp is 445122",  # the case that exposed a real bug: "is" is a letter, so a
        # `\W` gap could not span it
        "the otp 445122",  # same, minimal
        "reach me at someone@example.com",
        "my phone number is 9876543210",
    ],
)
def test_pii_is_detected(question: str) -> None:
    verdict = check_guards(question)
    assert verdict is not None, f"expected a guard verdict for PII: {question!r}"
    assert verdict.kind == "pii"


def test_asking_for_personal_records_is_refused() -> None:
    """A question can be about PII without containing any.

    "What is my PAN number for KYC?" holds no PAN-shaped string, so `detect_pii` correctly
    finds nothing — and the bot answered it with the NAV. Asking the assistant to supply a
    personal identifier is the same policy failure as pasting one, and it is only catchable by
    a pattern aimed at the request rather than at the value.
    """
    for question in [
        "What is my PAN number for KYC?",
        "Tell me the account number of this fund",
        "give me the aadhaar number of my father",
        "share his phone number",
    ]:
        verdict = check_guards(question)
        assert verdict is not None, f"no guard fired for: {question!r}"
        assert verdict.kind == "pii"


@pytest.mark.parametrize(
    "question",
    [
        # Process questions that mention an identifier but are not asking for one. Sweeping
        # these up would be the false positive the elicitation pattern is most likely to cause.
        "What documents do I need for KYC?",
        "How do I update my PAN on Groww?",
        "Which NAV is used for SWP?",
        "What is the address of the AMC?",
    ],
)
def test_identifier_process_questions_are_allowed(question: str) -> None:
    assert check_guards(question) is None, f"over-refused a process question: {question!r}"


@pytest.mark.parametrize(
    "question",
    [
        # Digits with no PII context. Every one of these is a plausible factual question, and
        # refusing them would gut the bot — the corpus is full of numbers.
        "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
        "Is the exit load 1% if redeemed within 1 year?",
        "What is the minimum SIP investment?",
        "What is the lock-in period, 3 years?",
        "The fund was launched in 1996, right?",
        "What is the NAV as of 2026-09-28?",
        "Is AUM above 50000 crore?",
        # Uppercase prose. Under a case-insensitive PAN pattern `[A-Z]{5}\d{4}[A-Z]`, ordinary
        # capitals would fire, which is why PAN alone is compiled case-sensitively.
        "LARGE CAP FUND DIRECT GROWTH",
        "NIFTY 50 Hybrid Composite Debt 50:50 Index",
        "BSE 100 and TRIX indices",
    ],
)
def test_ordinary_questions_are_not_pii(question: str) -> None:
    assert check_guards(question) is None, f"false PII positive on: {question!r}"
    assert detect_pii(question) == [], f"false PII positive on: {question!r}"


def test_pan_needs_five_leading_letters() -> None:
    """A precise control on the PAN shape, not just a plausible-looking string."""
    assert detect_pii("my PAN is ABCDE1234F")
    assert not detect_pii("my PAN is ABCD1234E"), "PAN must require 5 letters, not 4"


def test_detect_pii_reports_categories() -> None:
    found = detect_pii("my PAN is ABCDE1234F and my otp is 445122")
    assert len(found) >= 2, f"expected both categories, got {found}"


# ── precedence ────────────────────────────────────────────────────────────────


def test_pii_outranks_advice() -> None:
    """PII must win, because its remediation is different and stricter.

    A question that both asks for advice and contains a PAN has to be refused for the PAN
    and its number redacted. If the advice guard fired first the question would be logged
    with the PAN still in it, which defeats the entire purpose of the PII rule.
    """
    verdict = check_guards("Should I buy HDFC Small Cap? My PAN is ABCDE1234F")
    assert verdict is not None
    assert verdict.kind == "pii", "the PII verdict must take precedence over advice"
    assert "ABCDE1234F" not in verdict.redacted_question, "PAN must be redacted in the log copy"


def test_advice_outranks_performance_questions() -> None:
    """Both refuse, so the exact kind matters less than the fact that it refuses."""
    verdict = check_guards("Should I buy the fund that gave the best returns?")
    assert verdict is not None
    assert verdict.kind in {"advice", "returns", "comparative"}


# ── advice ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "Should I buy HDFC Small Cap Fund?",
        "Which HDFC fund is best for me?",
        "Is HDFC Flexi Cap a good investment?",
        "Can you recommend a scheme for retirement?",
    ],
)
def test_advice_questions_are_refused(question: str) -> None:
    verdict = check_guards(question)
    assert verdict is not None, f"expected a refusal for: {question!r}"
    assert verdict.kind == "advice"
    assert verdict.message == FACT_ONLY, (
        "every non-factual question gets the same one-line reply, whatever rule caught it"
    )
    assert verdict.link is None, (
        "the refusal must not append a link — pointing the user somewhere else is guidance"
    )


# ── returns and comparison ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "Which HDFC fund has the highest returns?",
        "What did HDFC Large Cap Fund return last year?",
        "Show me the 5-year CAGR for HDFC Flexi Cap Fund.",
    ],
)
def test_performance_questions_are_refused(question: str) -> None:
    """The bot must never state or rank returns. Linking to the factsheet is the only way out."""
    verdict = check_guards(question)
    assert verdict is not None, f"expected a refusal for: {question!r}"
    assert verdict.kind in {"returns", "comparative"}
    assert verdict.message == FACT_ONLY
    assert verdict.link is None, "the refusal must not append a link"


@pytest.mark.parametrize(
    "question",
    [
        "Is HDFC Large Cap better than HDFC Small Cap?",
        "Compare HDFC Flexi Cap and HDFC Balanced Advantage.",
    ],
)
def test_comparative_questions_are_refused(question: str) -> None:
    verdict = check_guards(question)
    assert verdict is not None, f"expected a refusal for: {question!r}"
    assert verdict.kind in {"comparative", "advice", "returns"}


# ── hypotheticals ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "What if the market crashes next year?",
        "What would happen if I redeem after 2 years?",
        "Suppose the NAV falls 10%, should I buy?",
        "Will it recover if markets fall?",
        "Can you predict the NAV next quarter?",
        "How much will I earn in 5 years?",
        "Where will the NAV go from here?",
    ],
)
def test_hypothetical_questions_are_refused(question: str) -> None:
    """A hypothetical has no published answer, so answering one is the same failure as guessing."""
    verdict = check_guards(question)
    assert verdict is not None, f"no guard fired for: {question!r}"
    assert verdict.message == FACT_ONLY


@pytest.mark.parametrize(
    "question",
    [
        # The corpus is full of forward-looking *facts*: a 3-year lock-in, a 1-year exit-load
        # window. A speculative guard broad enough to catch "what if the market crashes" but
        # not to catch these would be useless, so each is pinned here.
        "What is the exit load if redeemed within 1 year?",
        "What is the lock-in period?",
        "How do I exit after 5 years?",
        "What happens on maturity?",
        "When is the NAV updated?",
    ],
)
def test_conditional_facts_are_still_answerable(question: str) -> None:
    assert check_guards(question) is None, f"over-refused a published fact: {question!r}"


# ── the log copy ──────────────────────────────────────────────────────────────


def test_scrub_for_log_removes_digits() -> None:
    scrubbed = scrub_for_log("my PAN is ABCDE1234F, otp 445122, phone 9876543210")
    for secret in ("ABCDE1234F", "445122", "9876543210"):
        assert secret not in scrubbed, f"{secret} survived scrubbing: {scrubbed!r}"


def test_scrub_for_log_keeps_the_question_readable() -> None:
    """Redaction must not destroy the log's usefulness — an unreadable log is not a control."""
    scrubbed = scrub_for_log("What is the expense ratio of HDFC Large Cap Fund?")
    assert "expense ratio" in scrubbed.lower()
    assert "HDFC Large Cap" in scrubbed


def test_verdict_redacts_by_default() -> None:
    verdict = check_guards("my PAN is ABCDE1234F")
    assert verdict is not None
    assert verdict.redacted_question, "a verdict must always carry a log-safe question"
