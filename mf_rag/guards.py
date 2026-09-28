"""STAGE 5a — Guards: PII scrub and intent refusal. Leaf module — imports nothing downstream.

This is the component that makes the "facts-only" claim true, and it is deliberately a **leaf**:
it imports nothing from ``retriever``, ``store``, ``embedder`` or ``answerer``. That is how a
reviewer confirms guards are genuinely upstream of generation (P2) — by reading import lines,
not by trusting a comment. The call order in :func:`answer_question` is
``check_guards`` → ``retrieve`` → LLM, and no other order is possible without importing
backwards.

**No LLM is involved** (architecture §10.5). A keyword guard is deterministic, auditable,
unit-testable, and instant. A classifier would add latency, cost, and a new failure mode for a
rule set of about twenty patterns, and would introduce a "the classifier was feeling lenient
today" risk that is unacceptable in a compliance layer.

The four verdicts, in the order they are evaluated (P2 — the ordering is load-bearing):

1. ``pii``        → privacy notice, and the raw text is never logged (P9)
2. ``advice``     → one-line refusal
3. ``returns``    → one-line refusal
4. ``comparative``→ one-line refusal
5. ``speculative`` → one-line refusal
6. ``None``       → proceed to retrieval

Kinds 2-5 deliberately share one message. This bot answers published facts and nothing else,
so a question that is not asking for a published fact gets the same short reply whichever rule
caught it. Naming the category would mean telling someone who asked for guidance that their
question was "about returns" — a distinction they did not draw and that does not help them.

PII is checked **first** on purpose. The question *"my PAN is ABCDE1234F, should I buy?"* must
return the privacy notice rather than an advice refusal, and — more importantly — the PAN must
never be written to ``eval/queries.jsonl`` (P9). Checking advice first would log the PAN.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal, Pattern

from .config import settings

logger = logging.getLogger(__name__)

GuardKind = Literal["pii", "advice", "returns", "comparative", "speculative"]

#: Replaces every detected PII span in anything written to a log or the eval file.
PII_REDACTION = "[redacted]"


# ── User-facing copy ────────────────────────────────────────────────────────
# One refusal message for every non-factual question, whatever the reason. Advice, return
# figures, comparisons, hypotheticals and "what would you do" all land on the same short
# sentence, because from the user's side they are one situation: the question is not about a
# published fact. Distinguishing them in the reply would mean telling someone who asked for
# guidance that their question was specifically about returns — a distinction they did not
# make and does not help them.
#
# The message is deliberately short and states no opinion. It does not suggest an alternative
# question, describe what the bot can do, or link anywhere: every one of those is guidance,
# and this bot does not give guidance. See `FACT_ONLY` below.
FACT_ONLY = "I can help you with the factual details."

#: Kept as the PII-specific message. The uniform line above would be a worse answer here,
#: because a user who has just pasted their PAN needs to know it was not stored — that is a
#: factual statement about this system, which is exactly the one thing it may assert.
PRIVACY_NOTICE = (
    "Please don't share personal details. This assistant doesn't accept or store PAN, "
    "Aadhaar numbers, account numbers, OTPs, email addresses, or phone numbers. Your "
    "question was not saved. For account-specific queries, use the Groww app or your "
    "registered adviser."
)

#: Backwards-compatible alias. The advice, returns and comparative guards all share one text.
REFUSAL = FACT_ONLY


# ── PII patterns (architecture §10.2) ────────────────────────────────────────
#
# The context-sensitive classes (OTP, account number) are implemented as **lookaround**, never
# as a bare digit run. A bare ``\d{8,18}`` matches ISIN fragments, scheme codes and section
# numbers, and over-eager PII matching makes the bot look broken in a demo about mutual funds:
# it would refuse "exit load for 1 year" because "1" is near a digit run. This is a real
# trade-off, and it is resolved in favour of under-matching: a missed PII is a bad demo
# moment, a false positive is a bot that refuses to answer a fee question.


def _rx(pattern: str) -> Pattern[str]:
    """Case-insensitive. Correct for PII classes and for intent phrases alike.

    Users do not type in a consistent case: "My PAN is ...", "my pan is ..." and "Should I"
    are all the same message, and a case-sensitive guard silently misses the capitalised
    spellings. This is a compliance layer, so it errs toward matching.
    """
    return re.compile(pattern, re.IGNORECASE)


def _rx_cs(pattern: str) -> Pattern[str]:
    """Case-**sensitive**. Required for the PAN pattern only.

    Under IGNORECASE, ``[A-Z]{5}\\d{4}[A-Z]`` degrades to "any five letters, four digits, one
    letter" and would redact ordinary prose — and worse, would *report a PAN verdict* for
    ordinary prose, which is a false accusation, not just a missed detection. The other
    uppercase-sensitive character class in the codebase, ISINs, is unaffected either way.
    """
    return re.compile(pattern)


#: name -> compiled regex. Every entry is a hard match; a hit means "do not process this".
PII_PATTERNS: dict[str, Pattern[str]] = {
    # PAN: five letters, four digits, one letter. Anchored so it cannot match inside a word.
    "pan": _rx_cs(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    # Aadhaar: 4-4-4 digits, optional trailing X. \b on both sides keeps it from matching a
    # 12-digit slice out of the middle of a longer number.
    "aadhaar": _rx(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "email": _rx(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Phone: optional +91, then a 10-digit number starting 6-9.
    "phone": _rx(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b"),
    # OTP: six digits, but ONLY when the words otp/code/verify are close by. The two
    # alternations cover "your OTP is 123456" and "123456 is the code".
    #
    # The keyword-to-digits gap is [^0-9], NOT \W. In "my OTP is 445122" the words " is " sit
    # in between, and \W cannot match a letter — so a \W-based gap silently never fires, which
    # is exactly the bug this pattern had before it was caught by test_guards. Excluding only
    # digits still stops the match from jumping across an unrelated number.
    "otp": _rx(r"\b(?:otp|one[\s-]?time[\s-]?pass|verification|verify|code)\b[^0-9]{0,20}\d{6}\b"
               r"|\b\d{6}\b[^0-9]{0,20}\b(?:otp|one[\s-]time[\s-]?pass|verification|verify|code)\b"),
    # Account number: 8-18 digits, but ONLY next to account/a/c/acc. Same two-way alternation.
    "account": _rx(r"\b(?:a/?c|acc(?:ount)?(?:\s+no\.?)?)\b[^0-9]{0,20}\d{8,18}\b"
                   r"|\b\d{8,18}\b[^0-9]{0,20}\b(?:a/?c|acc(?:ount)?)\b"),
}

#: PII classes that are only credible with surrounding context. Distinguishing them is what
#: lets the tests assert the false-positive behaviour instead of only the positive.
CONTEXTUAL_PII: frozenset[str] = frozenset({"otp", "account"})

#: A question *asking the bot to supply* a personal identifier. This is a different failure from
#: PII_PATTERNS and cannot be caught by them.
#:
#: PII_PATTERNS look for the identifier itself, so they only fire when the user has already
#: typed the secret. "What is my PAN number for KYC?" contains no PAN-shaped string, so every
#: one of those patterns correctly declines to match — and the bot went on to answer with the
#: NAV, because nothing else in the pipeline knew the question was about a person's identity.
#:
#: The corpus cannot answer such a question under any circumstances, so the honest response is
#: to say the assistant has no access to personal records. This is deliberately narrow: it
#: requires a possessive or interrogative pairing with a named identifier, so "what documents
#: do I need for KYC" and "how do I update my PAN on Groww" (a legitimate process question)
#: are not swept up by it.
#:
#: "address" is deliberately absent. An AMC's registered address is a published fact that
#: belongs in the corpus, while a home address is PII, and no regex over the question can tell
#: them apart — "what is the address of the AMC" and "what is my address" are the same shape.
#: Excluding the ambiguous class loses little and avoids refusing a legitimate question, which
#: is the failure users actually notice.
PII_ELICITATION: Pattern[str] = _rx(
    r"\b(?:what|whats|which|tell\s+me|give\s+me|share|provide|find|show|lookup|look\s+up)\b"
    r"[^?]{0,40}?\b(?:my|our|his|her|their|your|the)\b"
    r"[^?]{0,40}?\b(?:pan|aadhaar|aadhar|account\s+(?:number|no)|"
    r"bank\s+account|ifsc|upi\s+id|phone\s+number|mobile\s+number|"
    r"email\s+address|passport|voter\s+id|date\s+of\s+birth|dob)\b"
)


# ── Intent patterns (architecture §10.3) ─────────────────────────────────────
#
# \b-delimited phrases, not bare substrings: "safe" alone would fire on "safely", and
# "vs" as a substring appears inside ordinary words.

ADVICE_PATTERNS: tuple[Pattern[str], ...] = tuple(
    _rx(p) for p in (
        r"\bshould\s+(?:i|we)\b",
        r"\bis\s+it\s+(?:good|safe|worth|advisable)\b",
        r"\b(?:good|safe|advisable|worth|suitable)\s+for\s+(?:me|my)\b",
        r"\bi\s+(?:recommend|suggest)\b",
        r"\brecommend(?:s|ed|ation|ations)?\b",
        r"\bbest\s+(?:fund|scheme|option|choice)\s+for\b",
        # "Which HDFC fund is best for me?" — the noun sits BEFORE the verb, so the pattern
        # above (`best fund for`) cannot see it. Personal suitability, in either word order.
        r"\b(?:is|would\s+be)\s+best\s+for\s+(?:me|my|us|our)\b",
        r"\bbest\s+for\s+(?:me|my|us|our)\b",
        # "Is HDFC Flexi Cap a good investment?" — an article plus the noun sits between the
        # copula and the adjective, which is why `is it good` never matched.
        r"\ba\s+(?:good|bad|safe|risky|profitable|worthwhile)\s+investment\b",
        r"\bsafe\s+to\b",
        # "rebalance my portfolio" and "how should I allocate my money" are advice. The bare
        # nouns are not. "Portfolio turnover ratio" and "asset allocation" are both fields
        # the scheme pages publish, and the bare-noun patterns refused them — "portfolio
        # turnover ratio of HDFC Small Cap Fund" came back as advice when it is a number the
        # page prints in plain sight. So each of these now requires a first-person or
        # possessive frame, which is what makes it advice rather than a published field.
        r"\b(?:my|our)\s+portfolios?\b",
        r"\bportfolios?\s+(?:allocation|rebalanc\w+|mix|composition|construction)\b",
        r"\brebalanc\w+\s+(?:the|my|our)?\s*portfolios?\b",
        r"\b(?:my|our)\s+allocations?\b",
        r"\ballocat(?:e|es|ed|ing)\s+(?:my|our|it|the\s+money)\b",
        r"\bhow\s+should\s+i\s+allocat",
        r"\bsuitabl(?:e|ity)\b",
        r"\bworth\s+buy(?:ing)?\b",
        r"\bwhich\s+should\s+(?:i|we)\b",
        r"\b(?:buy|invest\s+in|pick|choose)\s+(?:this|it|that)\b",
        r"\bhelp\s+me\s+choose\b",
        r"\b(?:can|should)\s+i\s+buy\b",
        r"\bis\s+it\s+too\s+risky\b",
        r"\bhow\s+much\s+should\s+i\b",
    )
)

# Word-boundary anchored so "returns" does not fire on "returns" inside a longer word, and so
# the ordinary English "performance review" style question is still caught (it is a returns
# question in intent, and refusing is the safe direction).
RETURNS_PATTERNS: tuple[Pattern[str], ...] = tuple(
    _rx(p) for p in (
        # "returned" and "returning" as well as "return"/"returns". `returns?` looks like it
        # covers the family and does not: "What has HDFC Large Cap Fund returned so far?"
        # sailed past the guard and was answered with the benchmark line. The same
        # morphology gap had already been fixed once in the answerer's post-check, which is
        # how a question can be refused in one place and answered in another.
        r"\breturn(?:s|ed|ing)?\b",
        r"\bperform(?:ance|ed|s|ing)?\b",
        r"\bcagr\b",
        r"\bbest\s+perform(?:ing|er|ance)\b",
        r"%\s*gain",
        r"\bgained\b",
        r"\byields?\b",
        # The subject between "has" and "given" is matched loosely rather than hardcoded as
        # "it", because a user names the fund: "How much has HDFC Equity Fund given?" left
        # the guard unsatisfied and fell through to retrieval, where it happened to decline
        # for want of a matching chunk. Declining for the wrong reason is still a refusal in
        # this case, but it is one corpus change away from being an answer.
        r"\bhow\s+much\s+(?:has|have)\s+[\w\s'()]{0,40}?\s+given\b",
        r"\bhow\s+much\s+did\s+[\w\s'()]{0,40}?\s+give\b",
        r"\bgood\s+returns?\b",
        r"\bhas\s+given\b",
        r"\bmade\s+money\b",
        r"\bprofit(?:able)?\b",
    )
)

COMPARATIVE_PATTERNS: tuple[Pattern[str], ...] = tuple(
    _rx(p) for p in (
        r"\bbetter\s+than\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bcompar(?:e|es|ed|ing|ison)\b",
        r"\brank(?:s|ed|ing)?\b",
        r"\btop\s*\d+\b",
        r"\bwhich\s+is\s+better\b",
        r"\bmost\s+(?:profitable|successful|perform\w*)\b",
        r"\bhighest\s+returns?\b",
    )
)

#: Hypothetical and forward-looking questions. These are the hardest of the four to catch and
#: the easiest to get wrong, because the corpus is full of forward-looking *facts* — a 3-year
#: lock-in, a benchmark name, "exit load applies if redeemed within 1 year" — and a pattern
#: broad enough to catch "what if the market crashes" would also catch "what is the exit load".
#:
#: So every entry is anchored on a word that appears only in a *speculation*:
#:
#: * "what if", "suppose", "imagine" — a hypothetical frame, never a request for a published
#:   figure. The boundary case is "what is the exit load if redeemed within 1 year": that asks
#:   about a published conditional, so "if redeemed" is deliberately absent from this list.
#: * "will it", "will the fund" — a prediction. These pages state facts in the present tense,
#:   so a future-tense question is asking for something the corpus cannot hold.
#: * "predict", "forecast", "should I expect" — unambiguous.
#:
#: Deliberately excluded: "next year", "in 5 years", "the future". Those appear inside
#: legitimate procedural questions such as "how do I exit after 5 years", and a temporal word
#: on its own says nothing about whether the user is speculating.
SPECULATIVE_PATTERNS: tuple[Pattern[str], ...] = tuple(
    _rx(p) for p in (
        r"\bwhat\s+if\b",
        r"\bwhat\s+would\s+happen\b",
        r"\bwhat\s+will\s+happen\b",
        r"^\s*(?:suppose|imagine|assume)\b",
        r"\b(?:suppose|imagine)\s+(?:the|if|i|we|there|rates?|markets?)\b",
        r"\bwill\s+(?:it|the\s+(?:fund|nav|scheme|market|price|returns?))\b",
        r"\b(?:predict|forecast|projection)s?\b",
        r"\bshould\s+i\s+expect\b",
        r"\bhow\s+much\s+will\s+(?:i|we)\s+(?:earn|make|get|save)\b",
        r"\bwhere\s+(?:will|do)\s+(?:the|my)\s+(?:nav|price|fund)\s+go\b",
    )
)


@dataclass
class GuardVerdict:
    """A decision to short-circuit answering, with the copy to show the user."""

    kind: GuardKind
    message: str
    link: str | None = None
    #: The question with every PII span replaced. This is the ONLY form that may be logged
    #: or written to the eval file (P9). Empty when nothing was redacted.
    redacted_question: str = ""


def _first_match(text: str, patterns) -> str | None:
    """Return the matching phrase, so the log can say *why* a guard fired."""
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def detect_pii(question: str) -> list[str]:
    """Names of every PII class found, in declaration order. Empty if the text is safe."""
    return [name for name, pattern in PII_PATTERNS.items() if pattern.search(question)]


def scrub_for_log(question: str) -> str:
    """Replace every detected PII span with :data:`PII_REDACTION`.

    The only function permitted to put a user question anywhere persistent. It redacts by
    substitution rather than by refusing, so a partly-PII question still yields a useful log
    line — and it redacts *all* classes, not just the one that would have tripped a guard, so
    a future call site cannot leak a class the caller happened not to check.
    """
    text = question
    for pattern in PII_PATTERNS.values():
        text = pattern.sub(PII_REDACTION, text)
    return text


def check_guards(question: str) -> GuardVerdict | None:
    """Return a verdict to short-circuit, or ``None`` to proceed to retrieval.

    Order is PII → advice → returns → comparative, and it is load-bearing. See the module
    docstring for why PII must come first.
    """
    pii_classes = detect_pii(question)
    if pii_classes:
        redacted = scrub_for_log(question)
        # Log the redacted form and the class names only. The raw text is never passed to the
        # logger, so it cannot reach a log file, a traceback, or a Streamlit console.
        logger.info("guard=pii classes=%s question=%r", ",".join(pii_classes), redacted)
        return GuardVerdict(
            kind="pii",
            message=PRIVACY_NOTICE,
            redacted_question=redacted,
        )

    # Elicitation is checked as part of the PII branch, not as a fifth intent guard: it is the
    # same policy outcome (nothing personal is handled here) for the same reason, and running
    # it after `detect_pii` would mean two code paths that both return kind="pii".
    elicitation = PII_ELICITATION.search(question)
    if elicitation:
        logger.info("guard=pii class=elicitation matched=%r question=%r", elicitation.group(0), question)
        return GuardVerdict(
            kind="pii",
            message=PRIVACY_NOTICE,
            redacted_question=scrub_for_log(question),
        )

    hit = _first_match(question, ADVICE_PATTERNS)
    if hit:
        logger.info("guard=advice matched=%r question=%r", hit, question)
        return GuardVerdict(
            kind="advice",
            message=REFUSAL,
            redacted_question=question,
        )

    hit = _first_match(question, RETURNS_PATTERNS)
    if hit:
        logger.info("guard=returns matched=%r question=%r", hit, question)
        return GuardVerdict(
            kind="returns",
            message=REFUSAL,
            redacted_question=question,
        )

    hit = _first_match(question, COMPARATIVE_PATTERNS)
    if hit:
        logger.info("guard=comparative matched=%r question=%r", hit, question)
        return GuardVerdict(
            kind="comparative",
            message=REFUSAL,
            redacted_question=question,
        )

    # Speculation is checked last of the intents. The earlier guards are about *what* is being
    # asked; this one is about the tense and framing, and a question can be both ("what if I
    # switch to the fund with better returns") — in which case the earlier, more specific
    # reason is the one worth logging.
    hit = _first_match(question, SPECULATIVE_PATTERNS)
    if hit:
        logger.info("guard=speculative matched=%r question=%r", hit, question)
        return GuardVerdict(
            kind="speculative",
            message=REFUSAL,
            redacted_question=question,
        )

    return None


if __name__ == "__main__":
    probes = [
        "my PAN is ABCDE1234F",
        "call me on 9876543210",
        "aadhaar 1234 5678 9012",
        "email me at priya.sharma@gmail.com",
        "my OTP is 445122",
        "account number 30123456789",
        "Should I buy HDFC Small Cap?",
        "which fund has best returns",
        "is Flexi Cap better than Large Cap",
        "what is the exit load for 1 year",
        "expense ratio of HDFC Large Cap?",
    ]
    print(f"{'question':<44} verdict")
    print("-" * 72)
    for probe in probes:
        verdict = check_guards(probe)
        kind = verdict.kind if verdict else "proceed"
        print(f"{probe:<44} {kind}")
    print()
    print("redaction check — the PAN must not survive:")
    print(f"  {scrub_for_log('my PAN is ABCDE1234F')}")
