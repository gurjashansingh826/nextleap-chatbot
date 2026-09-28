"""STAGE 6 — Answering: the LLM call, the post-checks, and the deterministic fallback.

**Guards run before this module does anything.** :func:`answer_question` calls
``guards.check_guards`` as its first statement, and this module does not import
``retriever``'s guard logic or reimplement it. A refusal, a PII notice, and a genuine
"not in my sources" therefore never reach the LLM, which is what makes P2 (no PII reaches the
model, no advice question is answered) a structural property rather than a hope.

The seven post-checks (architecture §12) are the reason a facts-only system can be trusted with
numbers. They are ordered, and the order is load-bearing:

* **1 → 2 first** (citation present, then citation *real*). An ungrounded citation is a trust
  failure that invalidates everything else about the answer, so it triggers the deterministic
  extractive fallback rather than a patch. Patching a bad citation produces an answer that
  looks verified and is not, which is strictly worse than an obviously-robotic one.
* **3 → 4 → 5** are corrections, not rejections: truncate an over-long answer, refuse advice
  language, refuse performance claims.
* **6 → 7** append the footer and disclaimer. These cannot fail into a fallback — they are
  appended unconditionally — which is why they are last.

Check 5 is deliberately **sentence-scoped and conservative**: a bare ``%`` is never blocked,
because an expense ratio *is* a percentage and four of the graded facts are percentages. Only a
percentage co-occurring with a return-context word in the same sentence counts as a performance
claim, so "0.63% expense ratio" passes and "18% annual returns" does not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from . import prompts
from .config import settings
from .memory import ConversationMemory
from .guards import check_guards
from .retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

AnswerMode = Literal["llm", "extractive", "refusal", "not_in_sources", "pii_refusal"]


@dataclass
class Answer:
    """The output object (architecture §7.5).

    ``mode`` is deliberately part of the contract: it makes the guard paths assertable in tests,
    visible in the UI, and provable in the demo log — you can see which path a given answer
    took, rather than having to infer it.
    """

    question: str
    text: str
    citations: list[str]
    as_of: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: AnswerMode = "llm"
    guard: str | None = None
    disclaimer: str = prompts.DISCLAIMER
    notes: list[str] = field(default_factory=list)

    @property
    def is_factual(self) -> bool:
        return self.mode in ("llm", "extractive")


# ── Post-check helpers ───────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+")

#: Phrases that turn a factual answer into advice. Matched case-insensitively, with a
#: word boundary, so "I recommend" is caught and "recommended minimum" is not.
_ADVICE_RE = re.compile(
    # "you must invest" is advice. "these funds must invest at least 35% of their assets" is a
    # SEBI rule quoted as fact, and the model produces that phrasing whenever asked about large
    # and mid-cap allocation — the bare `must invest` rejected the correct answer, turning a
    # published rule into a refusal. The subject is what separates the two: advice is addressed
    # to the reader, a regulation is addressed to the fund.
    r"\b(?:you\s+should|you\s+must\s+invest|i\s+recommend|i\s+suggest|buy\s+this|"
    r"good\s+for\s+(?:you|me|your)|advisable|must\s+invest\s+(?:in|your)|worth\s+buying|"
    r"you\s+can\s+buy|go\s+for\s+it|ideal\s+for\s+you)\b",
    re.IGNORECASE,
)

#: Words that make a percentage a *performance* claim. Required in the SAME sentence as the %.
#: `return(?:s|ed|ing)?` rather than `returns?`, because the bare form misses the most natural
#: phrasing of all — "the fund returned 22.4% last year" — which is how a language model
#: volunteers a return figure when asked a loosely-worded question.
_PERFORMANCE_CONTEXT_RE = re.compile(
    r"\b(?:return(?:s|ed|ing)?|cagr|gained|gain|yield|performance|profit|appreciation|ir)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    """Split into sentences, discarding empties and fragments with no letters.

    Fragments matter: a citation line like "https://groww.in/..." splits into pieces with no
    alphabetic content, and counting those as sentences would make a 1-sentence answer look
    like 3 and get truncated to nothing.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip())]
    return [p for p in parts if p and re.search(r"[A-Za-z]", p)]


def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    """Keep at most ``max_sentences`` prose sentences.

    Trailing non-prose lines (a bare citation, the footer) survive the cut, because a
    three-sentence answer with no citation fails post-check 1 and falls back to the
    extractive path — turning a recoverable formatting problem into a downgrade.
    """
    kept: list[str] = []
    counted = 0
    for part in [p.strip() for p in text.strip().splitlines() if p.strip()]:
        # A line with no letters is a citation or a divider, not prose. It always survives.
        if not re.search(r"[A-Za-z]", part):
            kept.append(part)
            continue
        for sentence in _sentences(part):
            # Once the cap is reached, prose sentences are dropped. An earlier version had two
            # branches that both appended — one for "over the cap" and one for "under it" —
            # so the function logged a truncation it never performed and the 3-sentence limit
            # was silently unenforced on every LLM answer.
            if counted < max_sentences:
                kept.append(sentence)
                counted += 1
    return "\n".join(kept).strip()


def _urls_in(text: str) -> list[str]:
    return [u.rstrip(".,;") for u in _URL_RE.findall(text)]


def _append_footer(text: str, as_of: str) -> str:
    """Append the as-of footer and disclaimer if absent (checks 6 and 7)."""
    out = text.strip()
    if prompts.AS_OF_TEMPLATE.split("{")[0] not in out:
        out = f"{out}\n\n{prompts.as_of_line(as_of)}"
    if prompts.DISCLAIMER not in out:
        out = f"{out}\n\n_{prompts.DISCLAIMER}_"
    return out


def _performance_claim(text: str) -> str | None:
    """Return the offending sentence, or ``None``.

    Sentence-scoped: a percentage alone is a fee, a lock-in, or a fund-size share, and the
    graded facts are full of them. Only ``%`` *and* a return-context word in the same sentence
    is a performance claim.
    """
    for sentence in _sentences(text):
        if "%" in sentence and _PERFORMANCE_CONTEXT_RE.search(sentence):
            return sentence
    return None


# ── The deterministic fallback (ADR-2, principle P8) ─────────────────────────


# Words that appear in almost any question and match almost any line of prose, so counting
# them rewards whichever line happens to contain them by accident. "on the page" was scoring
# the riskometer line level with a benchmark question purely on "the".
_STOPWORDS = frozenset(
    """what which when where who whom whose why how is are was were be been being do does
    did doing have has had having will would shall should can could may might must the a an
    and or but if then than that this these those there here for of to in on at by with from
    as it its into about over under between""".split()
)


def _query_terms(question: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", question.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def _stems(text: str) -> set[str]:
    """Content words, truncated to four characters, for morphology-tolerant matching.

    Exact-token matching cannot answer a question the corpus plainly answers. "Is HDFC ELSS
    Tax Saver Fund locked in?" shares no token with the chunk that states "Lock-in period:
    3 years" — not one — so an exact test scored the single most relevant chunk in the
    retrieval as unrelated, and the bot declined a fact it was holding. Four characters is
    enough to join ``lock``/``locked``/``locking`` and ``expense``/``expenses`` while staying
    short enough not to fuse distinct fields ("min" and "mini" are different questions, and
    this is a relevance test, not a claim of identity).
    """
    return {
        t[:4]
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


_FACT_LINE_RE = re.compile(r"^[-*]?\s*(?P<label>[^:]{1,80}):")


#: Words that cannot, on their own, establish that a line is on-topic.
#:
#: `_pick_lines` treats a question term appearing in a line as evidence the line answers
#: the question. That is wrong for words which appear across many field labels of the same
#: page and therefore identify no field at all. Two live failures, both producing a wrong
#: fact under a correct citation:
#:
#: * "What is the Sharpe ratio of HDFC Large Cap Fund?" shares ``ratio`` with
#:   "- Expense ratio (TER): 1.03%" and answered with the expense ratio.
#: * "Who manages HDFC Focused Large Cap Direct Plan?" shares ``plan`` with
#:   "- Plan type: Direct" and answered with the plan type.
#:
#: Both are the worst failure this system has: a real field, quoted verbatim, under a real
#: source URL, with nothing about it looking wrong. The user has no way to tell.
#:
#: The rule is that a match must rest on a term that *distinguishes* a field. Each word
#: below appears in several field labels on the same page, or in the plan name, so the
#: field's real name has to supply the distinguishing term instead: "expense" identifies the
#: expense ratio, "lock" the lock-in period, "exit" the exit load. None of those are listed,
#: so questions about them still match.
#:
#: This is spelled as *words* and stemmed by `_stems` below, never as hand-written
#: prefixes. Writing the prefixes by hand is a silent failure: `_stems` truncates to four
#: characters, so "ratio" is ``rati`` and not ``rat``, and a mistyped prefix does not
#: raise — it just quietly matches nothing and the wrong-fact bug comes straight back.
#: ``test_generic_field_match.py`` asserts every word here actually stems into the set.
#:
#: This list is tuned to *this* corpus. A new page whose fields share a word listed here
#: would need it re-reviewed — which is why it is a named constant with this note, rather
#: than a handful of words dropped into the matching function.
_GENERIC_FIELD_WORDS = (
    # appears in several field labels of one page
    "ratio", "plan", "type", "period", "value", "amount", "name", "date", "charge",
    "level", "tax", "days", "months", "years", "month", "day",
    # plan and page-name descriptors; the entity filter already resolved these (ADR-14)
    "growth", "direct", "fund", "hdfc", "cap", "invest",
)

_GENERIC_FIELD_STEMS = _stems(" ".join(_GENERIC_FIELD_WORDS))


def _field_label(line: str) -> str:
    """The ``Label`` of a ``- Label: Value`` fact line, or the whole line if it isn't one.

    Used only to break ties toward the more specific field, so an unlabelled prose line
    simply has a long "label" and loses to a precisely-named fact.
    """
    m = _FACT_LINE_RE.match(line)
    return m.group("label").strip() if m else line


def _pick_lines(chunk: RetrievedChunk, question: str) -> list[str] | None:
    """The lines of ``chunk`` that best answer ``question``, or ``None`` if it says nothing.

    ``None`` is the important return value: this chunk does not mention the subject, and the
    caller must keep looking, or decline. Returning the chunk's opening lines instead is what
    produced "who manages it" → "SIP available: Yes" — a real field, quoted verbatim, under a
    real citation, with nothing about it looking wrong.
    """
    # The scheme name, plan and page title are dropped from the query terms used for line
    # ranking. They are not subject matter: "fund" appears in nearly every field label on a
    # scheme page, so leaving it in makes "Base expense ratio (…fund expenses)" outscore
    # "Expense ratio (TER)" for a question about the expense ratio. The entity filter has
    # already resolved *which* page we are on, so re-matching the page's own name is pure
    # noise.
    # Both sides are stemmed: intersecting exact question terms with stemmed line words
    # silently matches nothing, because "expense" is not "expe".
    #
    # Generic field words are then dropped, because they cannot on their own show that a
    # line is about the subject — see `_GENERIC_FIELD_STEMS`. If that leaves nothing, the
    # question named no field at all ("what is the ratio?") and this chunk cannot answer
    # it, so the caller must keep looking.
    terms = (
        _stems(question)
        - _stems(f"{chunk.scheme} {chunk.plan} {chunk.title}")
        - _GENERIC_FIELD_STEMS
    )
    if not terms:
        return None

    # Rank the chunk's own lines by how many query terms they contain, so the answer leads
    # with the line the user actually asked about rather than the first line of the chunk.
    #
    # Two refinements, both learned from watching this answer "exit load" with a stray
    # "base expense ratio" line attached:
    #
    # * Only lines that tie for the BEST overlap are used. Accepting every line with
    #   `overlap > 0` pulls in any line that happens to share one common word — "fund" is in
    #   almost every label on a scheme page — which is noise dressed up as an answer.
    # * Ties break toward the shorter field label, i.e. the more specific field. The corpus
    #   publishes both "Expense ratio (TER)" and "Base expense ratio (excl. additional fund
    #   expenses)"; a question about the expense ratio should get the first, not the second.
    scored: list[tuple[int, int, str]] = []
    for line in (ln.strip() for ln in chunk.text.splitlines()):
        if not line or line.startswith(chunk.scheme):
            continue
        scored.append((len(terms & _stems(line)), len(_field_label(line)), line))
    scored.sort(key=lambda triple: (-triple[0], triple[1]))

    # The selected lines must cover every content term, or the chunk is passed over.
    #
    # Measured: every question that answers correctly covers all of its terms, and the two
    # confident wrong answers each covered only a common word —
    #
    #     "what is the difference between ELSS and SIP"       terms {diff, sip}
    #       -> "- SIP available: Yes"                           covered {sip}      1 of 2
    #     "what are the steps to invest in HDFC Mutual Funds on Groww"  terms {mutu, step}
    #       -> "- Fund house: HDFC Mutual Fund"                 covered {mutu}    1 of 2
    #
    # Both are a real field, quoted verbatim, under a real citation, with nothing on screen to
    # show they are wrong — the failure mode a facts-only bot must not have. The matching line
    # came from a scheme page ranked 25th while the page that answers the question ranked 1st
    # at 0.841; "SIP" and "Mutual Fund" are just common words in field labels.
    #
    # Ranking by term *rarity* was tried instead, on the theory that a word appearing in few
    # pages is better evidence. It is wrong for this corpus, which is 5 near-identical scheme
    # pages publishing the same fields: manager 12/15, benchmark 11/15, riskometer 14/15,
    # expense 13/15. Every real field word is common, so rarity rejects correct answers and
    # breaks the fund-manager question. Overlap *count* against the terms of this question is
    # the measure that separates them.
    #
    # Coverage is spread across the selected lines, not demanded of one, because a question may
    # legitimately name two fields published separately — "expense ratio exit load" is answered
    # by the TER line and the exit-load line.
    if not scored or scored[0][0] == 0:
        return None

    covered: set[str] = set()
    selected: list[str] = []
    # `scored` is best-first, so the first pass takes the best line and each later pass takes
    # the highest-ranked line that still covers a term nothing selected has covered.
    for overlap, label_len, line in scored:
        if overlap == 0:
            break
        if not selected:
            selected = [ln for o, ll, ln in scored if (o, ll) == (overlap, label_len)]
            for ln in selected:
                covered |= terms & _stems(ln)
        else:
            gained = (terms & _stems(line)) - covered
            if gained:
                selected.append(line)
                covered |= gained
        if covered >= terms:
            break

    if covered < terms:
        return None
    return selected


def extractive_fallback(chunks: list[RetrievedChunk], question: str = "") -> Answer:
    """Answer with verbatim sentences from the retrieved chunks. No LLM, no network.

    Selection is by query-term overlap, which is crude but *auditable* — and crucially it
    cannot fabricate, because every sentence is copied from a chunk that carries a URL we
    fetched. This is what guarantees the demo completes with no ``GROQ_API_KEY``.

    It scans the pool for the first chunk that actually mentions the subject, rather than
    trusting ``chunks[0]``. Those are different questions, and the embedding model answers
    the wrong one: "Who manages HDFC Large Cap Fund?" retrieved the "Lock-in and
    availability" group at 0.841 and pushed "Scheme identity" — the group that actually
    publishes "Fund manager: Prashant Jain" — to about 13th. Over short labelled bullets
    dominated by page boilerplate, cosine similarity leaves the group order close to
    arbitrary for any question phrased in ordinary English. The entity filter has already
    fixed *which page* is in scope; choosing the group within it lexically is the same
    deterministic-before-generative principle (ADR-3) applied to retrieval.
    """
    if not chunks:
        return Answer(
            question=question,
            text=prompts.not_in_sources_message(_coverage_text()),
            citations=[],
            as_of=_latest_fetched_at(chunks),
            mode="not_in_sources",
            guard=None,
        )

    as_of = _latest_fetched_at(chunks)

    best: RetrievedChunk | None = None
    chosen: list[str] = []
    for candidate in chunks:
        picked = _pick_lines(candidate, question)
        if picked:
            best, chosen = candidate, picked
            break

    if best is None:
        # Nothing in the retrieved pool mentions the subject. Answering anyway would mean
        # quoting a field the user did not ask about, under a citation that makes it look
        # authoritative. Declining is the only honest option.
        return Answer(
            question=question,
            text=prompts.not_in_sources_message(_coverage_text()),
            citations=[],
            as_of=as_of,
            chunks=chunks,
            mode="not_in_sources",
            guard=None,
            notes=[
                f"none of the {len(chunks)} retrieved chunks mentioned a term of the "
                "question, so no line was quoted"
            ],
        )

    sentences = _sentences(" ".join(chosen))[: settings.max_sentences]
    if not sentences:
        sentences = _sentences(best.text)[: settings.max_sentences]

    return Answer(
        question=question,
        text=prompts.extractive_template(sentences, best.source_url, as_of),
        citations=[best.source_url],
        as_of=as_of,
        chunks=chunks,
        mode="extractive",
        guard=None,
        notes=["deterministic extractive fallback — no LLM was called"],
    )


# ── Groq ─────────────────────────────────────────────────────────────────────


def call_groq(messages: list[dict], as_of: str) -> str | None:
    """One Groq call. Returns the text, or ``None`` on any failure whatsoever.

    ``None`` is the only failure signal, deliberately: a missing key, a rate limit, a
    timeout, a 500, and a malformed response all take the same path to
    :func:`extractive_fallback`. A live demo must never show a traceback because a third-party
    API had a bad minute (ADR-2).
    """
    if not settings.groq_api_key:
        logger.info("GROQ: no API key configured — using the extractive fallback")
        return None
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception as exc:  # noqa: BLE001 - a demo must not die on a provider error
        logger.warning("GROQ: call failed (%s: %s) — using the extractive fallback",
                       type(exc).__name__, exc)
        return None


# ── The post-check pipeline ──────────────────────────────────────────────────


def post_check(
    text: str,
    chunks: list[RetrievedChunk],
    question: str = "",
    as_of: str = "",
) -> Answer | None:
    """Apply the seven checks in order. ``None`` means "unusable, fall back".

    Returns the accepted :class:`Answer` on success. Returning ``None`` is reserved for the two
    trust failures (1 and 2) plus the two refusals (4 and 5); length and footer problems are
    repaired in place.
    """
    if not text or not text.strip():
        return None

    as_of = as_of or _latest_fetched_at(chunks)
    notes: list[str] = []
    body = text.strip()

    # ── Check 1: is there a citation at all? ─────────────────────────────────
    urls = _urls_in(body)
    if not urls:
        notes.append("check 1 failed: no citation")
        return None

    # ── Check 2: is the citation real? ──────────────────────────────────────
    from .sources import valid_urls

    allowed = valid_urls()
    real = [u for u in urls if u in allowed]
    if not real:
        invented = [u for u in urls if u not in allowed]
        notes.append(f"check 2 failed: citation not in sources.csv: {invented[:2]}")
        return None
    if len(real) > 1:
        notes.append(f"trimmed to the first of {len(real)} citations")
        body = _keep_first_url(body, real[0])

    # ── Check 3: at most 3 sentences ─────────────────────────────────────────
    if len(_sentences(body)) > settings.max_sentences:
        notes.append(f"check 3: truncated to {settings.max_sentences} sentences")
        body = _truncate_to_sentences(body, settings.max_sentences)
        # A truncation can drop the very citation the answer was carrying.
        if not _urls_in(body):
            body = f"{body}\n\nSource: {real[0]}"

    # ── Check 4: advice language ─────────────────────────────────────────────
    if _ADVICE_RE.search(body):
        notes.append("check 4 failed: advice language")
        return Answer(
            question=question,
            text=prompts.REFUSAL_TEXT,
            citations=[],
            as_of=as_of,
            chunks=chunks,
            mode="refusal",
            guard="advice",
            notes=notes,
        )

    # ── Check 5: performance claims ──────────────────────────────────────────
    offender = _performance_claim(body)
    if offender:
        notes.append(f"check 5 failed: performance claim {offender[:60]!r}")
        return Answer(
            question=question,
            text=prompts.REFUSAL_TEXT,
            citations=[],
            as_of=as_of,
            chunks=chunks,
            mode="refusal",
            guard="returns",
            notes=notes,
        )

    # ── Checks 6 and 7: footer and disclaimer ────────────────────────────────
    before = body
    body = _append_footer(body, as_of)
    if body != before:
        notes.append("checks 6/7: appended footer and/or disclaimer")

    return Answer(
        question=question,
        text=body,
        citations=[real[0]],
        as_of=as_of,
        chunks=chunks,
        mode="llm",
        guard=None,
        notes=notes,
    )


def _keep_first_url(text: str, keep: str) -> str:
    """Drop every URL except ``keep``, so the answer carries exactly one citation (rule 3)."""
    return _URL_RE.sub(lambda m: m.group(0) if m.group(0).rstrip(".,;") == keep else "", text)


# ── Shared context helpers ───────────────────────────────────────────────────


def _latest_fetched_at(chunks: list[RetrievedChunk]) -> str:
    """The most recent ``fetched_at`` across the evidence, as ``YYYY-MM-DD``.

    This is the "as of" date, and it is the date the *sources* were fetched — not today's
    date. Stamping today's date on an answer built from a page scraped last week would be a
    small lie, and this system's entire value is that its numbers are traceable.
    """
    dates = sorted({c.fetched_at for c in chunks if c.fetched_at})
    return dates[-1] if dates else ""


def _coverage_text() -> str:
    """Bullet list of what the corpus covers, for the out-of-scope answer (P4)."""
    from .sources import PRIMARY_SCHEMES, by_role

    lines = [f"- {s.scheme} — {s.title}" for s in PRIMARY_SCHEMES]
    context = [s.title for role in ("education", "regulatory", "amc", "category", "tool")
               for s in by_role(role)]
    lines += [f"- {t} (explainer page)" for t in dict.fromkeys(context)]
    lines.append("- Fees, exit load, minimum SIP and lump sum, lock-in, riskometer, benchmark")
    return "\n".join(lines)


# ── The entry point ──────────────────────────────────────────────────────────


def _lead_with_on_topic(pool: list[RetrievedChunk], question: str, top_k: int) -> list:
    """Reorder the pool so the chunk that actually answers the question comes first.

    The LLM and the extractive fallback were looking at different pools: the fallback scanned
    all ``extractive_pool`` chunks, the model got only the first ``top_k``. For "Who manages
    HDFC Small Cap Fund?" the "Scheme identity" group — the one publishing "Fund manager:
    Chirag Setalvad" — sat around 13th overall, so it was in the fallback's pool and absent
    from the model's. The model then answered "the information about the fund manager is not
    provided in the retrieved context", which is *true of its context* and false of the
    corpus. A correct fact, a real citation, and a confidently wrong answer.

    Rather than widen the model's context to the whole pool — which trades one failure for
    more chances to cite the wrong block — this reorders the same shortlist so the group
    that mentions the subject leads. Same evidence, better ordering, and the ordering is
    decided by the same deterministic test the fallback uses.
    """
    head = next((c for c in pool if _pick_lines(c, question)), None)
    if head is None:
        return list(pool[:top_k])
    ordered = [head] + [c for c in pool if c is not head]
    return ordered[:top_k]


#: Ceiling for the one-shot widening below. Large enough to reach a group that normally
#: ranks around 13th ("Scheme identity", which publishes the fund manager), small enough
#: that a second embedding call is not spent on questions that were never going to match.
_MAX_POOL = 30


def answer_question(
    question: str,
    top_k: int | None = None,
    memory: ConversationMemory | None = None,
) -> Answer:
    """Answer one question, with every guard and every fallback in the right order.

    1. guards — PII, advice, returns, comparative, speculative. Never calls the LLM (P2).
    2. retrieve — entity filter, cosine, MMR, threshold. Empty means "not in my sources" (P4).
    3. the LLM, if a key exists and the call succeeds.
    4. seven post-checks, in order.
    5. the deterministic extractive fallback whenever 3 or 4 cannot be trusted.

    ``memory`` is optional conversational context (STAGE 5c). It affects step 2 only, and only
    when the question names no scheme of its own. Critically, the guards in step 1 see the
    question **exactly as typed**: a rewrite that borrowed a previous subject must not be able
    to talk its way past a guard, and must not change what the user is shown either.
    """
    # ── 1 · Guards. First, before anything touches the network or the corpus. ──
    # On the raw question, always. Memory rewrites the retrieval query; it must never be able
    # to rewrite the rules.
    verdict = check_guards(question)
    if verdict:
        return Answer(
            question=question,
            text=verdict.message,
            citations=[],
            as_of=_latest_fetched_at([]),
            mode="pii_refusal" if verdict.kind == "pii" else "refusal",
            guard=verdict.kind,
            notes=[f"guard fired before retrieval; logged as {verdict.redacted_question!r}"],
        )

    # ── 2 · Retrieval ────────────────────────────────────────────────────────
    # A follow-up ("what about its exit load?") names no scheme, so entity resolution has
    # nothing to work with and similarity alone decides which page is cited — which is how a
    # Direct Growth question ends up on the regular-growth page. Memory supplies the one thing
    # the user did say: which fund they meant. The rewritten query is reported in `notes` so
    # the UI can show what was actually searched.
    query = question
    if memory is not None:
        query = memory.contextualise(question)
    # One embedding call serves both pools: the LLM sees the top `top_k`, the deterministic
    # fallback scans the deeper `extractive_pool` for the one group that mentions the subject.
    depth = max(top_k or settings.top_k, settings.extractive_pool)
    pool = retrieve(query, top_k=depth)
    chunks = pool[: top_k or settings.top_k]
    if not chunks:
        return Answer(
            question=question,
            text=prompts.not_in_sources_message(_coverage_text()),
            citations=[],
            as_of="",
            mode="not_in_sources",
            guard=None,
        )

    as_of = _latest_fetched_at(chunks)
    # Surfaced so the UI can show what was actually searched. A retrieval rewrite the user
    # cannot see is indistinguishable from the bot having guessed.
    rewrite_note = (
        [] if query.strip() == question.strip() else [f"follow-up resolved to: {query}"]
    )

    # ── 3 · The LLM ──────────────────────────────────────────────────────────
    # The model is given the *retrieval* query, so it can see the subject the chunks are
    # about. The Answer still reports `question` — what the user actually typed.
    # `llm_chunks` leads with the group that actually mentions the subject, and is also
    # what post-check validates against, since it is the context the model was shown.
    llm_chunks = _lead_with_on_topic(pool, query, top_k or settings.top_k)
    messages = prompts.build_messages(query, llm_chunks, as_of)
    raw = call_groq(messages, as_of)

    if raw is None:
        fallback = extractive_fallback(pool, query)
        # Nothing in the pool mentioned the subject. Widen once before declining.
        #
        # A field's rank in an approximate index is not a property of the question, it is a
        # property of the index and the day's HNSW graph — so "Scheme identity" (which
        # publishes the fund manager) can sit at 12th on one run and 13th on the next. That
        # makes a fact the corpus plainly contains alternate between answered and declined,
        # which is the worst kind of bug for a live demo: it looks like a bad question.
        #
        # The escalation only ever *widens*, never narrows, and is bounded to one extra
        # retrieval, so the cost is one embedding call on the rare declining path and the
        # behaviour stays deterministic in outcome. It cannot smuggle in a worse answer: the
        # same on-topic test still has to pass, against a larger pool.
        if fallback.mode == "not_in_sources" and depth < _MAX_POOL:
            wider = retrieve(query, top_k=_MAX_POOL)
            retry = extractive_fallback(wider, query)
            if retry.mode != "not_in_sources":
                retry.notes.append(
                    f"widened the search from {depth} to {_MAX_POOL} chunks to find the field"
                )
                retry.notes.append(f"llm unavailable (model={settings.llm_model})")
                retry.notes.extend(rewrite_note)
                return retry

        fallback.notes.append(f"llm unavailable (model={settings.llm_model})")
        fallback.notes.extend(rewrite_note)
        return fallback

    # ── 4 · Post-checks ──────────────────────────────────────────────────────
    checked = post_check(raw, llm_chunks, question=question, as_of=as_of)
    if checked is None:
        fallback = extractive_fallback(pool, query)
        fallback.notes.append("post-check rejected the LLM answer")
        fallback.notes.extend(rewrite_note)
        return fallback

    checked.notes.extend(rewrite_note)
    return checked


if __name__ == "__main__":
    import sys

    probes = sys.argv[1:] or [
        "expense ratio of HDFC Large Cap Fund Direct Growth",
        "Is HDFC ELSS Tax Saver Fund locked in?",
        "Should I buy HDFC Small Cap Fund?",
        "What is the weather in Delhi?",
        "my PAN is ABCDE1234F",
    ]
    for probe in probes:
        result = answer_question(probe)
        print("=" * 78)
        print(f"Q: {probe}")
        print(f"mode: {result.mode}   guard: {result.guard}   as_of: {result.as_of}")
        if result.citations:
            print(f"citation: {result.citations[0]}")
        print("-" * 78)
        print(result.text)
        for note in result.notes:
            print(f"  note: {note}")
