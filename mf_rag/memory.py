"""STAGE 5c — conversational memory, bounded to the last ``memory_turns`` exchanges.

**This is a scope addition.** PRD §3 lists "no saved chat history" among the exclusions, and
none of PRD.md / architecture.md / implementation.md defines a memory or a context window. It
is here because it was asked for, and it is deliberately built to weaken nothing that came
before: it resolves *which page* a follow-up is about, and it cannot add a fact, a citation or
a sentence to an answer.

Why it is needed at all
-----------------------
"what about its exit load?" is unanswerable as written. The entity filter in
``retriever.resolve_scheme`` looks for a scheme name, finds none, and the query falls back to
pure similarity — which on this corpus is how a Direct Growth question ends up citing the
regular-growth page. The missing piece is the one thing the user did say: *which* fund they
meant, in the previous turn. A one-line history resolves it deterministically, with no LLM
call and no second chance to get it wrong.

The three rules that keep this safe
----------------------------------
1. **Memory never changes what the bot is allowed to say.** Guards run on the raw question,
   before any rewrite, and the answer's text is built only from retrieved chunks. The rewrite
   changes the *query*, never the rules or the evidence.
2. **Memory is never persisted.** ``ConversationMemory`` lives in ``st.session_state`` and is
   dropped when the browser tab closes. Nothing is written to disk, which is what PRD §3
   requires and what keeps PII out of any log.
3. **Nothing identifiable enters the memory.** A turn is only recorded when no PII guard
   fired, and :meth:`ConversationMemory.add` re-checks with ``detect_pii`` rather than
   trusting its caller. Defence in depth, because this is the one component whose whole job is
   to hold on to text.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import settings
from .guards import detect_pii

#: A question is treated as a follow-up only if it looks like one. Each of these marks the
#: user referring backwards rather than naming a fresh subject.
_FOLLOWUP_MARKERS = (
    "what about",
    "how about",
    "and its",
    "and the",
    "its ",
    "their ",
    "same for",
    "that one",
    "this one",
    "there",
    "here",
    "too",
    "also",
)

#: Second-person references to the previously discussed scheme.
_PRONOUNS = ("it", "its", "they", "them", "this", "that", "these", "those")

#: Openings that name a *concept* rather than a field of a fund. A question that asks what
#: something is, or how two things differ, is about the idea — it must never borrow a subject
#: from the previous turn, or the bot answers a definition with a scheme's expense ratio.
_CONCEPT_MARKERS = (
    "difference between",
    "what is a ",
    "what is an ",
    "what are ",
    "vs.",
    " vs ",
    "compare",
    "explain",
    "tell me about",
    "which is better",
    "how does a ",
    "how do i ",
)

#: A bare field lookup with no subject, e.g. "what is the benchmark". Long enough to be a
#: question, short enough that the subject is obviously carried over from the last turn.
_LOOKUP_MAX_WORDS = 8


@dataclass(frozen=True)
class Turn:
    """One exchange, reduced to the only two things a later query might need.

    Deliberately not the full text. Keeping answers here would grow the memory without
    improving resolution, and would mean holding a second copy of every fact the bot has ever
    said — which is exactly the kind of copy that eventually disagrees with the source.
    """

    question: str
    scheme: str
    as_of: str
    mode: str

    @property
    def is_answerable(self) -> bool:
        """Only turns that actually answered something can lend their subject to a follow-up.

        A refusal has no subject to lend. If "Should I buy HDFC Small Cap?" set the context,
        the next "what about its exit load?" would silently inherit a fund the bot refused to
        discuss — the one way this feature could leak a decision into a later factual answer.
        """
        return self.mode in {"llm", "extractive"}


@dataclass
class ConversationMemory:
    """A bounded window over recent turns. The bound is the feature, not a detail.

    ``maxlen`` on the deque means the oldest turn is *dropped*, not summarised. A summary
    would be generated text describing what the bot said, which is a weaker and less
    inspectable input than the original turn, and the one place a memory of a conversation
    starts inventing things.
    """

    maxlen: int = field(default_factory=lambda: settings.memory_turns)
    turns: deque[Turn] = field(default_factory=deque)

    def __post_init__(self) -> None:
        # Bound honoured at construction, so a caller passing maxlen=0 gets no memory rather
        # than an unbounded deque that happens to be empty.
        if self.turns.maxlen != self.maxlen:
            self.turns = deque(self.turns, maxlen=self.maxlen)

    # ── writing ──────────────────────────────────────────────────────────────

    def add(self, turn: Turn) -> bool:
        """Record a turn. Returns False if it was withheld, and says why in the docstring.

        Withheld when: the memory window is closed, the turn is a refusal (see
        :attr:`Turn.is_answerable`), or the question contains PII.
        """
        if self.maxlen <= 0:
            return False
        if not turn.is_answerable:
            return False
        if detect_pii(turn.question):
            # Callers are expected to have guarded already. Re-checking here is cheap and is
            # the difference between "we don't log PII" being an invariant and being a habit.
            return False
        self.turns.append(turn)
        return True

    def add_from_answer(self, answer) -> bool:
        """Convenience wrapper that takes only the scheme slug and the redacted question.

        ``answer.question`` is used rather than the raw input, because by the time an Answer
        exists the raw text has already been through the guard and may have been redacted.
        """
        slug = answer.chunks[0].source_url.rstrip("/").rsplit("/", 1)[-1] if answer.chunks else ""
        return self.add(
            Turn(
                question=answer.question,
                scheme=slug,
                as_of=answer.as_of,
                mode=answer.mode,
            )
        )

    # ── reading ──────────────────────────────────────────────────────────────

    def recent(self, n: int | None = None) -> list[Turn]:
        """The last ``n`` answerable turns, oldest first."""
        items = [t for t in self.turns if t.is_answerable]
        return items[-n:] if n else items

    def last_scheme(self) -> str:
        """Slug of the most recent turn that named a scheme and produced an answer."""
        for turn in reversed(self.turns):
            if turn.scheme:
                return turn.scheme
        return ""

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

    # ── the actual purpose ───────────────────────────────────────────────────

    def contextualise(self, question: str) -> str:
        """Rewrite a follow-up into a self-contained retrieval query.

        Returns ``question`` unchanged unless all three conditions hold:

        1. the question names no scheme of its own — if it does, it is not a follow-up and
           inventing a subject for it would be wrong;
        2. it reads like a follow-up — a pronoun, one of the marker phrases, or a bare field
           lookup of at most :data:`_LOOKUP_MAX_WORDS` words;
        3. it is not a concept question (:data:`_CONCEPT_MARKERS`);
        4. the memory actually holds a scheme to borrow.

        The rewrite is a concatenation with the resolved scheme *name*, not the slug, because
        this string is both embedded and matched by name — and it is visible in the UI, so a
        reviewer can see exactly what was searched.
        """
        from .sources import scheme_name_for_slug  # local import: keeps sources a leaf

        if not question.strip() or len(self) == 0:
            return question

        # Condition 1. If the user named a scheme, that is the subject. Never override it.
        from .retriever import resolve_scheme

        if resolve_scheme(question)["slugs"]:
            return question

        # Condition 2. A pronoun or marker is the strong signal. A short bare lookup — "what is
        # the benchmark" — is the weaker one: it names a *field* and nothing else, so the
        # subject has to come from the last turn. Left unrewritten it drifted to whichever
        # scheme happened to score highest, which is not an answer to the question asked.
        lowered = f" {question.lower().strip()} "
        pronoun = any(f" {p} " in lowered for p in _PRONOUNS)
        marker = any(m in lowered for m in _FOLLOWUP_MARKERS)
        words = question.split()
        short_lookup = len(words) <= _LOOKUP_MAX_WORDS
        if not (pronoun or marker or short_lookup):
            return question

        # Condition 3. A concept question keeps its independence, so the rewrite cannot turn a
        # definition into a scheme figure.
        if any(c in lowered for c in _CONCEPT_MARKERS):
            return question

        # Condition 4.
        slug = self.last_scheme()
        if not slug:
            return question
        name = scheme_name_for_slug(slug)
        if not name:
            return question

        stem = question.strip().rstrip("?").strip()
        # Appended, not prefixed. "HDFC Large Cap Fund: and the minimum SIP" reads like a
        # label and embeds worse than the sentence a user would actually have typed, which
        # measurably changed which fact group came back. The subject belongs where it belongs
        # in an English question, not bolted on the front.
        return f"{stem} for {name}"
