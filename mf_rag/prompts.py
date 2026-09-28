"""STAGE 6 — Prompt construction and the exact wording of every user-facing string.

Kept separate from ``answerer`` so the *language* the model is given is reviewable in one file
without reading any control flow. Every string a user can see lives here, and the disclaimer,
refusal and privacy copy live in ``guards`` — those three are compliance text fixed by PRD
§10.2 and must not drift.

Design decisions worth knowing before changing anything here:

* **All 8 rules from architecture §11.1 are present.** Two are load-bearing beyond the obvious.
  Rule 6 bounds ``page_role: education`` blocks to explanation only, so a concept page can
  never become the source of a scheme number (P3 / ADR-10). Rule 7 tells the model to *say a
  field is missing* rather than guess — a negative instruction that pushes toward P4 instead of
  toward a fluent fabrication.
* **The format instruction is restated after the question**, not only in the system prompt.
  Recency matters: the last thing the model reads should be the format it must follow.
* **Numbered blocks with explicit metadata** let the model cite block *n*, which post-check 2
  maps back to a URL that we actually fetched with HTTP 200. That is what turns "exactly one
  real citation" from a hope into a mechanical check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import settings

# The refusal and privacy copy is re-exported rather than restated: it is compliance text
# fixed by PRD §10.2, it lives in `guards` so that the module which decides *whether* to
# refuse is the same module that owns the words, and a second copy here would be a second
# thing to keep in sync. There is no import cycle — `guards` imports only `config`.
from .guards import PRIVACY_NOTICE, REFUSAL

if TYPE_CHECKING:  # prompt construction is pure string work; no need to load torch to print it
    from .retriever import RetrievedChunk

#: Deliverable D6. Appended to every answer by post-check 7.
DISCLAIMER = (
    "Facts-only. No investment advice. Answers are generated from public HDFC "
    "Mutual Fund pages on Groww and may be outdated. Verify every number on the "
    "linked source page before acting. Mutual fund investments are subject to "
    "market risks; read all scheme-related documents carefully."
)

#: Linked by both refusal verdicts, per PRD §10.2.
FACTSHEET_LINK = settings.factsheet_link
EDUCATION_LINK = settings.education_link

#: The single refusal body, used by the guards *and* by post-checks 4 and 5, so a refusal
#: reached after generation is worded identically to one reached before it.
REFUSAL_TEXT = REFUSAL

#: The required answer footer. Post-check 6 asserts its presence.
AS_OF_TEMPLATE = "Last updated from sources: {date}"

SYSTEM_PROMPT = """You are a facts-only assistant for HDFC Mutual Fund scheme pages on Groww.

Rules:
1. Answer ONLY from the numbered context blocks below. If the answer is not in
   them, say: "I don't have that in my sources." Do not use outside knowledge.
2. Be factual and concise: maximum 3 sentences. No preamble, no summary,
   no "Great question".
3. End with exactly one source URL taken from the context block you used.
4. Never give investment advice. Never say whether to buy, sell, hold, or which
   scheme is better. If the question asks that, reply with exactly: "I can help
   you with the factual details." Do not explain why, do not suggest another
   question, and do not include a link.
5. Never state, calculate, or compare returns or performance figures. If asked,
   reply with exactly: "I can help you with the factual details." Same rule:
   no commentary, no alternative question, no link.
6. Context blocks with page_role "education" explain concepts and processes only.
   Never take a scheme number (expense ratio, exit load, minimum SIP) from them.
7. If a field is genuinely missing from the context, say it is missing and link
   the scheme page so the user can check. Never guess a number.
8. Do not write the disclaimer yourself. A later check appends it verbatim, so
   repeating it wastes one of your 3 sentences.
8. Finish with: "Last updated from sources: <date>".
"""


def system_prompt() -> str:
    """The system message. Kept as a function so the date can be interpolated if ever needed."""
    return SYSTEM_PROMPT


def as_of_line(as_of: str) -> str:
    """The required footer, e.g. ``Last updated from sources: 2026-09-28``."""
    return AS_OF_TEMPLATE.format(date=as_of)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered context blocks. Delegates to the retriever so the shape lives in one place.

    ``page_role`` is included in the prompt *text*, not only in metadata, because rule 6 can
    only be followed by a model that can see the field (P3 / ADR-10).
    """
    from .retriever import build_context as _build

    return _build(chunks)


def user_prompt(question: str, chunks: list[RetrievedChunk], as_of: str) -> str:
    """The user message: context, then question, then the format instruction restated."""
    return (
        f"Context blocks:\n{build_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Answer in at most {settings.max_sentences} sentences, then one source URL, then "
        f'"{as_of_line(as_of)}".'
    )


def build_messages(
    question: str, chunks: list[RetrievedChunk], as_of: str
) -> list[dict[str, str]]:
    """The full message list for the Groq call, in OpenAI chat format."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(question, chunks, as_of)},
    ]


def extractive_template(sentences: list[str], url: str, as_of: str) -> str:
    """The no-LLM fallback answer (architecture §11.3, principle P8).

    This is what guarantees the demo completes with no ``GROQ_API_KEY``, no network, and no
    model download — so the output is a *verbatim* selection from the retrieved chunk, never a
    paraphrase. A fabricated citation is impossible here by construction: the URL is the
    retrieved chunk's own ``source_url``.
    """
    body = " ".join(sentences).strip()
    return f"{body}\n\nSource: {url}\n{as_of_line(as_of)}"


def not_in_sources_message(coverage: str) -> str:
    """The honest out-of-scope answer (P4).

    Names what the corpus actually covers, because "I don't know" without saying what it does
    know is a dead end for the user, and because listing the real coverage is itself evidence
    that the refusal is a scope statement rather than a failure.
    """
    return (
        "I don't have that in my sources. I only answer from a fixed set of public HDFC "
        f"Mutual Fund pages on Groww.\n\nWhat I cover:\n{coverage}"
    )


if __name__ == "__main__":
    print("SYSTEM PROMPT")
    print("=" * 72)
    print(system_prompt())
    print()
    print(f"rule count: {sum(1 for line in system_prompt().splitlines() if line[:2].rstrip('.').isdigit())}")
    print(f"disclaimer chars: {len(DISCLAIMER)}")
    print(f"as_of sample: {as_of_line('2026-09-28')}")
