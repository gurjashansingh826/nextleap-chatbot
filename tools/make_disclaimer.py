"""Generate ``docs/disclaimer.md`` from the live constants in ``mf_rag.prompts``.

The disclaimer is a compliance artefact. If the copy in the docs and the copy the app
actually emits diverge, the one in the docs is the one a reviewer reads and the one that
matters. So the doc is generated from the same string constant, and this script asserts that
what it wrote is what the running app would print.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mf_rag import prompts  # noqa: E402
from mf_rag.answerer import Answer  # noqa: E402
from mf_rag.guards import FACT_ONLY, PRIVACY_NOTICE  # noqa: E402


def main() -> None:
    # Take the strings off a real Answer, not off the module, so this doc is proof the
    # pipeline emits them rather than a restatement of intent.
    real = Answer(question="x", text="", citations=[], as_of="2026-09-28")
    assert real.disclaimer == prompts.DISCLAIMER, "Answer default drifted from prompts.DISCLAIMER"
    # The system prompt instructs the model not to write the disclaimer itself; post-check 7
    # appends the exact string. So the invariant to assert is that the prompt *defers*, not
    # that it embeds the text — an LLM reproducing the disclaimer from memory is exactly the
    # drift the post-check exists to prevent.
    assert "disclaimer" in prompts.SYSTEM_PROMPT.lower(), "system prompt never mentions the disclaimer"
    assert prompts.DISCLAIMER not in prompts.SYSTEM_PROMPT, (
        "the disclaimer is appended by post-check 7, not asked for in the prompt; embedding it "
        "invites the model to paraphrase it"
    )
    for banned in ("factsheet", "factsheet_link"):
        assert banned not in prompts.SYSTEM_PROMPT.lower(), (
            f"the system prompt still offers a link ({banned}); the refusal must carry none"
        )
    assert prompts.REFUSAL == FACT_ONLY, "prompts.REFUSAL drifted from guards.FACT_ONLY"

    footer = prompts.as_of_line("2026-09-28")
    lines = [
        "# Disclaimer and compliance copy",
        "",
        f"Generated {date.today().isoformat()} by `tools/make_disclaimer.py`.",
        "",
        "Every string below is read from the constants the running app uses, and the script "
        "asserts they agree. These are not transcriptions.",
        "",
        "## 1. The answer footer (D6)",
        "",
        "Appended to **every** factual answer by post-check 6, and regenerated from the newest",
        "`fetched_at` in the retrieved chunks rather than from the clock. That is the point:",
        "the footer reports how old the *source* is, not how long ago the bot was asked.",
        "",
        "```",
        footer,
        "```",
        "",
        "## 2. The disclaimer",
        "",
        "Appended to every factual answer by post-check 7, and carried as the `disclaimer`",
        "field on every `Answer` so any UI cannot forget it.",
        "",
        "```",
        prompts.DISCLAIMER,
        "```",
        "",
        "## 3. The refusal",
        "",
        "One line, for every question that is not asking for a published fact — advice, "
        "returns, comparisons and hypotheticals alike:",
        "",
        "```",
        FACT_ONLY,
        "```",
        "",
        "No link, no suggestion of a better question, no description of what the bot can do. "
        "Each of those would be guidance, and this bot does not give guidance.",
        "",
        "## 4. The privacy notice",
        "",
        "Shown instead when the question contains — or asks for — personal data:",
        "",
        "```",
        PRIVACY_NOTICE,
        "```",
        "",
        "This one keeps its own wording. It tells the user their data was not stored, which "
        "is a factual claim about this system and therefore exactly the kind of claim the bot "
        "is permitted to make.",
        "",
        "## 5. What the bot will not do",
        "",
        "Each of these is enforced in code, not in the prompt, and each has a test:",
        "",
        "| Behaviour | Where | Test |",
        "| --- | --- | --- |",
        "| Give investment advice | `guards.check_guards` | `test_advice_questions_are_refused` |",
        "| Quote or rank returns | `guards.check_guards`, `answerer._performance_claim` | "
        "`test_performance_claim_in_the_answer_becomes_a_refusal` |",
        "| Compare two schemes | `guards.check_guards` | `test_comparative_questions_are_refused` |",
        "| Answer a hypothetical | `guards.check_guards` | `test_hypothetical_questions_are_refused` |",
        "| Store PAN, Aadhaar, OTP, account, email, phone | `guards.PII_PATTERNS` | "
        "`test_pii_is_detected` |",
        "| Supply a personal identifier it does not hold | `guards.PII_ELICITATION` | "
        "`test_asking_for_personal_records_is_refused` |",
        "| Answer without a citation | `answerer.post_check` 1-2 | "
        "`test_answer_without_a_citation_is_rejected` |",
        "| Exceed 3 sentences | `answerer.post_check` 3 | "
        "`test_long_answer_is_truncated_and_keeps_its_citation` |",
        "| Cite a URL outside the registry | `answerer.post_check` 2 | "
        "`test_invented_source_is_rejected` |",
        "",
        "Performance tables are also stripped at ingestion, so a return figure is not in the "
        "corpus to be quoted even if a prompt instruction were ignored.",
        "",
        "## 6. Where the facts come from",
        "",
        "Public HDFC Mutual Fund pages on Groww, listed in `docs/sources.md`. Nothing is "
        "hand-typed: a fact that is not on a fetched page cannot reach an answer. Figures are",
        "as-published and may be out of date, which is what the as-of footer is for — verify",
        "any number on the linked page before acting on it.",
        "",
    ]

    out = ROOT / "docs" / "disclaimer.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} (assertions passed)")


if __name__ == "__main__":
    main()
