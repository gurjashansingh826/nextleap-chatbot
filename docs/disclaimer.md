# Disclaimer and compliance copy

Generated 2026-09-28 by `tools/make_disclaimer.py`.

Every string below is read from the constants the running app uses, and the script asserts they agree. These are not transcriptions.

## 1. The answer footer (D6)

Appended to **every** factual answer by post-check 6, and regenerated from the newest
`fetched_at` in the retrieved chunks rather than from the clock. That is the point:
the footer reports how old the *source* is, not how long ago the bot was asked.

```
Last updated from sources: 2026-09-28
```

## 2. The disclaimer

Appended to every factual answer by post-check 7, and carried as the `disclaimer`
field on every `Answer` so any UI cannot forget it.

```
Facts-only. No investment advice. Answers are generated from public HDFC Mutual Fund pages on Groww and may be outdated. Verify every number on the linked source page before acting. Mutual fund investments are subject to market risks; read all scheme-related documents carefully.
```

## 3. The refusal

One line, for every question that is not asking for a published fact — advice, returns, comparisons and hypotheticals alike:

```
I can help you with the factual details.
```

No link, no suggestion of a better question, no description of what the bot can do. Each of those would be guidance, and this bot does not give guidance.

## 4. The privacy notice

Shown instead when the question contains — or asks for — personal data:

```
Please don't share personal details. This assistant doesn't accept or store PAN, Aadhaar numbers, account numbers, OTPs, email addresses, or phone numbers. Your question was not saved. For account-specific queries, use the Groww app or your registered adviser.
```

This one keeps its own wording. It tells the user their data was not stored, which is a factual claim about this system and therefore exactly the kind of claim the bot is permitted to make.

## 5. What the bot will not do

Each of these is enforced in code, not in the prompt, and each has a test:

| Behaviour | Where | Test |
| --- | --- | --- |
| Give investment advice | `guards.check_guards` | `test_advice_questions_are_refused` |
| Quote or rank returns | `guards.check_guards`, `answerer._performance_claim` | `test_performance_claim_in_the_answer_becomes_a_refusal` |
| Compare two schemes | `guards.check_guards` | `test_comparative_questions_are_refused` |
| Answer a hypothetical | `guards.check_guards` | `test_hypothetical_questions_are_refused` |
| Store PAN, Aadhaar, OTP, account, email, phone | `guards.PII_PATTERNS` | `test_pii_is_detected` |
| Supply a personal identifier it does not hold | `guards.PII_ELICITATION` | `test_asking_for_personal_records_is_refused` |
| Answer without a citation | `answerer.post_check` 1-2 | `test_answer_without_a_citation_is_rejected` |
| Exceed 3 sentences | `answerer.post_check` 3 | `test_long_answer_is_truncated_and_keeps_its_citation` |
| Cite a URL outside the registry | `answerer.post_check` 2 | `test_invented_source_is_rejected` |

Performance tables are also stripped at ingestion, so a return figure is not in the corpus to be quoted even if a prompt instruction were ignored.

## 6. Where the facts come from

Public HDFC Mutual Fund pages on Groww, listed in `docs/sources.md`. Nothing is hand-typed: a fact that is not on a fetched page cannot reach an answer. Figures are
as-published and may be out of date, which is what the as-of footer is for — verify
any number on the linked page before acting on it.
