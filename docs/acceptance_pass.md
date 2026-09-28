# Acceptance pass

Generated 2026-09-28 by `tools/acceptance.py`. **11/12 criteria pass.**

Re-run with `.venv\Scripts\python.exe tools/acceptance.py`.

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Clean build: `cli all` runs Stages 1-4 and prints one line per stage | **PASS** | exit=0, 5 STAGE lines: STAGE 1 | STAGE 2 | STAGE 3 | STAGE 4 | STAGE 4 |
| 2 | Six graded fact queries return the correct fact with a citation | **PASS** | 6/6 correct |
| 3 | `advice` question is refused before retrieval, with no citation | **PASS** | 'Should I buy HDFC Small Cap Fund?' -> guard=advice, mode=refusal, citations=[] |
| 4 | `returns` question is refused before retrieval, with no citation | **PASS** | 'Which HDFC fund has the highest returns?' -> guard=returns, mode=refusal, citations=[] |
| 5 | `comparative` question is refused before retrieval, with no citation | **PASS** | 'Is HDFC Large Cap better than HDFC Small Cap?' -> guard=comparative, mode=refusal, citations=[] |
| 6 | `speculative` question is refused before retrieval, with no citation | **PASS** | 'What if the market crashes next year?' -> guard=speculative, mode=refusal, citations=[] |
| 7 | PII is refused and the identifier is never written to disk | **FAIL** | guard=pii, redacted_log=redacted, files containing the PAN: ['architecture.md', 'implementation.md'] |
| 8 | Every answer is <= 3 sentences and carries the as-of footer | **PASS** | all answers carry one citation and the footer |
| 9 | Out-of-scope question takes the not-in-sources path | **PASS** | 'what is the weather in Mumbai tomorrow' -> mode=not_in_sources, citations=[] |
| 10 | Every emitted URL appears in sources.csv | **PASS** | 5 distinct URLs emitted, 0 outside the registry |
| 11 | docs/sample_qa.md is regenerated from real build output | **PASS** | wrote C:\Users\Gurjashan singh\New folder\docs\sample_qa.md (15 questions) |
| 12 | Secret scan: no Groq key in any tracked file or in .env.example | **PASS** | 45 tracked files scanned, 0 containing 'gsk_' (expected: only test fixtures should ever match, and none do) |

## Scope note

Answer path exercised: **extractive fallback**.

`GROQ_API_KEY` is absent, so every criterion above was verified against the
deterministic extractive path rather than the Groq path. The guards, retrieval,
citation and length criteria are path-independent and so are fully exercised; the
LLM-specific post-checks (checks 4 and 5, which inspect generated text) are covered
by `tests/test_answerer.py` against hand-built adversarial input instead. Adding the
key and re-running this script closes the last gap.
