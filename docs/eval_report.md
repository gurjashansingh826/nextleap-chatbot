# Gate 2 evaluation report

Generated 2026-09-28 against commit `65ae62d` plus the untracked STAGE 5/6/7 work.
Reproduce with `python -m mf_rag.cli eval`.

## What this report is for

Gate 2 exists to stop two specific failures before they reach the demo: picking a chunking
strategy by intuition, and picking a similarity threshold by intuition. Both are decisions
that look like configuration and behave like claims, so both were measured instead.

The headline: **the chunking strategy was confirmed, and the threshold decision did not
survive contact with the data.** Sections 3 and 4 explain both.

## 1. Method

- **Corpus:** 15 fetched pages, 5 in scope (HDFC Direct Growth schemes), 10 supporting.
- **Eval set:** `eval/chunking_eval.json`, 36 rows. 30 in-scope (q01–q25 scheme facts,
  c01–c05 concept/procedural) and 6 out-of-scope probes (x01–x06) used only for threshold
  calibration. Every in-scope `key_fact` is asserted present in its `expect_url`'s processed
  document by `tests/test_eval.py`, so a label cannot drift from the corpus unnoticed.
- **Metric — recall@5:** the graded `key_fact` appears somewhere in the top 5 chunks for the
  query. Chunk retrieval, no LLM. This isolates chunking quality from generation quality.
- **Metric — citation accuracy:** the **top-1** chunk came from the expected page. Stricter
  than recall, and the one that matters for a product whose contract is "one source link per
  answer" — a correct fact cited to the wrong page is still a wrong answer to a user.
- **Reproducibility:** one shared embedding model, cache-first, single evaluation index per
  strategy. Scores below 0.001 between runs.

## 2. Results — raw chunk retrieval, no entity filter

This isolates chunking quality. The entity filter is measured separately in section 3.

| strategy | chunks | mean len | recall@5 | citation acc. | orphan rate |
| --- | --- | --- | --- | --- | --- |
| a — fixed-width recursive | 70 | 583 | 83% | 77% | 0% |
| b — heading-aware | 68 | 598 | 73% | 80% | 0% |
| **c — fact-grouped** | **107** | **402** | **87%** | **83%** | **0%** |

By fact type, strategy c:

| fact type | n | recall@5 | citation acc. |
| --- | --- | --- | --- |
| numeric | ~14 | **100%** | 79% |
| attribute | ~11 | 73% | 91% |
| concept | 5 | 75% | 75% |
| procedural | ~4 | 100% | 100% |

**Decision: `chunk_strategy = "c"`.** It wins both aggregate metrics, and on numeric facts —
the type this corpus is mostly made of — it is perfect against 86% (a) and 71% (b). The reason
is structural rather than lucky: a scheme page is a `## Key facts` block of ~28 labelled
bullets, and strategies a and b cut that list mid-way, producing chunks that open on a bare
`- Minimum SIP investment: …` with no scheme context. Strategy c groups the labelled fields
into question-shaped groups that each stay whole and under the size cap.

## 3. The entity filter

ADR-14 exists because Groww publishes a separate page per plan and the question names a plan.
Measured on strategy c:

| | recall@5 | citation accuracy | misses |
| --- | --- | --- | --- |
| raw | 86.7% | 83.3% | 9 |
| **with entity filter** | 86.7% | **93.3%** | **6** |

Fixed: `q01`, `q03`, `q04` — all plan/scheme confusions, including `q01`'s top-1 being
`hdfc-large-cap-fund-regular-growth` for a Direct Growth question. That is the exact ADR-14
hazard, and it was the most common single error in the raw runs.

**Recall did not move, and the reason is worth stating plainly:** filtering can only remove
from a retrieved pool, never add to it. If the right chunk was not in the top 6, no filter can
rescue it. So the filter's ceiling is the raw pool size, and improving recall means a larger
`top_k` or better chunking — not a better filter. `q10` is the concrete case: after filtering
it retrieved the correct page but the graded fact was still outside the top 5.

Still failing after filtering: `q09`, `q19` (fund-manager facts, published on several pages),
`q24`, `c01`, `c03` (two ELSS pages both state the 3-year lock-in — genuine ambiguity in the
source, not a retrieval defect).

## 4. The threshold decision, reversed

`implementation.md` specified a decision table for `min_score` on the assumption that
in-scope and out-of-scope scores separate cleanly. **They do not.** Measured on strategy c:

```
in-scope     n=30   min 0.603   median 0.834   max 0.912
out-of-scope n= 6   min 0.116   median 0.212   max 0.641
```

The strongest out-of-scope probe (0.641) scores **above** the weakest in-scope question
(0.603). The distributions overlap, so no threshold separates them. Fitting a number to six
probes would have produced a confident-looking value with no evidential support behind it.

**Decision: `min_score` stays 0.25**, which passes all 30 in-scope questions with wide margin
(median 0.834) and drops the four weakest out-of-scope probes. Its actual job is removing
near-zero noise, not policing relevance.

Out-of-scope rejection is therefore done by mechanisms that do not depend on a similarity
margin that does not exist:
- the **entity filter** — a question naming no known scheme retrieves only what it can name;
- the **intent guards** — `advice`, `returns` and `comparative` refuse before retrieval.

`recommend_min_score()` in `mf_rag/evalkit.py` now detects the overlap case and says so rather
than emitting a number.

## 5. Two findings that contradict the plan

**Orphan rate has no discriminating power.** The plan expected strategy A to score high here,
predicting a measurable difference between strategies. All three measured 0.0%. The metric is
still valid — it proves no chunk lost its `{scheme} — {section}` prefix — but it cannot
separate the strategies, because the prefix is applied in the shared `_make_chunk` helper
rather than by any one strategy. It is a regression guard, not a selection criterion.

**The eval set is 36 rows, not 20.** The plan called for 4 questions per scheme and said to
"span the taxonomy", which are in tension: concept questions cannot be attributed to a scheme,
so satisfying the taxonomy meant exceeding the per-scheme count. Six out-of-scope probes were
added for the threshold work in section 4.

## 6. What would change these numbers

- Larger `top_k` for `q09`/`q19`/`q24`, trading latency for recall. Not done: `top_k=6`
  answers the questions users actually ask, and the misses are on fund-manager facts that
  several pages publish.
- Deduplicating fund-manager fields at ingestion, which would remove the `q09`/`q19` class
  entirely. The right fix, and the one to do before this grows past five schemes.
- Nothing can fix `c01`/`c03` from the retrieval side: two ELSS pages state the same lock-in,
  so the corpus is ambiguous and the honest answer cites one and says so.
