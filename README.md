# MF Facts Assistant — HDFC Mutual Funds on Groww

A facts-only retrieval-augmented chatbot over a fixed corpus of **15 public HDFC Mutual Fund
pages on Groww**. It answers factual questions about scheme data with one source link and a
date. It does not give investment advice, does not compute or compare returns, and does not
speculate.

> Facts only. No investment advice. Answers come from public pages and may be outdated.
> Verify every number on the linked source page. Mutual fund investments are subject to
> market risks.

---

## Quick start

```powershell
# 1 · venv (Python 3.12)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2 · API key (optional — see "No key? still works" below)
.\.venv\Scripts\python.exe tools\set_groq_key.py

# 3 · Build the index and launch the UI (~80 s first run, then cached)
.\.venv\Scripts\python.exe -m mf_rag.cli all
.\.venv\Scripts\python.exe -m mf_rag.cli app
```

Verify the install at any time:

```powershell
powershell -ExecutionPolicy Bypass -File tools\check_key.ps1   # which .env is read, is the key in it
.\.venv\Scripts\python.exe -m pytest -q                        # 268 tests
```

> Use `.\.venv\Scripts\python.exe` explicitly. A bare `python` on this machine resolves to a
> 0-byte Microsoft Store stub.

### CLI

| Command | What it does |
| --- | --- |
| `cli ingest` | Fetch the 15 pages into `data/raw/`, write `data/sources.csv` |
| `cli chunk` | Build chunks under the configured strategy (`c`, fact-grouped) |
| `cli embed` | Embed chunks into the on-disk Chroma collection |
| `cli query "…"` | One question, with the retrieved chunks and filter reasons printed |
| `cli eval` | Gate 2 harness — chunking scorecard, entity filter, score distribution |
| `cli app` | Launch the Streamlit UI |
| `cli all` | ingest → chunk → embed in one go |

---

## No key? It still works.

This is a deliberate design property, not a fallback bolted on afterwards (ADR-2, principle
P8). With no key — or a key without access to the configured model, or a network failure —
the pipeline runs identically up to the last step and answers from a **deterministic
extractive path**: it quotes verbatim lines from the retrieved chunk that actually mentions
the subject of the question.

That path cannot hallucinate, because it never generates prose. It always carries a real
citation, because the line it quotes came from a page we fetched. Every answer reports which
path produced it in `notes` (`llm`, `extractive`, `refusal`, `not_in_sources`).

If the key is missing at startup, `config.py` logs a warning naming the exact file and line
to edit. A silent downgrade to the extractive path would look like a working demo, and
"answers are just less fluent" is not a symptom anyone reports.

### Choosing a model

`MF_RAG_LLM_MODEL` must be a model **your key can actually reach**. Groq's catalogue varies
by account, so a model that works for one key returns `404 model_not_found` for another. To
list what your key can see:

```powershell
.\.venv\Scripts\python.exe -c "import json,urllib.request as u; from mf_rag.config import settings; r=u.Request('https://api.groq.com/openai/v1/models',headers={'Authorization':'Bearer '+settings.groq_api_key,'User-Agent':'curl/8.5.0'}); print('\n'.join(sorted(m['id'] for m in json.load(u.urlopen(r))['data'])))"
```

Configured default: **`openai/gpt-oss-20b`**. Verified working with the key this repo was
built against.

20b rather than 120b because the free tier allows **8000 tokens per minute** and this is a
live demo. Measured on the same five questions:

| model | latency | outcome |
|---|---|---|
| `openai/gpt-oss-20b` | 1.4–2.0 s | 5/5 correct |
| `openai/gpt-oss-120b` | 8–16 s | 5/5 correct, but trips the 8000 TPM limit after ~4 questions and silently downgrades to the extractive path |

The guards and the entity filter do the safety work; the model only writes prose, so the
larger model buys latency and nothing else. Set `MF_RAG_LLM_MODEL=openai/gpt-oss-120b` in `.env`
if you would rather have it and have rate-limit headroom.

Also verified to authenticate but **404** on that account: `llama-3.3-70b-versatile` — the
value the project originally shipped with, which is why the warning in `config.py` reports a
model problem rather than a key problem.

---

## Scope

| | |
| --- | --- |
| AMC | HDFC Mutual Fund (one) |
| Schemes | HDFC Large Cap · HDFC Equity (Flexi Cap) · HDFC ELSS Tax Saver · HDFC Small Cap · HDFC Balanced Advantage — all Direct Growth |
| Pages | 15 (5 primary scheme pages, 3 plan variants, 7 context) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, offline ONNX — model committed under `models/` |
| Vector store | ChromaDB, persistent on disk |

The 5 primary pages are the deliverable's source of record. The other 10 exist so the system
can be *tested* on hard cases: sibling plans of the same fund (where the pages differ by one
token), and concept pages that must never supply a scheme number.

`data/sources.csv` carries a `page_role` column — `primary`, `variant`, `amc`, `category`,
`tool`, `regulatory`, `education` — which is what makes "the 5 primary URLs" a CSV filter
rather than a hand-maintained list.

---

## Architecture

Seven stages. Each is one module, and the order is the safety property.

```
1  loaders.py    fetch 15 pages      → data/raw, data/sources.csv
2  chunkers.py   fact-grouped split → data/chunks
3  embedder.py   MiniLM, local      → 384-dim vectors
   store.py      Chroma, on disk    + index fingerprint
4  guards.py     PII / advice / returns / comparative / speculative  ← no LLM, no network
5  retriever.py  entity filter → cosine → MMR → threshold
   memory.py     10-turn window, rewrites follow-up queries
6  answerer.py   LLM → 7 post-checks → extractive fallback
7  app.py        Streamlit, shows the retrieved chunks and why each was kept
```

Three invariants the ordering buys:

- **Guards run before anything touches the network or the corpus.** A refusal cannot be
  produced *after* a model has already seen the question.
- **Retrieval resolves identity deterministically; only prose is generated.** The entity
  filter is a filter, not a bonus — measured on the real index, "expense ratio of HDFC Large
  Cap Fund" scores the Regular Growth page 0.9009 and the Direct Growth page 0.8985. A
  0.0024 gap is noise, and HNSW will happily return the wrong one first. Two pages differing
  by one token in a 250-character chunk cannot be separated by any embedding model.
- **Every answer degrades to something that cannot lie.** Seven post-checks, then verbatim
  extraction. There is no configuration in which the system answers without a citation.

---

## The two decisions that were measured, not chosen

### Chunking: strategy `c` (fact-grouped)

8 of the 15 pages are scheme pages whose entire body is a `## Key facts` list of ~28
labelled bullets (~1,100 chars, ~1.6× the chunk size), so any fixed-width splitter cuts that
list mid-way. Gate 2 measured all three strategies on the same 30-question labelled set:

| strategy | chunks | mean len | recall@5 | citation acc. | orphan rate |
| --- | --- | --- | --- | --- | --- |
| a — pure recursive | 70 | 583 | 83.3% | 76.7% | 0% |
| b — heading-aware | 68 | 598 | 73.3% | 80.0% | 0% |
| **c — fact-grouped** | **107** | **402** | **86.7%** | **83.3%** | **0%** |

`c` wins on both axes. Its numeric recall is 100% against 86% and 71% — grouping keeps a
field's label and its value in the same chunk, which is exactly what a factual lookup needs.
Orphan rate is 0% for all three because the scheme/section prefix is applied in shared
`_make_chunk`, so that metric is a regression guard rather than a discriminator.

### `min_score = 0.25`, and why the decision table in `implementation.md` is wrong

`implementation.md` assumed a gap between in-scope and out-of-scope scores. **There is no
gap.** Measured:

```
in-scope     n=30   min 0.603   median 0.834   max 0.912
out-of-scope  n=6    min 0.116   median 0.212   max 0.641
```

The strongest out-of-scope probe (0.641) scores *above* the weakest in-scope question
(0.603). No threshold separates them. `recommend_min_score()` detects the overlap and says so
rather than recommending a cut that would fail. Out-of-scope rejection is done by the entity
filter and the intent guards, not by a similarity cut. `min_score` stays at 0.25 because it
passes all 30 in-scope questions.

### Entity filter: 83.3% → 96.7% citation accuracy

On strategy `c`, the filter fixes 4 of 9 raw misses (q01, q03, q04, q10) and cannot improve
*recall* — filtering only ever removes from a pool, it never adds. The remaining 5 misses are
honest: q09/q19/q24 are fund-manager facts that sit in a "Scheme identity" group MiniLM ranks
around 13th, and c01/c03 are genuinely ambiguous because two ELSS pages both state the
3-year lock-in.

---

## Known limits

Stated plainly, because a demo that hides these is not demonstrating anything.

1. **The corpus is a snapshot of 15 pages.** It is not the internet and not HDFC's official
   site. If Groww republishes a page, the answer changes and this index will not know.
2. **Five misses out of 30 on the labelled set** (see above), and end-to-end accuracy is
   lower than retrieval recall because a correct fact can still be phrased so a post-check
   rejects it. Both numbers are in `docs/eval_report.md`; neither is rounded up.
3. **The extractive path answers in fragments.** `- Fund manager: Chirag Setalvad` is correct
   and cited, but it is not a sentence. That is the honest cost of a path that cannot
   generate prose, and it is why the LLM path exists.
4. **MMR is a soft diversifier, not a de-duplicator.** At `lambda=0.7` a near-duplicate with
   a slightly higher score legitimately wins. Tested at `lambda=0.1` to confirm the
   diversity term is live rather than decorative.
5. **Guard coverage is pattern-based.** Every guard kind is a tuple of regexes, so it is
   exact on the phrasings it was written for and approximate everywhere else. A refusal is
   not a proof.
6. **Conversational memory is in-memory only** — `st.session_state`, dropped when the tab
   closes, holding `(question, slug, mode)` and never answer text. PRD §3 excludes saved chat
   history; this is not that. It is also a scope addition: no design document defines a
   memory or context window.
7. **`data/raw/` is gitignored**, so context-page prose may differ on another machine. Every
   graded fact comes from a primary scheme page and is unaffected.

---

## What it refuses, and with what

| Kind | Example | Response |
| --- | --- | --- |
| PII supplied | `my PAN is ABCDE1234F` | privacy notice; the identifier is never logged or stored |
| PII elicited | `What is my PAN number for KYC?` | privacy notice |
| Advice | `Should I buy HDFC Small Cap Fund?` | `I can help you with the factual details.` |
| Returns | `Which fund gave the best returns?` | `I can help you with the factual details.` |
| Comparative | `Compare HDFC Large Cap and HDFC Small Cap` | `I can help you with the factual details.` |
| Speculative | `What if the market crashes next year?` | `I can help you with the factual details.` |

One refusal string for all four intent kinds, with **no trailing link**. An earlier version
appended a factsheet pointer; that was removed, because directing the user elsewhere is
itself guidance. The privacy notice keeps its own wording because it makes a factual claim
about the system rather than about the market.

The speculative guard is the delicate one: the corpus is full of forward-looking *facts* —
a 3-year lock-in, a benchmark name, "exit load applies if redeemed within 1 year" — so its
patterns are anchored only on hypothetical frames ("what if", "suppose", "imagine"). A
pattern broad enough to catch "what if the market crashes" would also catch "what is the exit
load", and the published conditional is answerable.

---

## Layout

```
mf_rag/        config, sources, loaders, scheme_facts, chunkers, embedder, store,
               guards, retriever, memory, prompts, answerer, evalkit, cli
app.py         Streamlit UI
eval/          chunking_eval.json — 36 labelled rows (30 in-scope, 6 out-of-scope)
data/          sources.csv, raw/, processed/, chunks/
docs/          eval_report, sample_qa, sources, disclaimer, acceptance_pass, demo_script
tools/         acceptance, make_sample_qa, make_sources, make_disclaimer, set_groq_key, check_key
tests/         268 tests
```

`docs/` is **generated from live constants** by `tools/make_*.py`, so it cannot drift from
the code it describes.
