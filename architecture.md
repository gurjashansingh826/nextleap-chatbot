# Architecture — MF Facts RAG Chatbot (HDFC AMC · Class Demo)

| Field | Value |
| --- | --- |
| Document | Software Architecture Document (v1.0) |
| Derived from | [`PRD.md`](./PRD.md) v1.0 |
| Status | Draft for review |
| Last updated | 2026-09-28 |
| Audience | Demo team + reviewers assessing the RAG pipeline design |
| Companion docs | `docs/eval_report.md` (chunking decision), `docs/sample_qa.md`, `README.md` |

---

## Table of Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Design principles](#2-design-principles)
3. [System context](#3-system-context)
4. [Component map](#4-component-map)
5. [Pipeline stages in detail](#5-pipeline-stages-in-detail)
6. [End-to-end query sequence](#6-end-to-end-query-sequence)
7. [Data model and contracts](#7-data-model-and-contracts)
8. [Module interfaces](#8-module-interfaces)
9. [Retrieval design](#9-retrieval-design)
10. [Guard layer](#10-guard-layer)
11. [Prompting design](#11-prompting-design)
12. [Answer post-check pipeline](#12-answer-post-check-pipeline)
13. [Configuration reference](#13-configuration-reference)
14. [Failure modes and fallbacks](#14-failure-modes-and-fallbacks)
15. [Cross-cutting concerns](#15-cross-cutting-concerns)
16. [Testing and evaluation architecture](#16-testing-and-evaluation-architecture)
17. [Extension points](#17-extension-points)
18. [Architecture decision records](#18-architecture-decision-records)
19. [Traceability matrix](#19-traceability-matrix)

---

## 1. Purpose and Scope

The PRD states *what* the system must do. This document states *how* it is built: the
components, the boundaries between them, the data contracts that cross those boundaries, and
the design decisions behind them.

**Architectural goal for this build:** make every stage of the RAG pipeline **visible,
logged, and individually inspectable**. The demo is graded on the pipeline being explainable
end to end, not on production-grade scale. That single goal drives most choices below — for
example, we use plain Python rather than a LangChain agent precisely so the data path from
raw HTML to answer is legible in a 3-minute walkthrough.

**In scope:** all 7 pipeline stages, 15-URL corpus, 5 schemes, CLI, Streamlit UI, guards,
eval harness.

**Out of scope (unchanged from PRD §3):** advice, return computation, live NAV, auth, saved
history, multi-AMC in v1, production deployment, PII collection.

---

## 2. Design Principles

Each principle traces to a hard constraint in PRD §10. Where a principle conflicts with
convenience, the constraint wins.

| # | Principle | Consequence in the design | Traces to |
| --- | --- | --- | --- |
| P1 | **Provenance is non-negotiable** | No chunk exists without a `source_url`; a URL absent from `sources.csv` invalidates the answer | §10.1, FR-6, FR-16 |
| P2 | **Refuse before you generate** | Guards run *before* retrieval and before the LLM; advice/returns/PII never reach the model context | §10.5, FR-9, FR-10, FR-12 |
| P3 | **Facts only, never claims** | Education pages may explain concepts but may never supply a scheme number; `page_role` is in the prompt | §5.3, §10.3 |
| P4 | **Say you don't know** | A similarity floor produces an explicit "not in my sources" path instead of a confident guess | FR-11, §10.7 |
| P5 | **Inspectable by humans** | Chunks are mirrored to a readable `.txt`; `query` prints chunks + scores; the UI exposes matched chunks | §7.3, FR-4, FR-14 |
| P6 | **One embedding model, two callers** | A single `embed()` function is the only path to the model, so chunks and queries cannot drift apart | Brief requirement, NFR-5 |
| P7 | **Deterministic where it counts** | Temperature 0, fixed seed, no sampling in retrieval; refusal paths are fixed strings | NFR-5 |
| P8 | **Degrade, don't die** | Extractive fallback, cached HTML, prebuilt `sample_qa.md` — a missing key or a network outage must not end the demo | §14, Open Q1 |
| P9 | **No PII ever touches disk** | Scrubbing happens *before* logging, not after | §10.2, FR-12 |
| P10 | **Ingest once, query many** | Chroma persists; re-running `embed` is idempotent by `chunk_id` | FR-3, §14 |

---

## 3. System Context

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            BUILD TIME (once)                                 │
│                                                                              │
│   ┌────────────┐   HTTPS    ┌──────────────────────────────────────┐          │
│   │ 15 public  │───────────▶│  STAGE 1  loaders.py                 │          │
│   │  URLs      │◀─ 200 OK  │  httpx + bs4 (+ trafilatura)         │          │
│   │ (Groww)    │            │  rate-limited, retried, cached      │          │
│   └────────────┘            └───────────────┬──────────────────────┘          │
│                                            ▼                                 │
│                          data/raw/*.html → data/processed/*.md               │
│                          data/sources.csv                                    │
│                                            ▼                                 │
│                            ┌──────────────────────┐                          │
│                            │ STAGE 2  chunkers.py │  Candidate A | B        │
│                            └──────────┬───────────┘                          │
│                                       ▼                                      │
│                       data/chunks/chunks.jsonl + chunks.txt                  │
│                                       ▼                                      │
│                            ┌──────────────────────┐                          │
│                            │ STAGE 3  embedder.py │  all-MiniLM-L6-v2         │
│                            └──────────┬───────────┘  384-dim, CPU            │
│                                       ▼                                      │
│                            ┌──────────────────────┐                          │
│                            │ STAGE 4  store.py    │  ChromaDB (persistent)    │
│                            └──────────────────────┘  chroma/                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ═══════════════════════╪═══════════════════════
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            QUERY TIME (per question)                         │
│                                                                              │
│   ┌───────────┐                                                                │
│   │  User     │  Streamlit UI (app.py) or CLI                                │
│   └─────┬─────┘                                                                │
│         │  question text                                                      │
│         ▼                                                                      │
│   ┌──────────────────┐   REFUSE (no retrieval, no LLM)                       │
│   │ STAGE 5a  guards │──────────────▶ privacy / advice / returns copy        │
│   └────────┬─────────┘                                                                │
│            ▼  factual query                                                      │
│   ┌──────────────────┐                                                            │
│   │ STAGE 5b retriev │  embed(query) → cosine top-6 → MMR → threshold         │
│   └────────┬─────────┘                                                            │
│            ▼  numbered context blocks                                           │
│   ┌──────────────────┐        ┌──────────────────────┐                        │
│   │ STAGE 6  prompts │───────▶│ Groq LLM (temp 0)    │                        │
│   │         answerer │◀───────│ or extractive fallback│                        │
│   └────────┬─────────┘        └──────────────────────┘                        │
│            ▼                                                                     │
│   post-check: citation ∈ sources.csv · ≤3 sentences · no advice verbs ·        │
│               no return patterns · "Last updated from sources:" footer          │
│            ▼                                                                     │
│   answer + clickable link + expandable matched chunks → user                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Trust boundaries.** Two places where untrusted data enters the system:

1. **Fetched HTML** → `loaders.py`. Untrusted; only extracted text is admitted, and
   boilerplate is stripped before chunking.
2. **User question** → `guards.py`. Untrusted; PII-scrubbed *before* logging, pattern-checked
   before retrieval.

No other input path exists — no auth, no user accounts, no third-party integrations.

---

## 4. Component Map

Modules map 1:1 onto pipeline stages so a reviewer can trace stage → file → log line.

| Module | Stage | Responsibility | Key dependency |
| --- | --- | --- | --- |
| `mf_rag/config.py` | — | All paths, model names, thresholds. Single source of tunables | `pydantic-settings`, `python-dotenv` |
| `mf_rag/sources.py` | 0 | The 15-URL registry; writes/reads `data/sources.csv` | `csv` |
| `mf_rag/loaders.py` | 1 | Fetch → cache raw HTML → extract main text → clean → write markdown + `sources.csv` | `httpx`, `beautifulsoup4`, `trafilatura` |
| `mf_rag/chunkers.py` | 2 | Candidate A (recursive char) and Candidate B (structure-aware) chunkers | `langchain-text-splitters` |
| `mf_rag/embedder.py` | 3 | The **only** embedding entry point. Lazy singleton model load | `sentence-transformers` |
| `mf_rag/store.py` | 4 | ChromaDB persistent client; idempotent upsert by `chunk_id`; similarity + MMR search | `chromadb` |
| `mf_rag/guards.py` | 5a | PII scrub, advice/returns intent, refusal copy, out-of-scope messages | `re` |
| `mf_rag/retriever.py` | 5b | Query embedding, top-k, MMR re-rank, threshold, prompt-context assembly | `store`, `embedder` |
| `mf_rag/prompts.py` | 6 | System prompt, citation wording, few-shot examples, extractive template | — |
| `mf_rag/answerer.py` | 6 | Groq call, sentence cap, post-checks, extractive fallback | `groq` |
| `mf_rag/evalkit.py` | 2/5 | Recall@5, citation accuracy, orphan rate; A-vs-B scorecard | `retriever`, `store` |
| `mf_rag/cli.py` | all | `ingest`/`chunk`/`embed`/`query`/`eval`/`app`/`all`; stage logging | `typer` |
| `app.py` | 7 | Streamlit chat UI, citation rendering, chunk disclosure, disclaimer | `streamlit` |

**Dependency rule:** arrows point one way. `guards` and `retriever` never import `answerer`;
`answerer` never imports `store`. This keeps the guard layer provably *upstream* of
generation (P2), which a reviewer can verify by reading import lines alone.

```
cli.py ──> loaders ──> sources
   │         chunkers
   ├──> guards            (no downstream imports — leaf)
   ├──> retriever ──> embedder
   │              └──> store ──> embedder
   ├──> answerer ──> prompts        (no retriever/store imports)
   ├──> evalkit ──> retriever
   └──> app.py
```

---

## 5. Pipeline Stages in Detail

### STAGE 1 — Loading (`loaders.py`)

**Responsibility.** Turn 15 URLs into 15 clean markdown files plus `sources.csv`, without
losing a scheme fact and without importing a single navigation crumb.

**Algorithm.**

```
for url in SOURCES (15):
    if data/raw/<slug>.html exists and not --refresh:
        reuse cached HTML                      # NFR-6: reproducible offline
    else:
        GET with httpx (timeout 30s, 3 retries, exponential backoff 1s/2s/4s)
        on failure: log, record http_status, CONTINUE  # FR-1: never fatal
        write data/raw/<slug>.html
    extract main content:
        1. trafilatura.extract(html, include_tables=True)   # handles nav/footer/ads
        2. fallback → bs4: drop nav, header, footer, aside, script, style
    clean:
        collapse whitespace, strip zero-width chars, normalise unicode dashes
        drop boilerplate patterns (§5.1)
    write data/processed/<slug>.md with front-matter
write data/sources.csv
log: "STAGE 1 · pages fetched 15/15 · processed 14 · failed 1"
```

**Boilerplate patterns stripped** (P5 — keeps them out of embeddings entirely):
`Groww App Download`, `Download the Groww app`, `Trusted by`, `Referral`, `cookie`,
`Disclaimer:.*mutual fund investments are subject to market risks`,
`Privacy Policy`, `Terms of Use`, `Track your investment`, app-store badges.

**Design notes.**
- *Cache-first, not fetch-always* — re-running `ingest` during the demo must be instant and
  must not hammer Groww. `--refresh` forces a re-fetch.
- *`include_tables=True`* — Groww renders expense ratio and exit load in tables, which is
  precisely the content a naive extractor throws away. This flag is load-bearing.
- *Failures are data, not exceptions.* A dropped page is recorded in `sources.csv` with its
  `http_status` and surfaced in the README, per PRD §5.3. We never back-fill it from memory.

**Failure modes.** Total network failure → `ingest` exits non-zero with a clear message if
*zero* pages succeeded, else continues with what it has. Extraction returning < 500 chars →
flag the page as `low_text` in `sources.csv` and warn; it will likely fail the Q1/Q5
acceptance criteria and the team should know during M1, not M6.

---

### STAGE 2 — Chunking (`chunkers.py`)

**Responsibility.** Produce chunks that are *self-describing* (scheme + section prefix) and
*numerically intact* (never orphan a percentage from its label).

**Two candidates, measured — not guessed** (PRD §7). Selection happens at runtime via
`--strategy`; the winner is recorded in `docs/eval_report.md`.

**Candidate A — recursive character splitting**
```
RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100,
    separators=["\n## ", "\n\n", "\n", ". ", " "],
)
```

**Candidate B — structure-aware** (provisional default, PRD §7.1)
```
1. Parse headings + FAQ question boundaries from the processed markdown
   (Groww scheme pages: Overview, Fees and charges, Exit load, Riskometer,
    Benchmark, Downloads, FAQ, About)
2. For each section:
     - if len <= 700: emit as one chunk
     - else: recurse into Candidate A within that section
3. Prefix every chunk:  "<scheme> — <section>\n<text>"
4. Enforce numeric-block protection: rejoin any chunk that begins with a
   bare number/% to the previous chunk (regex: ^\d|\d+(\.\d+)?\s?%)
5. Post-filters:
     - drop chunks < 80 chars          (nav crumbs, disclaimers)
     - keep chunks < 200 chars only if they are standalone FAQ answers
     - dedupe by sha256(normalized text), unioning the source_url list
```

**Metadata attached to every chunk** (P1, P5): `chunk_id`, `source_url`, `title`, `scheme`,
`category`, `page_role`, `section`, `char_len`, `fetched_at`, `text`.

**Outputs.** `data/chunks/chunks.jsonl` (machine) and `data/chunks/chunks.txt` (human,
D8). The `.txt` mirror is not a debug leftover — it is the artefact a reviewer reads to
confirm the chunking is sane before trusting retrieval quality.

**Design notes.**
- *The scheme prefix is a feature, not decoration.* It raises lexical overlap with queries
  that name the scheme, which measurably improves Recall@5 on a 5-scheme corpus where four
  pages share identical fee-table boilerplate.
- *Dedupe unions URLs, it does not discard them.* Direct-plan boilerplate repeats on every
  page; keeping one copy with all source URLs preserves citation options.

---

### STAGE 3 — Embedding (`embedder.py`)

**Responsibility.** Be the single, shared path from text → 384-dim vector.

```python
_MODEL: SentenceTransformer | None = None   # lazy singleton

def get_model() -> SentenceTransformer:      # loaded once per process
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(settings.EMBED_MODEL)  # all-MiniLM-L6-v2
    return _MODEL

def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(
        texts,
        normalize_embeddings=True,   # unit vectors → cosine == inner product
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
```

**Design notes.**
- *`normalize_embeddings=True`* makes Chroma's cosine distance a plain dot product and keeps
  thresholds interpretable — a similarity score of 0.3 means the same thing in the log as it
  does in the prompt.
- *Lazy singleton* — the model is ~80 MB and takes ~2 s to load. Loading lazily keeps `ingest`
  and `chunk` fast and keeps warm query latency under the NFR-2 budget, since only the first
  query pays the cost.
- *One function, two callers* (P6). `chunk`/`embed` pass chunk text; `retriever` passes the
  user's question. There is no second code path that could pick a different model.

---

### STAGE 4 — Vector Store (`store.py`)

**Responsibility.** Persist vectors so ingestion happens once, and make re-running `embed`
safe.

```
PersistentClient(path="chroma/")            # survives restarts (FR-3)
collection = get_or_create("mf_facts_hdfc", metadata={"hnsw:space": "cosine"})
```

**Idempotency (P10).** Every write is `collection.upsert(ids=[chunk_id], documents=[text],
metadatas=[meta])`. `chunk_id = f"{slug}#{index:04d}"` is deterministic, so re-embedding
after a re-chunk replaces rather than duplicates. A `--rebuild` flag drops and recreates the
collection for a clean slate.

**Design notes.**
- *Cosine space is declared in collection metadata*, not assumed — changing it later would
  otherwise silently invalidate every stored distance.
- *No separate vector store abstraction.* One backing store, one interface. An abstraction
  layer here would be speculative generality for a 15-page corpus.

---

### STAGE 5 — Retrieval (`guards.py` → `retriever.py`)

Covered in detail in §9 and §10. Order of operations is a hard design property:

```
question
  → 5a guards  ── refuse ──▶ done (no retrieval, no LLM, no logging of PII)
  → 5b embed   (same model, same normalization)
  → 5c cosine top-6 from chroma
  → 5d MMR re-rank (λ=0.7) for diversity
  → 5e drop chunks < MIN_SCORE ── all dropped ──▶ "not in my sources" (P4)
  → 5f assemble numbered context blocks + metadata table
  → STAGE 6
```

---

### STAGE 6 — Augmentation + Answering (`prompts.py` → `answerer.py`)

**Responsibility.** Turn retrieved chunks into ≤ 3 grounded sentences with exactly one
citation, or refuse.

```
build_prompt(question, chunks) -> messages
  ├── system: facts-only persona, ≤3 sentences, cite exactly one source,
  │           no advice, no return figures, education pages ≠ scheme facts
  ├── user: numbered context blocks [1]..[n] each with source_url + page_role
  │        + the question
  └── rules restated close to the question (recency in prompts matters)

call Groq (temperature=0, max_tokens=250)
  ├── success → raw_answer
  └── no key / error / timeout / rate limit → extractive_fallback(chunks)

post_check(raw_answer) -> Answer | Rejection
  1. citation present and ∈ sources.csv?      no → fallback
  2. ≤ 3 sentences?                            no → truncate to 3
  3. advice verbs?                             yes → refusal
  4. return/performance patterns?              yes → refusal
  5. append "Last updated from sources: …"
```

**Extractive fallback (P8).** The top chunk's most relevant sentence(s), trimmed to three,
with its `source_url` appended. No LLM, no network, always works — this is what guarantees the
demo completes even with no `GROQ_API_KEY`.

**Design notes.**
- *Post-check is a safety net, not the primary mechanism.* The prompt should make the model
  comply; the checks exist to catch the rare failure, and each rejection degrades to a
  deterministic path rather than shipping a bad answer.
- *`temperature=0` and a fixed seed* (P7) make the demo reproducible — a re-run should produce
  the same wording, which matters when someone is watching.

---

### STAGE 7 — UI (`app.py`)

**Responsibility.** Make the pipeline *visible* (P5) in three minutes.

```
┌───────────────────────────────────────────────────────────────┐
│  MF Facts Assistant — HDFC Mutual Funds                       │
│  Facts-only. No investment advice.                           │
│  Corpus: 15 public pages · 5 schemes · snapshot 2026-09-28    │
├───────────────────────────────────────────────────────────────┤
│  Try:                                                         │
│   • Expense ratio of HDFC Large Cap – Direct Growth?         │
│   • Is HDFC ELSS Tax Saver Fund locked in?                    │
│   • Should I buy HDFC Small Cap Fund?                        │
├───────────────────────────────────────────────────────────────┤
│  [ chat transcript ]                                         │
│   ▸ answer, ≤3 sentences                                     │
│   ▸ Source: https://… (clickable)                            │
│   ▸ Last updated from sources: 2026-09-28                    │
│   ▸ ▾ Retrieved context (3 chunks, scores shown)  ← FR-14     │
├───────────────────────────────────────────────────────────────┤
│  [__________________________] [Ask]                          │
│  ⚠ Facts-only. No investment advice.  Don't enter PAN,       │
│    Aadhaar, account numbers, OTPs, email or phone.           │
└───────────────────────────────────────────────────────────────┘
```

**Design notes.**
- *Retrieved chunks are expandable, not hidden.* FR-14 exists so the grader can see that
  retrieval actually found the fee section, not just that a number appeared.
- *The PII warning is visible in the footer*, not buried in a policy page — the guard is
  user-visible, which is a P9 design choice as much as a UX one.
- *No chat persistence.* `st.session_state` only; nothing written to disk (PRD §3).

---

## 6. End-to-end Query Sequence

```
User        Streamlit/CLI      guards        retriever      chroma      LLM(Groq)     answerer
 │                │               │              │            │            │            │
 │ "exit load     │               │              │            │            │            │
 │  on flexi?"    │               │              │            │            │            │
 │───────────────▶│               │              │            │            │            │
 │                │──────────────▶│              │            │            │            │
 │                │               │ PII? no      │            │            │            │
 │                │               │ advice? no   │            │            │            │
 │                │               │ returns? no  │            │            │            │
 │                │◀──"proceed"───│              │            │            │            │
 │                │──────────────▶│              │            │            │            │
 │                │               │              │ embed(q)   │            │            │
 │                │               │              │───────────▶│            │            │
 │                │               │              │◀── top-6 ──│            │            │
 │                │               │              │ MMR        │            │            │
 │                │               │              │ threshold  │            │            │
 │                │               │              │ (0 kept) → "not in my sources"     │
 │                │               │              │───────────▶ build prompt │            │
 │                │               │              │──────────────────────────▶│            │
 │                │               │              │            │◀── answer ──│            │
 │                │               │              │            │            │───────▶   │
 │                │               │              │            │            │            │ post_check
 │                │◀────────────────────────────────────────────────────────────────────── │ +footer
 │  answer + link + chunks ◀────────│                                                            
```

**Timing budget** (NFR-2: warm query < 5 s):

| Step | Budget | Note |
| --- | --- | --- |
| Guards | < 10 ms | regex only |
| Query embedding | 20–60 ms | after model warm-up |
| Chroma search | 10–50 ms | 15 pages — negligible at this scale |
| MMR + threshold | < 5 ms | |
| Groq call | 0.8–2.5 s | dominant cost; drops to 0 with extractive fallback |
| Post-check | < 10 ms | |
| **Total (warm)** | **~1–3 s** | within budget |

---

## 7. Data Model and Contracts

### 7.1 `data/sources.csv` — the provenance table

```csv
slug,url,title,scheme,category,page_role,fetched_at,http_status,char_count
hdfc-large-cap-fund-direct-growth,https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth,HDFC Large Cap Fund – Direct Growth,HDFC Large Cap Fund,large_cap,primary,2026-09-28,200,48213
```

| Field | Type | Notes |
| --- | --- | --- |
| `slug` | str | URL-safe id; primary key across all artefacts |
| `url` | str | The only citation surface (P1) |
| `title` | str | Page title |
| `scheme` | str | e.g. `HDFC ELSS Tax Saver Fund` |
| `category` | enum | `large_cap`, `flexi_cap`, `elss`, `small_cap`, `hybrid` |
| `page_role` | enum | `primary`, `variant`, `amc`, `category`, `tool`, `regulatory`, `education` |
| `fetched_at` | date | ISO `YYYY-MM-DD`; the corpus snapshot date |
| `http_status` | int | `200` or the failure code; non-200 ⇒ excluded from corpus |
| `char_count` | int | Extracted text length; low values flag `low_text` pages |

**Invariant (P1):** every URL emitted in an answer must exist in this file with
`http_status == 200`. Enforced in `answerer.post_check`.

### 7.2 `data/processed/<slug>.md`

```markdown
---
title: "HDFC Large Cap Fund - Direct Growth"
url: "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
scheme: "HDFC Large Cap Fund"
category: "large_cap"
page_role: "primary"
fetched_at: "2026-09-28"
---

## Key facts

Values published on the scheme page, extracted as labelled fields.

- Expense ratio (TER, direct plan): 1.03%
- Exit load: Exit load of 1% if redeemed within 1 year
...

## Overview

<narrative text, performance tables removed>
```

The front matter is quoted so a title containing a colon cannot corrupt the YAML block. The
`## Key facts` heading is what lets STAGE 2 treat the fact list as one structural unit (§17.1).

Committed to git (NFR-6) so the demo is reproducible without re-scraping.

> **Reproducibility caveat, measured 2026-09-28.** Groww serves a stable document on repeated
> fetches (two consecutive fetches are byte-identical), and all seven graded facts on all eight
> scheme pages are byte-identical across independent live fetches — the factual content is
> fully reproducible. The *narrative* of the seven context pages can shift between days (one
> page gained 78 performance tables between two fetches, which the stripper removed). Since
> `data/raw/` is gitignored, regenerating the context-page prose on another machine may
> therefore differ slightly from the committed copy. This is the accepted cost of not
> committing ~5 MB of HTML, and it affects no graded fact.

### 7.2.1 Chunking strategy — the measurement behind ADR-13

Measured on the real corpus rather than chosen by intuition:

| | Candidate A (recursive) | Candidate B (heading-aware) | **Candidate C (fact-grouped)** |
| --- | --- | --- | --- |
| chunks | 72 | 70 | **107** |
| mean length | 589 | 605 | **396** |
| max length | 1,230 | 1,234 | **747** |
| graded facts retrievable | yes | yes | **yes** |

A and B produce *fewer, larger* chunks because they split the ~1,100-char `## Key facts` list
at a character offset, mid-list. A question about minimum SIP then retrieves text beginning
`- Minimum additional investment: ...` with no scheme in it. C groups the 29 labelled fields
into seven question-shaped groups — fees, minimum investments, lock-in, risk+benchmark,
identity, NAV/size, objective — each of which stays whole, stays under `chunk_size`, and
answers one complete question class on its own.

Two non-obvious rules make this work, both found by testing rather than by design:

- **Fact groups are exempt from the prose minimum-length filter.** Applying `chunk_min_chars`
  (200) to them silently discarded *Minimum investments*, *Risk and benchmark*, *NAV and fund
  size*, *Lock-in and availability* and *Scheme objective* — i.e. the minimum SIP, the
  riskometer and the benchmark. `FACT_MIN_CHARS = 30` applies instead. Regression-guarded by
  `tests/test_corpus.py::test_gate2_fact_groups_survive_the_length_filter`.
- **Fact groups are never deduplicated across pages.** Two schemes legitimately share a
  riskometer value; merging them is not deduplication but data loss. Dedupe applies to prose
  only, which is where repeated boilerplate actually occurs.

An LLM-based semantic chunker was considered and rejected: slower, non-deterministic,
metered, and strictly unnecessary when the fields are already labelled in the source. Phase 5's
eval still measures all three and can overturn the default.

### 7.3 `data/chunks/chunks.jsonl`

```json
{
  "chunk_id": "hdfc-large-cap-fund-direct-growth#0007",
  "text": "HDFC Large Cap Fund — Fees and charges\nExpense ratio (Direct Growth): 0.63% p.a. ...",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "title": "HDFC Large Cap Fund – Direct Growth",
  "scheme": "HDFC Large Cap Fund",
  "category": "large_cap",
  "page_role": "primary",
  "section": "Fees and charges",
  "char_len": 612,
  "fetched_at": "2026-09-28"
}
```

### 7.4 Chroma metadata

Identical to the JSONL fields minus `text` (Chroma stores `documents` separately). `text` is
intentionally *not* in metadata — it would bloat the metadata payload and duplicate storage.

### 7.5 `Answer` — the output object

```python
@dataclass
class Answer:
    question: str
    text: str                       # ≤ 3 sentences
    citations: list[str]            # ⊆ sources.csv urls
    as_of: str                      # "2026-09-28"
    chunks: list[RetrievedChunk]    # evidence, for FR-14
    mode: Literal["llm", "extractive", "refusal", "not_in_sources", "pii_refusal"]
    guard: str | None               # which guard fired, if any
    disclaimer: str                 # verbatim from docs/disclaimer.md
```

`mode` is deliberately part of the contract: it makes the guard paths assertable in tests and
visible in the UI, and it proves in the demo log which path a given answer took.

---

## 8. Module Interfaces

Type hints on every public function (NFR-7); each module docstring names its stage.

```python
# ── STAGE 1 ──────────────────────────────────────────────────────────────────
def fetch_page(url: str, slug: str, refresh: bool = False) -> FetchResult: ...
def extract_main_text(html: str, url: str) -> str: ...
def clean_text(text: str) -> str: ...
def ingest_all(refresh: bool = False) -> IngestSummary: ...

# ── STAGE 2 ──────────────────────────────────────────────────────────────────
class Chunker(Protocol):
    name: str
    def split(self, doc: ProcessedDoc) -> list[Chunk]: ...

class RecursiveChunker:      # Candidate A
    def split(self, doc: ProcessedDoc) -> list[Chunk]: ...

class StructureAwareChunker: # Candidate B (default)
    def split(self, doc: ProcessedDoc) -> list[Chunk]: ...

def build_chunks(strategy: Literal["a", "b"]) -> list[Chunk]: ...

# ── STAGE 3 ──────────────────────────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]: ...

# ── STAGE 4 ──────────────────────────────────────────────────────────────────
def upsert_chunks(chunks: list[Chunk], rebuild: bool = False) -> int: ...
def search(query_vec: list[float], top_k: int) -> list[ScoredChunk]: ...
def collection_size() -> int: ...

# ── STAGE 5 ──────────────────────────────────────────────────────────────────
def check_guards(question: str) -> GuardVerdict | None: ...      # None = proceed
def retrieve(question: str, top_k: int, min_score: float) -> list[RetrievedChunk]: ...
def mmr_rerank(candidates: list[ScoredChunk], lambda_: float, top_k: int) -> list[RetrievedChunk]: ...
def build_context(chunks: list[RetrievedChunk]) -> str: ...

# ── STAGE 6 ──────────────────────────────────────────────────────────────────
def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]: ...
def answer_question(question: str, top_k: int = 6) -> Answer: ...
def post_check(text: str, chunks: list[RetrievedChunk]) -> Answer | None: ...
def extractive_fallback(chunks: list[RetrievedChunk]) -> Answer: ...

# ── STAGE 7 ──────────────────────────────────────────────────────────────────
# app.py — Streamlit; calls guards.retrieve.answer_question
```

---

## 9. Retrieval Design

### 9.1 Why hybrid (vector + keyword) is *not* used

Worth stating explicitly because it will be asked. BM25 or hybrid search would help on exact
terms like `80C` or `IDCW`. But scheme pages are small, highly lexically similar to their own
questions, and the corpus is 15 pages — pure cosine already scores near-ceiling Recall@5 in the
M2 eval, and adding a keyword index adds a component to explain. **If the M2 eval shows
Recall@5 < 0.8**, the change is to add a lightweight BM25 tiebreak, not to redesign retrieval.

### 9.2 Scoring

```
score = 1 − cosine_distance(query_vec, chunk_vec)      # unit vectors → [0, 1]-ish
```

Chroma returns L2 or cosine distance; we use cosine and convert, so `MIN_SCORE` is
interpretable and can be tuned against observed score distributions in the eval log.

### 9.3 MMR re-rank

Retrieval returns 6 raw candidates; MMR selects 4 to maximise relevance *and* diversity so a
question about exit load does not receive three near-identical fee chunks from the same page.

```
MMR = argmax_{c ∈ C \ S}  λ · sim(c, q) − (1 − λ) · max_{s ∈ S} sim(c, s)
λ = 0.7      # relevance-dominant; lower values drift toward diversity
```

### 9.4 Threshold calibration

`MIN_SCORE` starts at **0.25** and is tuned during M2 by inspecting the score distribution on
the 20-question eval set:

| Observed | Action |
| --- | --- |
| In-scope questions score > 0.5, out-of-scope < 0.15 | Keep 0.25 — clean separation |
| Overlap band; in-scope answers get dropped | Lower to 0.20 |
| Out-of-scope questions score > 0.30 | Raise to 0.35 |

The threshold is **empirically calibrated, never guessed** — and the calibration table gets
copied into `docs/eval_report.md` so the choice is auditable.

### 9.5 Scheme-aware boosting (optional, only if eval justifies it)

On a 5-scheme corpus, a query naming "HDFC ELSS Tax Saver" should be biased toward that
scheme's chunks. If the M2 eval shows cross-scheme confusion (a flexi-cap question retrieving
large-cap chunks), a small similarity bonus is added for chunks whose `scheme` matches a scheme
named in the query. **Off by default** — added only with eval evidence.

---

## 10. Guard Layer

The guard layer is the component that makes the compliance story true rather than aspirational.
It runs in a fixed order and is the only thing allowed to short-circuit the pipeline.

### 10.1 Order of operations

```
1. PII scrub      ── match?  → privacy notice; DO NOT LOG; return
2. PII redaction  ── (for borderline matches) redact before any downstream use
3. advice intent  ── match?  → refusal + educational link; return
4. returns intent ── match?  → refusal + factsheet link; return
5. proceed to retrieval
```

PII is checked **first** so a message like *"my PAN is ABCDE1234F, should I buy?"* triggers
the privacy notice rather than the advice refusal — and, critically, so the PAN is never
written to the JSONL eval log (P9).

### 10.2 PII patterns

| Pattern | Regex (illustrative) |
| --- | --- |
| PAN | `\b[A-Z]{5}\d{4}[A-Z]\b` |
| Aadhaar | `\b\d{4}\s?\d{4}\s?\d{4}\b` (with optional X) |
| Email | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` |
| Phone | `\b(?:\+91[\-\s]?)?[6-9]\d{9}\b` |
| OTP | `\b\d{6}\b` when adjacent to `otp\|code\|verify` |
| Account no. | `\b\d{8,18}\b` when adjacent to `account\|a/c\|acc` |

**Long digit runs get a lower-confidence check** than PAN/email. A bare `12345678` is more
likely a scheme code fragment than a real account number, and over-eager matching would make
the bot look broken during a demo about mutual funds.

### 10.3 Intent patterns

| Guard | Triggers (non-exhaustive) | Response mode |
| --- | --- | --- |
| advice | should I, is it good for me, recommend, best fund for, safe, allocate, portfolio, suitable, worth buying, which should I | `refusal` + AMC page link |
| returns | returns, performance, CAGR, best performing, % gain, how much did it give, perform better | `refusal` + factsheet link |
| comparative | better than, vs, versus, compare, ranking, top 10 | `refusal` + factsheet link |

`comparative` is folded in because *"is HDFC Flexi Cap better than HDFC Large Cap?"* is a
returns question wearing a comparison costume.

### 10.4 Refusal copy (verbatim, PRD §10.2)

> I can share published facts about HDFC Mutual Fund schemes — fees, exit load, minimum SIP,
> lock-in, riskometer, benchmark and how to get documents. I can't give investment advice or
> compare returns. For that, please read the official factsheet:
> `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`

Privacy notice (verbatim):

> Please don't share personal details. This assistant doesn't accept or store PAN, Aadhaar
> numbers, account numbers, OTPs, email addresses, or phone numbers. Your question was not
> saved. For account-specific queries, use the Groww app or your registered adviser.

### 10.5 Non-goal (deliberate)

There is **no LLM-based classifier** for intent. A keyword/pattern guard is deterministic,
auditable, testable, and instant; a classifier would add latency, cost, and a new failure mode
for a rule set of about twenty patterns. Guards are verified by unit tests; there is no
"the classifier was feeling lenient today" risk.

---

## 11. Prompting Design

### 11.1 System prompt (sketch)

```
You are a facts-only assistant for HDFC Mutual Fund scheme pages on Groww.

Rules:
1. Answer ONLY from the numbered context blocks below. If the answer is not in
   them, say: "I don't have that in my sources." Do not use outside knowledge.
2. Be factual and concise: maximum 3 sentences. No preamble, no summary,
   no "Great question".
3. End with exactly one source URL taken from the context block you used.
4. Never give investment advice. Never say whether to buy, sell, hold, or which
   scheme is better. If the question asks that, refuse briefly and suggest
   reading the official factsheet.
5. Never state, calculate, or compare returns or performance figures. If asked,
   say the figures are in the official factsheet and link it.
6. Context blocks with page_role "education" explain concepts only. Never take a
   scheme number (expense ratio, exit load, minimum SIP) from them.
7. If a field is genuinely missing from the context, say it is missing and link
   the scheme page so the user can check. Never guess a number.
8. Finish with: "Last updated from sources: <date>".
```

### 11.2 User message shape

```
Context blocks:
[1] source_url: https://groww.in/... | page_role: primary | scheme: HDFC Equity (Flexi Cap) Fund | section: Exit load
    <chunk text>

[2] ...

Question: <user question>
Answer in at most 3 sentences, then one source URL, then "Last updated from sources: <date>".
```

**Design notes.**
- *Numbered blocks with explicit metadata* let the model cite block *n*, which the post-check
  maps back to a URL — this makes "exactly one citation" mechanically enforceable instead of
  hoped-for.
- *`page_role` is in the prompt text* (rule 6) so the education-vs-primary boundary is
  enforced at generation time, not only in metadata (P3).
- *The instruction is restated after the question* as well as in the system prompt — recency
  matters, and the last thing the model reads should be the format it must follow.
- *Negative instruction* (rule 7, "say it is missing") is as important as the positive ones; it
  is what pushes the model toward P4 instead of a plausible-sounding fabrication.

### 11.3 Extractive fallback template

```
<most relevant sentence from the top chunk, trimmed to ≤3 sentences>
Source: <source_url>
Last updated from sources: <date>
```

Selected by sentence-overlap with the query. No LLM, no network, always available (P8).

---

## 12. Answer Post-Check Pipeline

| # | Check | Method | On failure |
| --- | --- | --- | --- |
| 1 | Citation present | URL regex over the answer | Extractive fallback |
| 2 | Citation is real | `url ∈ sources.csv` **and** `http_status == 200` | Extractive fallback |
| 3 | ≤ 3 sentences | Split on `[.!?]` | Truncate to 3 |
| 4 | No advice language | verb/phrase blacklist: *you should, I recommend, buy this, good for, advisable, must invest* | Refusal |
| 5 | No performance claims | `\d+(\.\d+)?\s?%` **and** a return context word in the same sentence | Refusal + factsheet link |
| 6 | Footer present | Contains `Last updated from sources:` | Append |
| 7 | Disclaimer present | `Answer.disclaimer == docs/disclaimer.md` verbatim | Append |

**Check 5 is deliberately conservative.** A bare percentage is not a performance claim —
expense ratio *is* a percentage and Q1 depends on it being allowed through. The check only
fires when a percentage co-occurs with a return-context word (*returns, CAGR, gained, yield*),
so `0.63% expense ratio` passes and `18% returns` does not.

**Ordering matters.** Checks 1→2 run first: an ungrounded citation is a trust failure and
invalidates everything else, so we fall back rather than trying to patch a bad answer.

---

## 13. Configuration Reference

All tunables live in `mf_rag/config.py`, overridable by environment variable, defaulted from
`.env.example`.

| Setting | Default | Where used | Tuned in |
| --- | --- | --- | --- |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `embedder.py` | fixed by brief |
| `EMBED_DIM` | `384` | docstring/assertion | fixed by brief |
| `CHROMA_PATH` | `chroma/` | `store.py` | — |
| `CHROMA_COLLECTION` | `mf_facts_hdfc` | `store.py` | — |
| `CHUNK_SIZE` | `700` | `chunkers.py` | M2 eval |
| `CHUNK_OVERLAP` | `100` | `chunkers.py` | M2 eval |
| `CHUNK_MIN_CHARS` | `200` (80 hard drop) | `chunkers.py` | M2 eval |
| `CHUNK_STRATEGY` | `b` | `chunkers.py` | M2 eval decides |
| `TOP_K` | `6` | `retriever.py` | M2 eval |
| `MMR_LAMBDA` | `0.7` | `retriever.py` | M2 eval |
| `MIN_SCORE` | `0.25` | `retriever.py` | **M2 calibration (§9.4)** |
| `MAX_SENTENCES` | `3` | `answerer.py` | fixed by brief |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | `answerer.py` | latency vs quality |
| `LLM_TEMPERATURE` | `0.0` | `answerer.py` | fixed (NFR-5) |
| `LLM_MAX_TOKENS` | `250` | `answerer.py` | keeps answers short |
| `GROQ_API_KEY` | — (`.env`) | `answerer.py` | never committed (NFR-4) |
| `HTTP_TIMEOUT` | `30` | `loaders.py` | — |
| `HTTP_RETRIES` | `3` | `loaders.py` | — |
| `FETCH_DELAY_SECONDS` | `1.5` | `loaders.py` | politeness |

---

## 14. Failure Modes and Fallbacks

| # | Failure | Detection | Response | Demo impact |
| --- | --- | --- | --- | --- |
| F1 | No `GROQ_API_KEY` | exception at client init | Extractive fallback + log notice | None — demo still answers |
| F2 | Groq 429 / timeout / 5xx | exception | Retry once, then extractive fallback | Slight delay |
| F3 | Page fetch fails at build | non-200 after retries | Log, record in `sources.csv`, exclude | Fewer pages; documented in README |
| F4 | ~~Client-rendered page → almost no text~~ **RESOLVED 2026-09-28** | `char_count < 500` | **Did occur, and was worse than predicted: a text-only extractor found 0 of 7 graded facts.** Root cause was not rendering but *placement* — the facts sit in `__NEXT_DATA__`. Fixed by ADR-11 structured extraction. Guarded by `test_gate1_primary_carries_every_applicable_graded_fact`. | **None.** All 5 primary schemes verified to carry all graded facts. |
| F5 | No chunks above `MIN_SCORE` | empty post-threshold | "Not in my sources" + coverage list | Correct behaviour |
| F6 | Model emitted an unknown URL | post-check 2 | Extractive fallback | None |
| F7 | Model emitted advice language | post-check 4 | Refusal | None |
| F8 | Chroma dir deleted / index missing | collection missing | Raise a clear "run `embed` first" error | Recoverable in 10 s |
| F9 | Ingestion never run | no `data/processed/` | Clear error with the exact command | Recoverable |
| F10 | API key committed to git | CI/pre-commit secret scan | Fail the build | Must be checked before M6 |
| F11 | **A corpus URL from the brief is dead** | non-200 after retries | Corrected against the server, deviation recorded | **Occurred:** the brief's `best-flexi-cap-mutual-fund` (singular) returns 404. The live page is `best-flexi-cap-mutual-funds`. Corrected in `sources.py` with an inline note. |
| F12 | **A field is absent on one scheme's page** | `extract_facts` returns fewer labels | Skipped, never defaulted; coverage table shows `-` | Observed on `hdfc-focused-large-cap` (a `variant` page, not graded): no `expense_ratio`, no `min_sip_investment`. Real data, handled honestly. |

**F4 is the real risk**, not the code — it is a data problem that a retry cannot fix, which is
why PRD §14 flags it and why the M1 milestone ends with an inspection of what was actually
extracted.

---

## 15. Cross-Cutting Concerns

**Determinism.** `temperature=0`, fixed seed, sorted iteration over sources, no sampling in
retrieval, MMR with a fixed λ. Re-running the demo produces the same answers.

**Idempotency.** Deterministic `chunk_id` + Chroma `upsert` means `all` can be re-run freely.
`--rebuild` is the explicit escape hatch.

**Logging and observability.** One structured log line per stage, mirroring P5:

```
STAGE 1 · pages requested 15 · fetched 14 · cached 1 · failed 1 · low_text 2
STAGE 2 · strategy=b · chunks 187 · dropped_short 24 · deduped 9 · mean_len 648
STAGE 3 · model=all-MiniLM-L6-v2 · dim=384 · encoded 187 in 3.1s
STAGE 4 · collection=mf_facts_hdfc · upserted 187 · total 187
STAGE 5 · q="exit load on flexi cap" · top_k=6 · mmr=4 · kept 3 · best_score=0.61
STAGE 6 · mode=llm · model=llama-3.3-70b-versatile · sentences=2 · citation=ok · 1.4s
```

**PII-safe logging.** The question is scrubbed *before* it is written to `eval/queries.jsonl`.
`guards.check_guards()` returns a verdict that includes a `redacted_question`, and only that
field is ever persisted. Grep the log for `@` and `\d{10}` as a pre-submission check.

**Configuration.** Single `config.py`; no module reads `os.environ` directly. This makes the
threshold calibration in §9.4 a config change, not a code change.

**Error philosophy.** Deterministic guard paths return typed refusals. Only genuinely
unexpected errors raise, and `cli.py` converts them to a one-line message rather than a
traceback — a stack trace during a graded demo is a worse failure than a clear error.

---

## 16. Testing and Evaluation Architecture

Three layers, cheapest first.

**L1 — Unit tests (`tests/`), no network, no models.** Guards (every PII pattern, every advice
keyword), cleaning, boilerplate stripping, chunk size/overlap invariants, the numeric-block
rejoin, post-check rules 1–7, sentence truncation. Fast, deterministic, and the layer that
protects the compliance claims.

**L2 — Retrieval eval (`evalkit.py`).** The 20-question labelled set (4 per primary scheme),
each row `{query, expect_url, key_fact}`. Metrics: **Recall@5**, **citation accuracy**,
**orphan rate**. Run for both candidates into separate collections; scorecard to
`docs/eval_report.md` (D7). Answers to Open Question #4: the `key_fact` strings are read off
the *fetched* corpus, never from memory, so expected facts match the snapshot.

**L3 — Acceptance pass.** The 12 PRD §11 criteria executed against the built index,
`docs/sample_qa.md` (D5) generated by the real build, secret scan for F10.

**Why the eval set is small and hand-labelled.** With 5 schemes and 15 pages, 20 questions
cover the taxonomy (numeric fact, yes/no fact, concept, procedural) without becoming a
benchmark. A hand-labelled set also means every expected value is a human decision a reviewer
can audit — the opposite of the hallucination failure mode the project exists to avoid.

---

## 17. Extension Points

The architecture is designed for v1 scope, with seams — not speculative features — for what
comes next.

| Extension | What changes | What does *not* change |
| --- | --- | --- |
| **Add an AMC** | Add rows to `sources.py`; `category`/`scheme` already generalise | Chunking, guards, prompts, UI, post-checks. Only the corpus grows. |
| **Swap embedder** | Change `EMBED_MODEL` in config **and rebuild the index** | Everything else — but mixing vector spaces in one collection silently breaks retrieval, so a model change forces `--rebuild`. |
| **Swap LLM** | `LLM_MODEL`; swap `groq` import for another client | Prompts, post-checks, extractive fallback, guards |
| **Add a reranker** (cross-encoder) | Insert a stage between 5c and 5d | Threshold, guards, prompt assembly |
| **Add BM25 tiebreak** | Union of vector + keyword results before MMR | Threshold calibration would need redoing (§9.1) |
| **Add HDFC factsheet PDFs** | `loaders.py` gains a PDF branch | Chunking and the rest, if text extraction keeps section headings |
| **Multi-turn context** | Pass prior turns to `build_messages` | Guards must run on *every* turn, not just the first — noted so the PII rule is not lost |

**The one thing that must not be relaxed:** a new AMC or scheme still requires a `source_url`
per chunk and an entry in `sources.csv`. Provenance is the product.

---

## 18. Architecture Decision Records

| ADR | Decision | Alternatives rejected | Rationale |
| --- | --- | --- | --- |
| ADR-1 | **Plain Python orchestration, no LangChain agent** | LangChain LCEL chains, LlamaIndex | Keeps every stage visible and loggable — the thing being graded. A framework hides the pipeline the demo is supposed to show. |
| ADR-2 | **Groq primary + deterministic extractive fallback** | Groq only; OpenAI; Ollama | Brief mandates Groq. The fallback means a missing key or a rate limit cannot end the demo. |
| ADR-3 | **Pattern guards, not an LLM classifier** | LLM intent classification | Deterministic, auditable, instant, testable. ~20 patterns do not justify a model call on the critical path. |
| ADR-4 | **Single shared `embed()`** | Separate chunk/query embedding paths | Structurally prevents model drift between corpus and query — the classic silent RAG bug. |
| ADR-5 | **Post-check with graceful degradation** | Trust the model; hard-fail on violation | Every violation degrades to a deterministic path instead of shipping a bad answer. |
| ADR-6 | **Structure-aware chunking as provisional default, decided by eval** | Hard-coding one strategy; LLM-based chunking | PRD §7 requires inspecting the data first. Pre-committed rule guarantees a documented result either way. |
| ADR-7 | **Cache-first ingestion** | Always fetch | Fast, repeatable demo builds; avoids hammering Groww; enables offline rebuilds (NFR-6). |
| ADR-8 | **ChromaDB persistent, no vector-DB abstraction** | Postgres/pgvector, Qdrant, FAISS, abstraction layer | 15 pages. An abstraction is speculative generality; Chroma is local, free, and zero-config. |
| ADR-9 | **Threshold empirically calibrated, documented** | Guessing 0.25 and shipping it | A threshold is a product decision that must be traceable to the score distribution, especially since it is the fabrication gate (P4). |
| ADR-10 | **`page_role` in prompt text and metadata** | Filtering education pages at retrieval | Lets a concept question *use* an education page while structurally preventing a scheme number from one (P3). |
| ADR-11 | **Structured scheme facts from `__NEXT_DATA__`, narrative from trafilatura** | Text extraction alone (the original §5 Stage 1 design) | Measured 2026-09-28. A text-only extractor reported **all seven** graded facts ABSENT on a scheme page, because Groww renders fees, riskometer and benchmark only inside the Next.js JSON payload, while the visible text is entirely returns and holdings tables. Reading the JSON turns risk F4 from a project-killing data problem into a solved one, and yields labelled deterministic facts instead of scraped prose. See §5.1. |
| ADR-12 | **Performance data excluded at ingestion, not at query time** | Relying on the intent guard and post-checks to catch it | Leaving `return_stats`, `sip_return`, `peerComparison` and holdings in the corpus makes performance data *retrievable*, and every guard that would later refuse to cite it is a chance to fail. Removing it at the source is auditable and cheap. |
| ADR-13 | **Fact-grouped chunking (Candidate C) as default** | Candidates A/B only (original §7) | Measured on the real corpus, see §7.1. Fixed-width splitting of a ~1,100-char labelled fact list cuts it mid-way, producing chunks that open on a bare `- Minimum SIP investment: ...` with no scheme context — unusable for retrieval and uncitable. |

---

## 17.1. Stage 1 — Structured Extraction (added after Phase 2 implementation)

The original §5 Stage 1 design assumed the visible page text carried the graded facts. It does
not. Verified on all 15 corpus pages on 2026-09-28:

| Fact | In visible text? | In `__NEXT_DATA__`? |
| --- | --- | --- |
| Expense ratio | no | `expense_ratio`, `base_expense_ratio` |
| Exit load | no | `exit_load`, `historic_exit_loads` |
| Minimum SIP / lump sum | no | `min_sip_investment`, `min_investment_amount` |
| Lock-in | no | `lock_in` |
| Riskometer | no | `nfo_risk` |
| Benchmark | no | `benchmark`, `benchmark_name` |
| NAV, AUM, ISIN, plan type | no | all present |

So STAGE 1 now runs **two extractors** and merges them:

1. **`scheme_facts.scheme_payload()`** parses `<script id="__NEXT_DATA__">` →
   `props.pageProps.mfServerSideData` and renders a `## Key facts` block of 29 labelled,
   verbatim-copied fields. Absent fields are skipped, not defaulted.
2. **`loaders.extract_main_text()`** (trafilatura, `include_tables=True`) supplies narrative
   for the seven non-scheme pages, which have no scheme payload.

**Why this is strictly better than scraping prose.** Every value in the fact block is copied
byte-for-byte from a field Groww itself publishes, so there is no OCR-ish guesswork, no
re-parsing of a rendered sentence, and nothing for a language model to paraphrase. The block
is deterministic, diffable, and reviewable against the page.

**The `## Key facts` heading is load-bearing.** STAGE 2 splits on headings, so the block
survives as a single structural unit and its groups can be chunked deliberately rather than by
character offset.

### 17.1.1. Fields deliberately *not* extracted

`expense_ratio` is a fee; these are returns, and the project's hard constraint is that this
assistant makes no performance claims (PRD §10.3):

- `sip_return`, `simple_return`, `return_stats`, `peerComparison` — performance
- `groww_rating` — a rating, not a scheme fact we can cite
- `holdings` — 50 rows of noise that would dominate the chunk count
- `historic_fund_expense`, `historic_exit_loads` — superseded values; the current figure is
  the one the page presents

The narrative extractor additionally drops markdown tables and sections whose own header marks
them as returns, rankings, or holdings (`loaders.strip_performance_tables`). This is defence in
depth: the graded fees live in the structured block, so nothing graded is lost, and the corpus
is provably free of period-bound return figures — asserted by
`tests/test_corpus.py::test_no_chunk_states_a_performance_figure`, which is itself validated
against synthetic leaks so it cannot pass vacuously.

---

## 19. Traceability Matrix

### PRD requirement → architecture

| PRD ref | Requirement | Where implemented | Verified by |
| --- | --- | --- | --- |
| FR-1 | `ingest` fetches 15 URLs, retries, non-fatal | §5 Stage 1, `loaders.py` | L1 + stage log line |
| FR-2 | `chunk` per §7 → `chunks.jsonl` + `chunks.txt` | §5 Stage 2, `chunkers.py` | `docs/eval_report.md` |
| FR-3 | Idempotent Chroma upsert | §5 Stage 4, `store.py` | Re-run `embed`, count unchanged |
| FR-4 | `query` prints chunks + scores + answer | §6 sequence, `cli.py` | Manual demo |
| FR-5 | `app` launches Streamlit | §5 Stage 7, `app.py` | Manual demo |
| FR-6 | ≥1 clickable source URL | §11 prompt, §12 check 1 | L1 post-check |
| FR-7 | `Last updated from sources:` footer | §11 rule 8, §12 check 6 | L1 post-check |
| FR-8 | ≤ 3 sentences | §12 check 3 | L1 post-check |
| FR-9 | Advice refused pre-LLM | §10.3, `guards.py` | L1 guard tests |
| FR-10 | Returns refused, factsheet linked | §10.3, §12 check 5 | L1 guard tests |
| FR-11 | Below-threshold → "not in my sources" | §9.4, §10.1 step 5 | Weather query |
| FR-12 | PII refused, never stored | §10.2, §15 PII-safe logging | L1 + log grep |
| FR-13 | Welcome + 3 examples + disclaimer | §5 Stage 7 | Screenshot in README |
| FR-14 | Source links + expandable chunks | §5 Stage 7 | Manual demo |
| FR-15 | Stage counts logged | §15 observability | Stage log output |
| FR-16 | Citation must exist in `sources.csv` | §12 check 2, ADR/P1 | L1 + invented-URL test |
| NFR-1 | CPU, build < 3 min | §6 timing, Stage 1–4 | Timed `all` run |
| NFR-2 | Warm query < 5 s | §6 timing table | Timed `query` |
| NFR-3 | Offline after first build | ADR-2 fallback, ADR-7 cache | Airplane-mode test |
| NFR-4 | Secrets in `.env`, never committed | §13, F10 | Pre-commit secret scan |
| NFR-5 | Temperature 0, fixed seed | §13 config, ADR-5 | Two identical runs |
| NFR-6 | Processed markdown committed | §7.2, ADR-7 | `git status` clean clone |
| NFR-7 | Type hints + stage docstrings | §8 | Lint/type check |
| §10.1 | Public sources only | ADR-10, §7.1 invariant | Grep emitted URLs vs `sources.csv` |
| §10.2 | No PII | §10.2, §15 | L1 + log grep |
| §10.3 | No performance claims | §12 check 5 | L1 |
| §10.4 | ≤3 sentences, 1 citation, footer | §12 | L1 |
| §10.5 | No advice | §10.3, ADR-3 | L1 guard tests |
| §10.6 | Disclaimer in header + footer | §5 Stage 7, §12 check 7 | Screenshot |
| §10.7 | No fabrication | §9.4, ADR-9 | Weather query |

### Principle → stage

| Principle | Primary enforcement point | Secondary (defence in depth) |
| --- | --- | --- |
| P1 provenance | Stage 1–2 chunk contract | Post-check 2 |
| P2 refuse before generating | Stage 5a guards | Post-check 4, import-direction rule |
| P3 facts only | `page_role` in prompt | §7.1 metadata invariant |
| P4 say you don't know | Stage 5e threshold | Prompt rule 1 + 7 |
| P5 inspectable | `chunks.txt`, `--show-chunks` | Stage 7 expandable chunks |
| P6 one embedder | `embedder.py` single function | `EMBED_DIM` assertion |
| P7 determinism | `temperature=0`, fixed seed | Sorted iteration, MMR λ fixed |
| P8 degrade don't die | Extractive fallback | Cached HTML, prebuilt `sample_qa.md` |
| P9 no PII on disk | Guards before logging | Redacted-only log field |
| P10 ingest once | Chroma persistence | Deterministic `chunk_id` upsert |

---

## Appendix — Implementation Order (suggested)

Derived from PRD §13 milestones, expressed as file-level work:

| Milestone | Files | Exit criterion |
| --- | --- | --- |
| M0 | `.env`, `requirements.txt`, venv | `python --version` and an import of `chromadb` + `sentence_transformers` succeed |
| M1 | `config.py`, `sources.py`, `loaders.py`, `cli.py` (ingest) | 15 pages in `data/processed/`, `sources.csv` written, **`char_count` reviewed for F4** |
| M2 | `chunkers.py`, `evalkit.py`, `eval/chunking_eval.json` | `docs/eval_report.md` with both scorecards and a chosen winner |
| M3 | `embedder.py`, `store.py` | Collection populated, stage 3/4 counts logged, re-run is a no-op |
| M4 | `guards.py`, `retriever.py`, `prompts.py`, `answerer.py`, `cli.py` (query) | All 10 canonical queries return correct mode; guards fire on Q9/Q10 |
| M5 | `app.py` | UI live with citations, chunk disclosure, disclaimer |
| M6 | `tests/`, `docs/*`, `README.md` | PRD §11 acceptance criteria 1–12 pass |
