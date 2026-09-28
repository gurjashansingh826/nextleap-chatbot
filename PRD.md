# PRD — MF Facts RAG Chatbot (HDFC AMC · Class Demo)

| Field | Value |
| --- | --- |
| Document | Product Requirements Document (v1.0) |
| Status | Draft for review |
| Owner | RAG demo team |
| Last updated | 2026-09-28 |
| Type | Academic / class demo prototype — **not** a production financial product |
| Source brief | `Problem Statement` (problem statement + deliverables) |
| Target stack | Python 3.11 · MiniLM-L6-v2 · ChromaDB · Groq · Streamlit |

---

## 0. Decisions made while writing this PRD

Three points in the brief were ambiguous or in conflict. They are resolved here so the
build does not stall; each is recorded in the decisions log (§16).

| # | Conflict | Resolution |
| --- | --- | --- |
| DR-1 | Brief says "collect **10** public pages" but the deliverable asks for a source list of "the **5** URLs you used" | Ship **15 pages** = **5 primary scheme pages** (the ones cited in answers) + **10 context pages** (variants, AMC, category, tool, regulatory, education). `sources.csv` marks `page_role`, so the 5-URL deliverable is a one-line filter, not a re-scrape. |
| DR-2 | Brief's tech constraints say **LLM: Groq**; an earlier draft said OpenAI/Ollama | **Groq is primary** (brief wins). A deterministic extractive fallback ships alongside it so the demo never hard-fails on a missing/rate-limited key. |
| DR-3 | All five schemes named in the brief are **Direct – Growth**; the URL list also contains Regular and IDCW variants | The 5 in-scope schemes are the **Direct – Growth** plans. The Regular/IDCW/Focused pages are ingested as **variant context only** and are never cited for a Direct-plan number. |

---

## 1. Problem Statement

Retail users and support teams ask the same factual questions about mutual fund schemes over
and over: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer level, benchmark, and
how to download statements. Today those answers are scattered across an AMC's site, Groww scheme
pages, and SEBI disclosures, so answering them means tab-hunting and reading a wall of tables.
Generic AI chatbots make it worse: they blend published facts with opinions, and they invent
figures and citations.

We will build a **facts-only RAG chatbot** that answers mutual fund scheme questions using
**only official public pages** as its knowledge base. Every answer must carry one source link.
It must refuse opinionated and portfolio questions politely and redirect to an educational page.
It must never give investment advice, never compute or compare returns, and never store
personal data.

**Why a RAG and not a fine-tune.** The facts we care about (expense ratio, exit load, lock-in)
change over time and must be traceable to a page a human can open. Retrieval gives us
provenance for free: every sentence we emit is anchored to a chunk that carries a `source_url`.
A fine-tune would give us fluent answers with no verifiable citation.

**This is a class demo.** The goal is to demonstrate the full RAG pipeline end to end —
**Loading → Chunking → Embedding → Storing vector data → Retrieval → Augmentation → Answering** —
with an architecture that is visible, logged, and explainable in three minutes.

---

## 2. Goals

| # | Goal | Measure |
| --- | --- | --- |
| G1 | Answer factual MF scheme questions from an official-only corpus | ≥ 80% of the eval set returns the correct fact with a valid citation |
| G2 | Show every RAG stage explicitly and repeatably | One command runs ingest → chunk → embed → store; each stage logs its counts |
| G3 | Guarantee a citation on every factual answer | 100% of factual answers contain ≥ 1 source URL present in `sources.csv` |
| G4 | Never produce advice or performance claims | 100% refusal on the advice/returns/PII test set |
| G5 | Be demo-able in 3 minutes | Cold start to first answer < 60 s on a warm index; warm query < 5 s |

---

## 3. Non-Goals (Out of Scope)

- No investment advice, buy/sell/hold calls, suitability opinions, or portfolio recommendations.
- No return computation, no CAGR/absolute-return figures, no fund comparison or ranking. If
  asked, link to the official factsheet instead.
- No live NAV, no real-time prices, no generated performance data.
- No login, signup, user accounts, saved chat history, or analytics of real users.
- No multi-AMC coverage in v1 (HDFC only) — but the schema must allow adding an AMC later.
- No mobile app, no production deployment, no SLA, no monitoring.
- No PII collection of any kind (PAN, Aadhaar, account number, OTP, email, phone number).

---

## 4. Users and Use Cases

**Primary — retail investor (comparison stage).**
"I want to compare two HDFC schemes on facts before I read the factsheet." Wants expense ratio,
exit load, minimum SIP, benchmark, riskometer, and a link to verify each number.

**Secondary — support / content team.**
"What's the ELSS lock-in period?" Wants the same answer, fast, in a form they can paste into a
ticket — short, with a link.

Both want a short factual answer with a source, not a paragraph of prose and not a sales pitch.

### 4.1 Canonical queries (the demo script)

These ten queries are the acceptance set. They double as the three example questions in the UI
and the rows in the sample Q&A deliverable.

| # | Intent | Query | Expected behaviour |
| --- | --- | --- | --- |
| Q1 | Fees | Expense ratio of HDFC Large Cap Fund – Direct Growth? | Fact + citation |
| Q2 | Exit load | Exit load on HDFC Equity (Flexi Cap) – Direct Growth? | Fact + citation |
| Q3 | Minimum SIP | Minimum SIP for HDFC Small Cap – Direct Growth? | Fact + citation |
| Q4 | Lock-in | Is HDFC ELSS Tax Saver Fund locked in? | Fact (3 years) + citation |
| Q5 | Riskometer | What is the riskometer level of HDFC Balanced Advantage Fund? | Fact + citation |
| Q6 | Benchmark | Benchmark of HDFC Equity (Flexi Cap) Fund? | Fact + citation |
| Q7 | Documents | How do I download my capital gains statement? | Steps + citation |
| Q8 | Tax (education) | How is ELSS taxed? | Educational answer + citation |
| Q9 | Advice (refuse) | Should I buy HDFC Small Cap Fund? | Polite refusal + educational link |
| Q10 | Returns (refuse) | Which HDFC fund has given the best returns? | Polite refusal + factsheet link |

Plus two guard tests outside the demo script: an out-of-scope query ("What is the weather in
Delhi?") must return *not in my sources*, and a query containing a PAN/email/phone must be
refused and never stored.

---

## 5. Scope — AMC, Schemes, and Corpus

**AMC:** HDFC Mutual Funds
**Platform for public pages:** Groww (public scheme pages) + Groww AMC page + Groww/SEBI
riskometer and investor-education pages.

All five in-scope schemes are **Direct – Growth** plans, which is the plan Groww's public pages
expose most completely and which avoids the regular/direct fee ambiguity in a facts demo.

### 5.1 Primary corpus — 5 scheme pages

| # | Category | Scheme | URL |
| --- | --- | --- | --- |
| S1 | Large Cap | HDFC Large Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| S2 | Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| S3 | ELSS | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| S4 | Small Cap | HDFC Small Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| S5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

Coverage intent: one large-cap, one flexi-cap, one ELSS, one small-cap, one hybrid — enough
variety to show the retriever handling different page structures, different section names, and
different question types (numeric fact, yes/no fact, concept fact, procedural fact).

### 5.2 Context corpus — 10 support pages

| # | `page_role` | Topic | URL |
| --- | --- | --- | --- |
| 6 | `variant` | Large Cap – Regular Growth (direct vs regular) | `https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth` |
| 7 | `variant` | Large Cap – Direct IDCW (plan-variant context) | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw` |
| 8 | `variant` | Focused Large Cap – Direct Growth (disambiguation) | `https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth` |
| 9 | `amc` | HDFC Mutual Funds (fund house) | `https://groww.in/mutual-funds/amc/hdfc-mutual-funds` |
| 10 | `category` | Large & Mid Cap funds | `https://groww.in/mutual-funds/category/best-large-and-midcap-mutual-funds` |
| 11 | `category` | Flexi Cap funds | `https://groww.in/mutual-funds/category/best-flexi-cap-mutual-fund` |
| 12 | `tool` | SIP calculator (SIP concepts) | `https://groww.in/calculators/sip-calculator` |
| 13 | `regulatory` | Riskometer (SEBI) | `https://groww.in/p/riskometer` |
| 14 | `education` | ELSS vs SIP | `https://groww.in/questions/what-is-the-difference-between-elss-and-sip` |
| 15 | `education` | Tax on mutual funds | `https://groww.in/blog/tax-on-mutual-funds/` |

**Total corpus: 15 public pages, 5 of them primary.**

The context pages earn their place: `variant` pages let the bot explain what "Direct" vs
"Regular" vs "IDCW" means instead of guessing; `regulatory` explains the riskometer scale so a
level question can be answered with meaning; `education` answers concept questions (ELSS vs SIP,
tax treatment) that no scheme page covers; `category` and `amc` pages give the bot a safe landing
spot for its refusals.

### 5.3 Source rules (hard constraints)

- **Public pages only.** No logged-in screens, no back-end dashboards, no screenshots of the app.
- **No third-party blogs as sources.** `groww.in/blog/...` is admitted as *investor-education*
  content only, never as the source of a scheme number (expense ratio, exit load, benchmark,
  riskometer). `page_role` encodes this and the prompt enforces it.
- **No hand-typed facts.** Every number in an answer must originate in a fetched chunk. If a
  field is absent from the fetched HTML, the bot says so and links the page.
- Any page that cannot be fetched publicly at build time is dropped from the corpus and recorded
  as dropped in the README. Nothing is back-filled from memory.

---

## 6. System Architecture

The brief's framing, made explicit:

```
< DATA INGESTION :  Loading → Chunking → Embedding  →>  Vector Space
< DATA RETRIEVAL :  Query → embedding → find chunk in vector space
                    → (Augmentation) LLM + Chunk + system prompt → Generate Answer
```

Full pipeline:

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 1 · LOADING                                                     │
 │  fetch 15 public URLs (rate-limited, retried, cached)                 │
 │  → data/raw/<slug>.html                                               │
 │  → extract main text, drop nav / footer / cookie banners / app CTAs   │
 │  → clean: collapse whitespace, strip boilerplate, fix mojibake        │
 │  → data/processed/<slug>.md  (text + front-matter metadata)          │
 └──────────────────────────┬────────────────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 2 · CHUNKING   (structure-aware, see §7)                        │
 │  heading / FAQ-boundary split first, recursive character fallback     │
 │  → every chunk prefixed with scheme + section heading                 │
 │  → data/chunks/chunks.jsonl  (human-readable mirror: chunks.txt)      │
 └──────────────────────────┬────────────────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 3 · EMBEDDING                                                    │
 │  sentence-transformers/all-MiniLM-L6-v2  (384-dim, CPU, no API key)   │
 │  same model embeds chunks and queries — enforced by a single fn       │
 └──────────────────────────┬────────────────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 4 · VECTOR STORE                                                 │
 │  ChromaDB persistent client · collection: mf_facts_hdfc               │
 │  metadata: chunk_id, source_url, title, scheme, category,             │
 │            page_role, section, char_len, fetched_at                   │
 └──────────────────────────┬────────────────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 5 · RETRIEVAL  (cheap, deterministic, pre-LLM)                  │
 │  a. PII guard          → privacy notice, never logged                  │
 │  b. intent guard       → advice / returns / suitability ⇒ refuse       │
 │  c. embed query with the same MiniLM model                             │
 │  d. cosine top-k = 6, then MMR re-rank for diversity                   │
 │  e. drop chunks below min_score ⇒ "not in my sources" path            │
 │  f. build prompt: numbered context blocks + system prompt              │
 └──────────────────────────┬────────────────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 6 · AUGMENTATION + ANSWERING  (Groq)                            │
 │  answer grounded only in retrieved blocks, ≤ 3 sentences              │
 │  output = answer + exactly one citation + "Last updated from sources:" │
 │  post-check: citation ∈ sources.csv, no advice verbs,                 │
 │              no return/performance patterns, sentence count            │
 └──────────────────────────┬────────────────────────────────────────────┘
                            ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STAGE 7 · UI  (Streamlit)                                             │
 │  welcome line + 3 example questions + "Facts-only. No advice." note   │
 │  per answer: rendered source link + expandable matched chunk evidence  │
 └───────────────────────────────────────────────────────────────────────┘
```

### 6.1 Stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.11+ | Ecosystem for ChromaDB + sentence-transformers |
| Fetch | `httpx` (+ `beautifulsoup4`, optional `trafilatura`) | Static HTML, no browser needed |
| Chunking | `langchain-text-splitters` `RecursiveCharacterTextSplitter` + custom heading/FAQ splitter | Mature, tunable, no LLM cost |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~80 MB, CPU-only, no API key (per brief) |
| Vector DB | `chromadb` persistent client | Free, local, zero-config, ingest runs once |
| Orchestration | Plain Python (no LangChain agent) | Keeps every stage visible for the demo |
| LLM | **Groq** (primary) + deterministic extractive fallback | Brief requires Groq; fallback guarantees the demo always runs |
| UI | `streamlit` | `streamlit run app.py` → shareable local link |
| Eval | Local JSONL log + small offline harness | Show retrieval quality live during the demo |

> **Environment note:** this machine currently has Node 24 and Git but **no Python**. Install
> Python 3.11+ and add it to PATH before the first `pip install`. This is the single biggest
> setup risk and belongs in the README as step 0.

### 6.2 Repository layout

```
mf-rag-chatbot/
├─ app.py                     # Streamlit UI (stage 7)
├─ pyproject.toml / requirements.txt
├─ .env                       # GROQ_API_KEY — gitignored, never committed
├─ .env.example               # template only
├─ .gitignore                 # .env, data/raw/, chroma/, __pycache__/
├─ README.md                  # setup, scope, chunking rationale, limits
├─ PRD.md
├─ mf_rag/
│  ├─ config.py               # paths, model names, thresholds (all tunable)
│  ├─ sources.py              # the 15-URL source registry → data/sources.csv
│  ├─ loaders.py              # STAGE 1  fetch + clean + extract
│  ├─ chunkers.py             # STAGE 2  Candidate A and Candidate B
│  ├─ embedder.py             # STAGE 3  single shared embedding function
│  ├─ store.py                # STAGE 4  ChromaDB upsert by chunk_id
│  ├─ guards.py               # PII + advice/returns + out-of-scope guards
│  ├─ retriever.py            # STAGE 5  embed query, top-k, MMR, threshold
│  ├─ prompts.py              # system prompt + citation wording
│  ├─ answerer.py             # STAGE 6  Groq call + post-checks + fallback
│  ├─ evalkit.py              # Recall@k, citation accuracy, orphan rate
│  └─ cli.py                  # ingest | chunk | embed | query | eval | app | all
├─ data/
│  ├─ sources.csv             # D3 — machine-readable source list
│  ├─ raw/<slug>.html         # cached raw HTML (gitignored, rebuildable)
│  ├─ processed/<slug>.md     # extracted text + front-matter (committed, NFR-6)
│  └─ chunks/
│     ├─ chunks.jsonl         # the chunk records
│     └─ chunks.txt           # human-readable mirror — inspect before trusting
├─ chroma/                    # persisted vector store (gitignored)
├─ eval/
│  ├─ chunking_eval.json      # 20 labelled questions: query, fact, expect_url
│  └─ report.md               # D7 — A vs B scores
└─ docs/
   ├─ sources.md              # D3 — source list, MD form
   ├─ sample_qa.md            # D5 — real generated Q&A
   ├─ disclaimer.md           # D6 — verbatim disclaimer
   └─ demo_script.md          # the 3-minute run-of-show
```

### 6.3 Data contracts

`data/sources.csv`

```csv
slug,url,title,scheme,category,page_role,fetched_at,http_status
hdfc-large-cap-fund-direct-growth,https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth,HDFC Large Cap Fund – Direct Growth,HDFC Large Cap Fund,large_cap,primary,2026-09-28,200
```

`data/processed/<slug>.md` front-matter: `title`, `url`, `scheme`, `category`, `page_role`, `fetched_at`.

`data/chunks/chunks.jsonl` — one JSON object per line:

```json
{"chunk_id":"hdfc-large-cap-fund-direct-growth#0007","source_url":"https://groww.in/...","title":"HDFC Large Cap Fund – Direct Growth","scheme":"HDFC Large Cap Fund","category":"large_cap","page_role":"primary","section":"Fees and charges","char_len":612,"fetched_at":"2026-09-28","text":"…"}
```

Chroma metadata per chunk: `chunk_id`, `source_url`, `title`, `scheme`, `category`,
`page_role`, `section`, `char_len`, `fetched_at`. **No chunk exists without a `source_url`.**

### 6.4 CLI

| Command | Stages run | Purpose |
| --- | --- | --- |
| `python -m mf_rag.cli ingest` | 1 | Fetch 15 URLs → raw HTML → processed markdown → `sources.csv` |
| `python -m mf_rag.cli chunk [--strategy a\|b]` | 2 | Chunk processed docs → `chunks.jsonl` + `chunks.txt` |
| `python -m mf_rag.cli embed` | 3, 4 | Embed + upsert into Chroma; idempotent by `chunk_id` |
| `python -m mf_rag.cli query "<question>" [--show-chunks]` | 5, 6 | One question → retrieved chunks + scores → answer with citation |
| `python -m mf_rag.cli eval` | 2, 5 | Score chunking A vs B on the labelled set → `docs/eval_report.md` |
| `python -m mf_rag.cli app` | 7 | Launch the Streamlit chat UI |
| `python -m mf_rag.cli all` | 1→4 | Clean full build; prints stage counts |

---

## 7. Chunking Strategy

**The AI agent inspects the data and proposes the strategy before writing code.** No strategy
is hard-coded from intuition; two candidates are implemented and measured, and the winner is
justified with numbers in the README.

### 7.1 Candidates

**Candidate A — Recursive character splitting**
`RecursiveCharacterTextSplitter(chunk_size≈700, chunk_overlap≈100, separators=["\n## ", "\n\n", "\n", ". ", " "])`
- Pro: robust to messy HTML-derived text, never drops content, trivially tunable.
- Con: can split a fee table across two chunks, so one chunk may not answer a question alone.

**Candidate B — Structure-aware splitting**
Split on the page's own headings and FAQ question boundaries first — scheme pages have natural
sections (Overview, Fees and charges, Exit load, Riskometer, Benchmark, Downloads, FAQ). Pack
sections under the size cap; recurse into A only when a section is oversized. Every chunk is
prefixed with `scheme — section` so it is self-describing.
- Pro: self-describing chunks, measurably better citation accuracy, fewer orphan chunks.
- Con: more code; depends on heading markup surviving extraction.

### 7.2 Decision rule (pre-committed, so the demo has an answer either way)

A **20-question labelled eval set** (4 per primary scheme), each row carrying the query, the
expected source URL, and the expected key fact string. Both candidates are indexed into separate
Chroma collections and scored on:

1. **Recall@5** — does the chunk containing the key fact appear in the top 5?
2. **Citation accuracy** — is the returned `source_url` the expected one?
3. **Orphan rate** — % of chunks with no heading/scheme context prefix.

Ship the higher-scoring candidate and record both scorecards in `docs/eval_report.md` and the
README.

**Provisional default:** Candidate B, with A as the inner fallback, because scheme pages are
section-structured and "every answer has a citation" is a hard constraint. The eval either
confirms this or overturns it — either outcome is a fine demo result.

### 7.3 Chunking requirements

- Every chunk ≥ 200 chars unless it is a standalone FAQ answer; drop fragments < 80 chars (nav
  crumbs, disclaimer footers, app-download CTAs).
- **Never split a numeric block.** Treat `Expense ratio`, `Exit load`, `Minimum SIP`, and
  `Minimum lump sum` as atomic units so a percentage is never orphaned from its label.
- Strip boilerplate that would pollute retrieval: "Groww App Download", disclaimer footers,
  referral text, cookie banners, "Trusted by 3M+ investors".
- Deduplicate near-identical chunks (e.g. Direct-plan boilerplate repeated on every page) by
  hashing normalized text; keep one copy, keep all source URLs so citations survive.
- Every chunk is mirrored to `data/chunks/chunks.txt` in readable form so a human can audit the
  chunking before trusting the retrieval.

---

## 8. Functional Requirements

| ID | Requirement |
| --- | --- |
| FR-1 | `ingest` fetches all 15 URLs with a polite rate limit and retry/backoff, saves raw HTML, extracts clean text, and writes `sources.csv`. Individual failures are logged and skipped, never fatal. |
| FR-2 | `chunk` splits processed docs per §7 and writes `chunks.jsonl` + `chunks.txt` with full metadata. |
| FR-3 | `embed` builds/updates Chroma collection `mf_facts_hdfc`; re-running is idempotent (upsert by `chunk_id`). |
| FR-4 | `query` prints retrieved chunks with similarity scores, then the final answer with its citation. |
| FR-5 | `app` launches the Streamlit chat UI. |
| FR-6 | Every factual answer contains at least one full source URL, rendered as a clickable link. |
| FR-7 | Every answer ends with `Last updated from sources: <fetched_at date>`. |
| FR-8 | Answers are ≤ 3 sentences. The system prompt enforces it; a post-check truncates at the third sentence and, if still over, falls back to the extractive answer. |
| FR-9 | Advice/portfolio/suitability questions ("should I buy", "is it good for me", "which is best", "compare returns", "portfolio allocation", "is it safe") get a fixed polite refusal plus an educational link, and **never reach the LLM context**. |
| FR-10 | Performance/return questions are refused for computation; the answer points to the official factsheet. No digits-with-`%`-return pattern is ever emitted. |
| FR-11 | If retrieval returns nothing above the similarity threshold, the bot says the fact is not in its sources and lists what it does cover. |
| FR-12 | PII detection: if input matches a PAN-like, Aadhaar-like, account-number-like, OTP, email, or phone pattern, the bot does not process it, does not log it, and replies with a privacy notice. |
| FR-13 | The UI shows a welcome line, 3 example questions, and the note "Facts-only. No investment advice." |
| FR-14 | The UI shows, per answer, the source link(s) and the matched chunk text (expandable) so retrieval is demonstrable live. |
| FR-15 | Every stage logs its counts — pages fetched → chunks created → vectors stored — so the pipeline is visible. |
| FR-16 | The post-check rejects any citation URL not present in `sources.csv`. |

---

## 9. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | all-MiniLM-L6-v2 runs on CPU; a full build completes in < 3 minutes on a laptop. |
| NFR-2 | Warm query → answer in < 5 seconds (excluding first-load model warm-up). |
| NFR-3 | Works offline after the first build — no runtime network calls except the LLM, and the extractive fallback needs none. |
| NFR-4 | No secrets in code. `GROQ_API_KEY` lives in `.env`; `.env.example` is committed; `.env` is gitignored. |
| NFR-5 | Deterministic-enough: temperature 0, fixed seed, no sampling randomness in retrieval. |
| NFR-6 | Processed markdown is committed, so the demo is reproducible without re-scraping. |
| NFR-7 | Every module has type hints and a docstring naming its pipeline stage. |

---

## 10. Safety, Compliance, and Transparency

These are **hard constraints**, tested before submission.

1. **Public sources only.** Provenance is the `source_url` on every chunk; no chunk exists
   without one, and every emitted URL is verified against `sources.csv`.
2. **No PII.** Nothing is collected or stored. Detection runs *before* logging, so a PAN never
   reaches disk. The UI copy tells users not to share personal identifiers.
3. **No performance claims.** No return computation or comparison anywhere in code or output.
   Enforced by a deterministic intent guard *and* a post-generation regex check.
4. **Clarity.** ≤ 3 sentences, one citation, and `Last updated from sources:` on every answer.
5. **No advice.** A deterministic intent guard runs before retrieval and before the LLM. Advice,
   "should I", suitability, and portfolio questions get a fixed refusal with an educational link.
6. **Disclaimer.** Visible in the UI header and appended to the footer of every answer.
7. **No fabrication.** A similarity threshold plus a "not in my sources" path prevents invented
   facts. When a field is missing from the source, the bot says it is missing and links the page.

### 10.1 Guard behaviour

| Input class | Detection | Response | Reaches LLM? |
| --- | --- | --- | --- |
| PII | regex: PAN `[A-Z]{5}\d{4}[A-Z]`, Aadhaar `\d{4}\s?\d{4}\s?\d{4}`, email, phone, 6-digit OTP, 8–18 digit account number | Privacy notice + "do not share personal identifiers" | No |
| Advice | keyword/pattern: *should I, is it good for me, recommend, best fund for, safe to, allocate, portfolio, suitable, worth buying* | Fixed refusal + educational link | No |
| Returns | *returns, performance, CAGR, best performing, % gain, how much did it give* | Refusal to compute + factsheet link | No |
| Out of scope | all retrieved chunks < `min_score` | "Not in my sources" + coverage list | No |
| Factual | everything else | Grounded answer + citation | Yes |

### 10.2 UI refusal copy (verbatim)

> I can share published facts about HDFC Mutual Fund schemes — fees, exit load, minimum SIP,
> lock-in, riskometer, benchmark and how to get documents. I can't give investment advice or
> compare returns. For that, please read the official factsheet:
> `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`

### 10.3 Disclaimer snippet (verbatim, used in the UI)

> **Facts-only. No investment advice.** Answers are generated from public HDFC Mutual Fund pages
> on Groww and may be outdated. Verify every number on the linked source page before acting.
> Mutual fund investments are subject to market risks; read all scheme-related documents
> carefully.

---

## 11. Acceptance Criteria

The demo is accepted if all of the following hold:

1. `python -m mf_rag.cli all` builds the index from a clean checkout and prints stage counts
   (pages fetched → chunks created → vectors stored).
2. Q1–Q6 (expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark) each
   return a correct fact **with a working citation**.
3. Q7 returns document-download steps with a citation.
4. Q8 returns an educational tax answer with a citation, sourced from an `education` page.
5. Q9 returns the refusal + link, with no advice language anywhere in the answer.
6. Q10 returns a refusal linking to the official factsheet, with no return figures.
7. A query containing a PAN / email / phone is refused and nothing is written to disk.
8. Every answer is ≤ 3 sentences and ends with the `Last updated from sources:` line.
9. "What is the weather in Delhi?" → *not in my sources*, with the coverage list.
10. `docs/sample_qa.md` contains 5–10 query/answer/citation rows produced by the **actual**
    build, not hand-written.
11. Both chunking candidates are scored and the winner is documented with numbers.
12. The `.env` holding `GROQ_API_KEY` is not committed and not present in git history.

---

## 12. Deliverables

| # | Deliverable | Location |
| --- | --- | --- |
| D1 | This PRD | `PRD.md` |
| D2 | Working prototype — Streamlit app + CLI (or a ≤3-min demo video if hosting isn't possible) | `app.py`, `mf_rag/`, link in README |
| D3 | Source list of the URLs used (CSV + MD) | `data/sources.csv`, `docs/sources.md` |
| D4 | README — setup steps, scope (AMC + schemes), architecture, chunking rationale, known limits | `README.md` |
| D5 | Sample Q&A — 5–10 queries with the assistant's real answers + links | `docs/sample_qa.md` |
| D6 | Disclaimer snippet used in the UI | `docs/disclaimer.md` (copy in §10.3) |
| D7 | Eval report — Recall@k, citation accuracy, orphan rate for both chunking candidates | `docs/eval_report.md` |
| D8 | Human-readable chunk dump for inspection | `data/chunks/chunks.txt` |

---

## 13. Milestones

| # | Milestone | Output | Est. |
| --- | --- | --- | --- |
| M0 | Environment: install Python 3.11, create venv, install deps, `.env` | `python --version` works | 0.25 day |
| M1 | Source registry + fetch/clean pipeline | `ingest` works, 15 pages stored, `sources.csv` written | 0.5 day |
| M2 | Chunking A vs B + eval harness | `docs/eval_report.md`, decided winner, `chunks.txt` | 0.5 day |
| M3 | Embedding + ChromaDB store | `embed` works, counts logged, idempotent | 0.25 day |
| M4 | Retrieval + guards + Groq answering with citations | `query` CLI works end to end | 0.5 day |
| M5 | Streamlit UI + refusal / PII / out-of-scope states | Runnable demo | 0.5 day |
| M6 | Acceptance pass, sample Q&A, README, demo recording | All deliverables | 0.5 day |

M0 exists because the missing Python install is the most likely thing to cost us the demo.

---

## 14. Known Limits, Risks, and Mitigations

| Limit / Risk | Impact | Mitigation |
| --- | --- | --- |
| **No Python on this machine** | Blocks everything | M0 is first; verify `python --version` before writing pipeline code |
| Groww pages are client-rendered; fee/riskometer data may be absent from static HTML | Missing facts in corpus | Use `trafilatura` full-text extraction; if a field is still absent, drop the claim and link the page. Never hand-enter a number. |
| Scraping is rate-limited or blocked | Fewer pages | Polite delay + retry with backoff; cache raw HTML so the build is reproducible offline |
| Pages change → answers go stale | Wrong facts over time | Store `fetched_at`, surface "Last updated from sources:", date the demo snapshot in the README |
| all-MiniLM-L6-v2 is a small English model | May miss paraphrase | Keep queries short and factual; MMR re-rank; escalate to a larger embedding only if eval shows recall problems |
| Chunk boundary splits a fee table | Incomplete answers | Structure-aware chunking (B) with numeric-block protection |
| Small corpus (15 pages) | Narrow coverage | Bot states its scope explicitly; out-of-scope questions take the "not in my sources" path |
| Hallucinated citation URL | Trust break | Post-check every emitted URL against `sources.csv`; drop the answer if absent |
| Education pages mistaken for scheme facts | Wrong facts | `page_role` metadata separates `primary` from `education`; the prompt forbids scheme numbers from education pages |
| Groq rate limit / outage mid-demo | Dead demo | Deterministic extractive fallback; ship a pre-built `sample_qa.md` so the demo never depends on a live call |
| Scope creep to "add one more AMC" | Schedule slip | Schema already supports it via `sources.csv`; explicitly out of scope for v1 |

---

## 15. Open Questions

1. **Demo hosting** — local Streamlit link plus a screen recording, or a free-tier deploy? Decided
   by M6; recording is the fallback and needs no hosting.
2. **Groq model** — `llama-3.3-70b-versatile` (best grounding, slower) vs `llama-3.1-8b-instant`
   (fast, good enough for extractive short answers). Default to 70b, drop to 8b if latency bites.
3. **Corpus breadth** — 15 Groww pages as listed, or add HDFC's official factsheet/SID PDFs if
   public direct links turn out to be reachable. Adding them would strengthen Q5 (riskometer) and
   Q6 (benchmark) considerably, since those are the two facts most likely to be missing from
   client-rendered HTML.
4. **Eval set authorship** — build the 20 labelled questions by reading the *fetched* corpus
   (not by memory), so expected facts match what the pages actually say on the snapshot date.

---

## 16. Decisions Log

| # | Decision | Rationale | Date |
| --- | --- | --- | --- |
| D1 | Corpus = 15 pages, 5 primary | Resolves the brief's "10 pages" vs "5 URLs" tension without re-scoping later | 2026-09-28 |
| D2 | Groq primary + extractive fallback | Brief mandates Groq; the fallback means a missing key can't kill the demo | 2026-09-28 |
| D3 | All in-scope schemes are Direct – Growth | Avoids regular/direct fee ambiguity in a facts demo | 2026-09-28 |
| D4 | ChromaDB persistent, ingest runs once | Brief requirement; makes warm start < 60 s | 2026-09-28 |
| D5 | Chunking decided by eval, not intuition | Brief explicitly requires inspecting data before choosing; pre-committed rule guarantees a result either way | 2026-09-28 |
| D6 | Plain Python orchestration, no LangChain agent | Keeps every RAG stage visible — the thing being graded | 2026-09-28 |

---

## Appendix A — Sample Q&A (shape only)

> These illustrate the **format** the bot must produce. D5 must be regenerated from the real
> build so the numbers are the ones actually in the fetched corpus.

**Q1. What is the expense ratio of HDFC Large Cap Fund – Direct Growth?**
HDFC Large Cap Fund – Direct Growth's expense ratio is listed in the "Fees and charges" section
of its scheme page on Groww. Direct plans charge a lower expense ratio than Regular plans because
no distribution commission is paid. See the source page for the current figure.
Source: `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`
Last updated from sources: 2026-09-28

**Q2. Is HDFC ELSS Tax Saver Fund locked in?**
Yes. HDFC ELSS Tax Saver Fund has a mandatory 3-year lock-in, which is also the minimum holding
period for the Section 80C deduction. Units cannot be redeemed before it completes.
Source: `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth`
Last updated from sources: 2026-09-28

**Q3. Should I buy HDFC Small Cap Fund?**
I can share published facts about HDFC schemes — fees, exit load, minimum SIP, lock-in,
riskometer and benchmark — but I can't give investment advice or recommend a scheme. Please read
the official scheme documents and factsheet, and consult a SEBI-registered investment adviser.
Educational link: `https://groww.in/mutual-funds/amc/hdfc-mutual-funds`

**Q4. Which HDFC fund has given the best returns?**
I don't compute or compare returns, and this assistant makes no performance claims. Return
figures for each scheme are published in its official factsheet, which you can reach from the
scheme page linked below.
Factsheet: `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`

**Q5. What is the riskometer level of HDFC Balanced Advantage Fund?**
SEBI's riskometer assigns every scheme a level from 1 (very low) to 7 (very high) based on the
risk of its underlying portfolio. HDFC Balanced Advantage Fund's current level is shown in the
riskometer section of its scheme page, and the scale itself is explained on Groww's riskometer
page.
Source: `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` ·
Riskometer scale: `https://groww.in/p/riskometer`
Last updated from sources: 2026-09-28

---

## Appendix B — Source List (MD form, mirrors `data/sources.csv`)

| # | `page_role` | Scheme / topic | URL |
| --- | --- | --- | --- |
| 1 | `primary` | HDFC Large Cap – Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| 2 | `primary` | HDFC Equity (Flexi Cap) – Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| 3 | `primary` | HDFC ELSS Tax Saver – Direct Plan Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| 4 | `primary` | HDFC Small Cap – Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| 5 | `primary` | HDFC Balanced Advantage – Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |
| 6 | `variant` | HDFC Large Cap – Regular Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth` |
| 7 | `variant` | HDFC Large Cap – Direct IDCW | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw` |
| 8 | `variant` | HDFC Focused Large Cap – Direct Growth | `https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth` |
| 9 | `amc` | HDFC Mutual Funds (fund house) | `https://groww.in/mutual-funds/amc/hdfc-mutual-funds` |
| 10 | `category` | Best Large & Mid Cap mutual funds | `https://groww.in/mutual-funds/category/best-large-and-midcap-mutual-funds` |
| 11 | `category` | Best Flexi Cap mutual funds | `https://groww.in/mutual-funds/category/best-flexi-cap-mutual-fund` |
| 12 | `tool` | SIP calculator | `https://groww.in/calculators/sip-calculator` |
| 13 | `regulatory` | Riskometer (SEBI) | `https://groww.in/p/riskometer` |
| 14 | `education` | ELSS vs SIP | `https://groww.in/questions/what-is-the-difference-between-elss-and-sip` |
| 15 | `education` | Tax on mutual funds | `https://groww.in/blog/tax-on-mutual-funds/` |

**AMC:** HDFC Mutual Funds · **Schemes in scope:** 5 (all Direct – Growth) · **Corpus:** 15 public pages.
