# PRD — MF Facts RAG Chatbot (HDFC AMC · Class Demo)

| Field | Value |
| --- | --- |
| Document | Product Requirements Document (v1) |
| Status | Draft for review |
| Owner | RAG demo team |
| Last updated | 2026-09-28 |
| Type | Academic / class demo prototype — not a production financial product |

---

## 1. Problem Statement

Retail users and support teams repeatedly ask the same factual questions about mutual fund schemes: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer level, benchmark, and how to download statements. Today these answers live across many pages of an AMC's site, Groww scheme pages, and SEBI disclosures, so answering them means tab-hunting, and generic AI chatbots happily blend them with opinions and return figures.

We will build a **facts-only RAG chatbot** that answers mutual fund scheme questions using **only official public pages** as its knowledge base. Every answer must carry one source link. It must refuse opinionated or portfolio questions politely and point to an educational link instead. It must never give investment advice, never compute or compare returns, and never store personal data.

This is a class demo. The goal is to demonstrate the full RAG pipeline end to end — **Loading → Chunking → Embedding → Storing vector data → Retrieval → Answering** — with clean architecture, visible citations, and compliance guardrails.

---

## 2. Goals

| # | Goal | Measure |
| --- | --- | --- |
| G1 | Answer factual MF scheme questions from an official-only corpus | ≥ 80% of the eval set answered with a correct fact and a valid citation |
| G2 | Show every RAG stage explicitly and repeatably | One command ingests → chunks → embeds → stores; stages are logged with counts |
| G3 | Guarantee a citation on every factual answer | 100% of factual answers include ≥ 1 source URL |
| G4 | Never produce advice or performance claims | 100% refusal on the advice/returns test set |
| G5 | Be demo-able in 3 minutes | Cold start to first answer < 60s on a warm index |

## 3. Non-Goals (Out of Scope)

- No investment advice, buy/sell/hold calls, or portfolio recommendations.
- No return computation, no CAGR/absolute-return figures, no fund comparison or ranking. If asked, link to the official factsheet.
- No live NAV, no real-time prices, no fund performance data generation.
- No login, signup, user accounts, or saved chat history.
- No multi-AMC coverage in v1 (HDFC only). Architecture must allow adding an AMC later.
- No mobile app, no production deployment, no SLA.
- No PII collection of any kind (PAN, Aadhaar, account numbers, OTPs, emails, phone numbers).

---

## 4. Users and Use Cases

**Primary — retail investor (comparison stage)**
"I want to compare two HDFC schemes on facts before I read the factsheet." → Wants expense ratio, exit load, minimum SIP, benchmark, riskometer, and a link to verify each.

**Secondary — support/content team**
"What's the ELSS lock-in period?" → Wants the same answer, fast, with a link they can paste into a ticket.

**Both** want short, factual, source-backed answers — not a paragraph of prose and not a sales pitch.

### Canonical queries the bot must handle

| Intent | Example question | Expected behaviour |
| --- | --- | --- |
| Fees | "Expense ratio of HDFC Large Cap?" | Fact + citation |
| Exit load | "Exit load on HDFC Flexi Cap?" | Fact + citation |
| Minimum SIP | "Minimum SIP for HDFC Small Cap?" | Fact + citation |
| Lock-in | "Is HDFC ELSS Tax Saver locked in?" | Fact (3 years) + citation |
| Riskometer | "What is the riskometer level of HDFC Balanced Advantage?" | Fact + citation |
| Benchmark | "Benchmark of HDFC Equity (Flexi Cap)?" | Fact + citation |
| Documents | "How do I download my capital gains statement?" | Steps + citation |
| Tax | "How is ELSS taxed?" | Educational answer + citation |
| Advice (refuse) | "Should I buy HDFC Small Cap?" | Polite refusal + educational link |
| Returns (refuse) | "Which HDFC fund gave the best returns?" | Polite refusal + link to official factsheet |

---

## 5. Scope — AMC and Schemes

**AMC:** HDFC Mutual Funds
**Platform used for public pages:** Groww (public scheme pages) + Groww AMC page + Groww/SEBI riskometer and investor-education pages.

All five schemes are **Direct – Growth** plans.

| # | Category | Scheme | Groww URL |
| --- | --- | --- | --- |
| S1 | Large Cap | HDFC Large Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| S2 | Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| S3 | ELSS | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| S4 | Small Cap | HDFC Small Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| S5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

Coverage intent: one large-cap, one flexi-cap, one ELSS, one small-cap, one hybrid — enough variety to show the retriever handling different page structures and different question types.

### 5.1 Source pages (public only)

**Scheme pages (primary corpus, 5)**
1. `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`
2. `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth`
3. `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth`
4. `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth`
5. `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth`

**Context / variants (secondary, 5)**
6. `https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth` (plan-variant context: Regular vs Direct)
7. `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw` (IDCW variant context)
8. `https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth` (nearby-scheme disambiguation)
9. `https://groww.in/mutual-funds/amc/hdfc-mutual-funds` (AMC / fund house context)
10. `https://groww.in/mutual-funds/category/best-flexi-cap-mutual-fund` (category context)

**Tools / regulatory / education (tertiary, 5)**
11. `https://groww.in/calculators/sip-calculator` (SIP concepts)
12. `https://groww.in/p/riskometer` (SEBI riskometer — regulatory)
13. `https://groww.in/questions/what-is-the-difference-between-elss-and-sip` (ELSS vs SIP concepts)
14. `https://groww.in/blog/tax-on-mutual-funds/` (tax concepts)

Total: 15 public pages, 5 primary scheme pages. Machine-readable copy lives in `data/sources.csv`.

**Source rules (hard constraints)**
- Public pages only. No logged-in/back-end app screens, no screenshots of dashboards.
- No third-party blogs as sources. `groww.in/blog/...` is allowed only as *investor-education* content, never as a source of scheme facts (expense ratio, exit load, benchmark, riskometer).
- Any page that cannot be fetched publicly at build time is dropped from the corpus and recorded as such in the README. No hand-typed facts.

---

## 6. System Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │  1. LOADING / INGESTION                      │
                 │  fetch 15 public URLs (rate-limited, retried) │
                 │  → raw HTML saved to data/raw/<slug>.html     │
                 │  → extract main text, drop nav/footer/ads     │
                 │  → clean: collapse whitespace, fix mojibake   │
                 │  → data/processed/<slug>.md (text + metadata)│
                 └───────────────────┬──────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  2. CHUNKING  (adaptive — see §7)            │
                 │  heading/FAQ-aware first, recursive fallback  │
                 │  → every chunk carries scheme + source_url    │
                 └───────────────────┬──────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  3. EMBEDDING                                │
                 │  sentence-transformers/all-MiniLM-L6-v2      │
                 │  384-dim, CPU-friendly, no API key            │
                 └───────────────────┬──────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  4. VECTOR STORE                             │
                 │  ChromaDB (persistent_client)                │
                 │  collection: mf_facts_hdfc                  │
                 │  metadata: source_url, scheme, section,      │
                 │            chunk_id, fetched_at             │
                 └───────────────────┬──────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  5. RETRIEVAL  (pre-LLM, cheap filters)     │
                 │  a. guard: advice / returns / PII → refuse    │
                 │  b. embed query, cosine top-k=6              │
                 │  c. optional MMR re-rank for diversity       │
                 │  d. threshold: drop chunks below min_score    │
                 │     → "I don't have that in my sources"       │
                 │  e. build prompt with numbered context blocks │
                 └───────────────────┬──────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  6. ANSWERING                                │
                 │  LLM: grounded, facts-only, ≤3 sentences      │
                 │  output = answer + one citation + as_of date  │
                 │  post-check: no digits-with-% returns, no    │
                 │  advice verbs, citation present               │
                 └───────────────────┬──────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  7. TINY UI  (Streamlit chat)                 │
                 │  welcome line + 3 example questions +        │
                 │  "Facts-only. No investment advice."          │
                 └──────────────────────────────────────────────┘
```

### 6.1 Stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.11+ | Ecosystem for ChromaDB + sentence-transformers |
| Fetch | `httpx` + `beautifulsoup4` + `trafilatura` (optional) | Static HTML, no browser needed |
| Chunking | `langchain-text-splitters` (RecursiveCharacterTextSplitter) + custom heading splitter | Mature, tunable, no LLM cost |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~80MB, CPU, no API key |
| Vector DB | `chromadb` persistent client | Free, local, zero-config, demo-friendly |
| Orchestration | Plain Python + optional LangChain | Keeps the pipeline visible for the demo |
| LLM | Configurable: OpenAI **or** Ollama (`llama3.1`) **or** deterministic extractive fallback | Demo must run without paid API keys |
| UI | `streamlit` | `streamlit run app.py` → shareable local link |
| Tracking | `langchain` optional / simple local JSONL eval log | Show retrieval quality during demo |

> **Prerequisite on this machine:** Python is **not currently installed** (only Node 24 and Git). Install Python 3.11+ and add it to PATH before the first `pip install`.

### 6.2 Data contracts

`data/sources.csv`
```csv
slug,url,title,scheme,category,page_role,fetched_at,http_status
hdfc-large-cap-fund-direct-growth,https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth,HDFC Large Cap Fund – Direct Growth,HDFC Large Cap Fund,large_cap,primary,2026-09-28,200
```

`data/processed/<slug>.md` front-matter: `title`, `url`, `scheme`, `category`, `page_role`, `fetched_at`.

Chroma metadata per chunk: `chunk_id`, `source_url`, `title`, `scheme`, `category`, `page_role`, `section`, `char_len`, `fetched_at`.

---

## 7. Chunking Strategy (decision required, and how we decide it)

The data drives this. We do **not** lock a single strategy blindly; we pick between two candidates by measuring them on a small labelled eval set built from the 5 primary scheme pages.

**Candidate A — Recursive character splitting**
`RecursiveCharacterTextSplitter(chunk_size≈700 chars, chunk_overlap≈100, separators=["\n## ", "\n\n", "\n", ". ", " "])`
- Pro: robust to messy HTML-derived text, never drops content, trivially tunable.
- Con: can split a fee table across two chunks, so a single chunk may not answer a question alone.

**Candidate B — Semantic / structure-aware splitting**
Split on page headings and FAQ question boundaries first (scheme pages have natural sections: Overview, Fees, Exit load, Riskometer, Benchmark, Downloads, FAQ). Pack sections under the size cap; if a section is oversized, recurse (A) inside it. Each chunk is prefixed with its section heading and its scheme name.
- Pro: chunks are self-describing, which measurably improves citation accuracy and reduces "orphan" chunks.
- Con: more code; depends on heading markup surviving extraction.

**Decision rule (pre-committed, so the demo has an answer either way)**
Build a 20-question eval set (4 per scheme) with the expected source URL and the expected key fact. For each candidate, run retrieval and score:
1. **Recall@5** — does the chunk containing the fact appear in the top 5?
2. **Citation accuracy** — is the returned `source_url` the expected one?
3. **Orphan rate** — % of chunks with no heading/scheme context.

Ship the higher-scoring candidate. Record both scores in the README as the justification. **Provisional default: Candidate B with Candidate A as the inner fallback**, because scheme pages are section-structured and the citation requirement is a hard constraint.

**Chunking requirements**
- Every chunk ≥ 200 chars unless it is a standalone FAQ answer; drop fragments < 80 chars (nav crumbs, disclaimers).
- Never split mid-number for fee values — treat `Expense ratio`, `Exit load`, `Minimum SIP`, `Minimum lump sum` blocks as atomic.
- Strip boilerplate that would pollute retrieval: "Groww App Download", "Disclaimer" footers, referral text, cookie banners.
- Deduplicate near-identical chunks (e.g. Direct-plan boilerplate repeated on every page) by hashing normalized text; keep one, keep all source URLs.

---

## 8. Functional Requirements

| ID | Requirement |
| --- | --- |
| FR-1 | `ingest` command fetches all 15 URLs with a polite rate limit, saves raw HTML, extracts clean text, and writes `sources.csv`. Failures are logged, not fatal. |
| FR-2 | `chunk` command splits processed docs per §7 and writes chunks to `data/chunks.jsonl` with full metadata. |
| FR-3 | `embed` command builds/updates the ChromaDB collection `mf_facts_hdfc`; re-running is idempotent (upsert by `chunk_id`). |
| FR-4 | `query` command takes one question, prints the retrieved chunks with scores, then the final answer with citation. |
| FR-5 | `app` command launches the Streamlit chat UI. |
| FR-6 | Every factual answer contains at least one full source URL rendered as a clickable link. |
| FR-7 | Every answer ends with `Last updated from sources: <fetched_at date>`. |
| FR-8 | Answers are ≤ 3 sentences. The prompt enforces this and a post-check truncates/regenerates if exceeded. |
| FR-9 | Advice/portfolio questions ("should I buy", "is it good for me", "which is best", "compare returns", "portfolio allocation") are refused with a fixed polite message + educational link, and never reach the LLM context. |
| FR-10 | Performance/return questions are refused for computation; the answer links to the official factsheet instead. |
| FR-11 | If retrieval returns nothing above the similarity threshold, the bot says it does not have that in its sources and lists what it does cover. |
| FR-12 | PII detection: if input contains a PAN-like, Aadhaar-like, account-number-like, OTP, email, or phone pattern, the bot does not process it, does not log it, and replies with a privacy notice. |
| FR-13 | The UI shows a welcome line, 3 example questions, and the disclaimer "Facts-only. No investment advice." |
| FR-14 | The UI shows, for each answer, the source link(s) and the matched chunk text (expandable) so retrieval is demonstrable in a 3-minute demo. |
| FR-15 | Stage logs print counts: pages fetched → chunks created → vectors stored, so the pipeline is visible. |

## 9. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | All-MiniLM-L6-v2 runs on CPU; full build completes in < 3 minutes on a laptop. |
| NFR-2 | Warm query → answer in < 5 seconds. |
| NFR-3 | Works offline after first build (no runtime network calls needed for the LLM if Ollama is used). |
| NFR-4 | No secrets in code; `.env` for the LLM key, `.env.example` provided. |
| NFR-5 | Deterministic-enough: temperature 0, fixed seed, no sampling randomness in retrieval. |
| NFR-6 | Corpus is versioned in git as processed markdown, so the demo is reproducible without re-scraping. |
| NFR-7 | Every module has type hints and a short docstring naming its pipeline stage. |

## 10. Safety, Compliance, and Transparency

These are **hard constraints**, tested before submission.

1. **Public sources only** — provenance is the `source_url` on every chunk; no chunk exists without one.
2. **No PII** — no PAN, Aadhaar, account number, OTP, email, or phone is stored. Logs store question text only after a PII scrub.
3. **No performance claims** — no return computation or comparison anywhere in the code or output. Enforced by a post-generation regex check plus a refusal rule.
4. **Clarity** — ≤ 3 sentences, one citation, "Last updated from sources:" on every answer.
5. **No advice** — a deterministic intent guard runs before retrieval/LLM. Advice, "should I", suitability, and portfolio questions get a fixed refusal.
6. **Disclaimer** — visible in the UI header and appended to every answer footer.
7. **No fabrication** — a chunk threshold plus a "not in my sources" path prevents invented facts.

### UI refusal copy
> I can share published facts about HDFC Mutual Fund schemes — fees, exit load, minimum SIP, lock-in, riskometer, benchmark and how to get documents. I can't give investment advice or compare returns. For that, please read the official factsheet: `https://groww.in/mutual-funds/hdfc-mutual-funds`

### Disclaimer snippet (used in the UI, verbatim)
> **Facts-only. No investment advice.** Answers are generated from public HDFC Mutual Fund pages on Groww and may be outdated. Verify every number on the linked source page before acting. Mutual fund investments are subject to market risks; read all scheme-related documents carefully.

## 11. Acceptance Criteria

The demo is accepted if all of the following hold:

1. `python -m mf_rag.cli all` builds the index from a clean checkout and prints stage counts.
2. The 5 target queries (expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer/benchmark) each return a correct fact **with a working citation**.
3. "Should I buy HDFC Small Cap?" returns the refusal + link, with no advice language.
4. "Which HDFC fund has the highest returns?" returns a refusal linking to the official factsheet, with no numbers.
5. A PAN/email/phone in the query is refused and not stored.
6. Every answer is ≤ 3 sentences and ends with the "Last updated from sources:" line.
7. Unrelated question ("What is the weather in Delhi?") → "not in my sources".
8. Sample Q&A file contains 5–10 real query/answer/citation rows generated by the actual build, not hand-written.
9. The two chunking candidates are scored and the winner is documented in the README.

## 12. Deliverables

| # | Deliverable | Location |
| --- | --- | --- |
| D1 | This PRD | `Problem Statement/Prd.md` |
| D2 | Working prototype (Streamlit app + CLI) or ≤3-min demo video | repo root / link in README |
| D3 | Source list of the URLs used (CSV + MD) | `data/sources.csv`, `docs/sources.md` |
| D4 | README: setup steps, scope (AMC + 5 schemes), architecture, chunking rationale, known limits | `README.md` |
| D5 | Sample Q&A: 5–10 queries with the assistant's real answers + links | `docs/sample_qa.md` |
| D6 | Disclaimer snippet used in the UI | `docs/disclaimer.md` (copy in §10) |
| D7 | Eval report: Recall@k, citation accuracy, orphan rate for both chunking candidates | `docs/eval_report.md` |

## 13. Milestones

| # | Milestone | Output | Est. |
| --- | --- | --- | --- |
| M1 | Scaffold + sources.csv + fetch/clean pipeline | `ingest` works, 15 pages stored | 0.5 day |
| M2 | Chunking A vs B + eval harness | `docs/eval_report.md` with a decided winner | 0.5 day |
| M3 | Embedding + ChromaDB store | `embed` works, counts logged | 0.25 day |
| M4 | Retrieval + guard rails + LLM answering with citations | `query` CLI works | 0.5 day |
| M5 | Streamlit UI + refusal/PII states | Runnable demo | 0.5 day |
| M6 | Acceptance test pass, sample Q&A, README, demo recording | All deliverables | 0.5 day |

## 14. Known Limits & Risks

| Limit / Risk | Impact | Mitigation |
| --- | --- | --- |
| Groww pages are client-rendered; some fee/riskometer data may not be in static HTML | Missing facts in corpus | Use `trafilatura` full-text extraction; if a field is still absent, drop the claim and point to the page link. Never hand-enter numbers. |
| Scraping is rate-limited or blocked | Fewer pages | Polite delay + retry with backoff + cache raw HTML in git so the build is reproducible offline. |
| Pages change → answers go stale | Wrong facts over time | Store `fetched_at`, surface "Last updated from sources:", and note the demo snapshot date in the README. |
| all-MiniLM-L6-v2 is a small English model | May miss paraphrase | Keep queries short and factual; use MMR re-rank; escalate to a larger model only if eval shows a problem. |
| Chunk boundary splits a fee table | Incomplete answers | Structure-aware chunking (Candidate B) with numeric-block protection. |
| Small corpus (≈15 pages) | Narrow coverage | Bot states its scope explicitly; out-of-scope questions get the "not in my sources" path. |
| Hallucinated citation URL | Trust break | Post-check that every emitted URL exists in `sources.csv`; drop the answer if not. |
| Educational blog pages could be mistaken for scheme facts | Wrong facts | `page_role` metadata separates `primary` from `education`; the prompt forbids using education pages for scheme numbers. |

## 15. Open Questions

1. **LLM choice for the demo** — OpenAI API (needs a key) vs Ollama local (no key, slower first load) vs extractive fallback (always works, no LLM). Recommendation: implement the deterministic extractive fallback so the demo never breaks, and use an LLM when available.
2. **Embedding model** — confirm `all-MiniLM-L6-v2` (per the brief) for v1; consider `all-mpnet-base-v2` only if the eval shows recall problems.
3. **Corpus breadth** — 15 pages as listed, or add HDFC's official factsheet/SID PDFs if public direct links are available.
4. **Demo hosting** — local Streamlit link recorded in a video, or deployed free tier?

---

## Appendix A — Sample Q&A (shape; regenerated from the real build for D5)

**Q1. What is the expense ratio of HDFC Large Cap Fund – Direct Growth?**
Direct Growth plans of HDFC Large Cap Fund carry a lower expense ratio than Regular plans because distribution commissions are not paid. The exact percentage is listed in the scheme's "Fees" section — see the source page for the current figure.
Source: `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`
Last updated from sources: 2026-09-28

**Q2. Is HDFC ELSS Tax Saver Fund locked in?**
Yes. HDFC ELSS Tax Saver Fund has a mandatory 3-year lock-in period, the minimum period for claiming the Section 80C tax benefit. Units cannot be redeemed before it completes.
Source: `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth`
Last updated from sources: 2026-09-28

**Q3. Should I buy HDFC Small Cap Fund?**
I can share published facts about HDFC schemes — fees, exit load, minimum SIP, lock-in, riskometer and benchmark — but I can't give investment advice or recommend a scheme. Please read the official scheme documents and factsheet, and consult a SEBI-registered adviser.
Educational link: `https://groww.in/mutual-funds/amc/hdfc-mutual-funds`

**Q4. Which HDFC fund has given the best returns?**
I don't compute or compare returns, and this assistant makes no performance claims. Return figures for each scheme are published in its official factsheet, which you can open from the scheme page linked below.
Factsheet: `https://groww.in/mutual-funds/hdfc-mutual-funds`

**Q5. What is the riskometer level of HDFC Balanced Advantage Fund?**
The SEBI riskometer assigns each scheme a level from 1 (very low) to 7 (very high) based on the risk of the underlying portfolio. HDFC Balanced Advantage Fund's current level is shown in the "Riskometer" section of its scheme page, and the scale itself is explained on Groww's riskometer page.
Source: `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` · Riskometer scale: `https://groww.in/p/riskometer`
Last updated from sources: 2026-09-28

## Appendix B — Source list (MD form, mirrors `data/sources.csv`)

| # | Page role | Scheme / topic | URL |
| --- | --- | --- | --- |
| 1 | primary | HDFC Large Cap – Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| 2 | primary | HDFC Equity (Flexi Cap) – Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| 3 | primary | HDFC ELSS Tax Saver – Direct Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| 4 | primary | HDFC Small Cap – Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| 5 | primary | HDFC Balanced Advantage – Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |
| 6 | variant | HDFC Large Cap – Regular Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth` |
| 7 | variant | HDFC Large Cap – Direct IDCW | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw` |
| 8 | variant | HDFC Focused Large Cap – Direct Growth | `https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth` |
| 9 | amc | HDFC Mutual Funds (fund house) | `https://groww.in/mutual-funds/amc/hdfc-mutual-funds` |
| 10 | category | Best Flexi Cap mutual funds | `https://groww.in/mutual-funds/category/best-flexi-cap-mutual-fund` |
| 11 | tool | SIP calculator | `https://groww.in/calculators/sip-calculator` |
| 12 | regulatory | Riskometer (SEBI) | `https://groww.in/p/riskometer` |
| 13 | education | ELSS vs SIP | `https://groww.in/questions/what-is-the-difference-between-elss-and-sip` |
| 14 | education | Tax on mutual funds | `https://groww.in/blog/tax-on-mutual-funds/` |

(15th slot reserved for a category page, e.g. `https://groww.in/mutual-funds/category/best-large-and-midcap-mutual-funds`, added to `sources.csv` at build time and logged in the README.)
