# Implementation Guide — MF Facts RAG Chatbot

| Field | Value |
| --- | --- |
| Document | Phase-wise Implementation Guide (v1.0) |
| Derived from | [`architecture.md`](./architecture.md) v1.0 → [`PRD.md`](./PRD.md) v1.0 |
| Purpose | Drive an AI coding agent (OpenCode / Cursor / Claude Code) through the build, one phase at a time |
| Last updated | 2026-09-28 |
| Total phases | 14 (2 hard gates) |

---

## How to Use This Document

Work top to bottom. Each phase is **independently verifiable** and ends with a commit.

### Operating rules for the coding agent

| # | Rule | Why |
| --- | --- | --- |
| R1 | **Never invent a mutual fund fact.** Every number in an answer traces to a fetched chunk. | PRD §5.3, §10.7 — the whole point of the project |
| R2 | **Never write a value from memory into a file.** If a fact isn't in the fetched corpus, the bot must say it's missing. | Prevents exactly the failure this demo exists to avoid |
| R3 | **Type hints + a docstring naming the pipeline stage on every module and public function.** | NFR-7 |
| R4 | **All tunables in `config.py`.** No module reads `os.environ` directly. | §13 config reference |
| R5 | **Never commit `.env`.** `.env.example` only. | NFR-4, acceptance criterion 12 |
| R6 | **Do not start a phase until the previous phase's exit criteria pass.** | Ordering dependency |
| R7 | **A `⛔ GATE` phase must be reported to the user with real output before continuing.** | Two gates protect against building on bad data |
| R8 | **Log the stage count line on every run.** | FR-15, principle P5 |
| R9 | **Comments explain *why*, not *what*.** Don't narrate the code. | — |
| R10 | **If something fails, report it — do not silently work around it.** | Demo honesty |

### Phase map

| Phase | Stage | Output | Gate |
| --- | --- | --- | --- |
| 0 | — | Python env, venv, deps, `.env` | |
| 1 | — | `config.py`, `sources.py` | |
| 2 | 1 | `loaders.py` — fetch + clean | |
| 3 | 1 | **Data inspection** | ⛔ **GATE 1** |
| 4 | 2 | `chunkers.py` — Candidates A + B | |
| 5 | 2/5 | `evalkit.py` — chunking decision | ⛔ **GATE 2** |
| 6 | 3+4 | `embedder.py`, `store.py` | |
| 7 | 5a | `guards.py` | |
| 8 | 5b | `retriever.py` | |
| 9 | 6 | `prompts.py`, `answerer.py` | |
| 10 | all | `cli.py` | |
| 11 | 7 | `app.py` — Streamlit UI | |
| 12 | — | `tests/` | |
| 13 | — | `README.md`, `docs/*`, acceptance pass | |

---

## Phase 0 — Environment Setup

**Goal.** A working Python 3.11+ environment with all dependencies installed.

**Why first.** This machine has Node 24 and Git but **no Python**. Nothing else can be built
until this passes, and it is the single most likely cause of a failed demo.

### Steps

1. Install Python 3.11+ and add it to PATH. Verify:
   ```powershell
   python --version     # expect 3.11.x or newer
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Create `requirements.txt`:
   ```
   httpx>=0.27
   beautifulsoup4>=4.12
   trafilatura>=1.12
   langchain-text-splitters>=0.2
   sentence-transformers>=3.0
   chromadb>=0.5
   groq>=0.11
   streamlit>=1.37
   pydantic-settings>=2.3
   python-dotenv>=1.0
   typer>=0.12
   pandas>=2.2
   pytest>=8.0
   ```
4. Install:
   ```powershell
   pip install -r requirements.txt
   ```
5. Create `.env` and `.env.example`:
   ```dotenv
   # .env — NEVER COMMIT
   # Overrides are namespaced: every key must start with MF_RAG_
   MF_RAG_GROQ_API_KEY=your_key_here
   MF_RAG_LLM_MODEL=llama-3.3-70b-versatile
   MF_RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
   MF_RAG_MIN_SCORE=0.25
   ```
   `.env.example` has the same keys with an empty `MF_RAG_GROQ_API_KEY=`.
6. Create `.gitignore`:
   ```
   .env
   .venv/
   __pycache__/
   *.pyc
   chroma/
   data/raw/
   .pytest_cache/
   ```
7. Create the package skeleton and the directory tree:
   ```
   mf-rag-chatbot/
   ├─ mf_rag/{__init__,config,sources,loaders,chunkers,embedder,store,
   │          guards,retriever,prompts,answerer,evalkit,cli}.py
   ├─ tests/
   ├─ eval/
   ├─ docs/
   └─ data/{raw,processed,chunks}/
   ```
   `mf_rag/__init__.py` should contain only a docstring and `__version__ = "1.0.0"`.

### Verification

```powershell
python -c "import chromadb, sentence_transformers, streamlit, groq, typer, trafilatura; print('deps OK')"
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print(m.encode(['test']).shape)"
# expect (1, 384)
git status --porcelain   # .env must NOT appear
```

### Exit criteria

- [ ] `python --version` → 3.11+
- [ ] All imports succeed
- [ ] Embedding model loads and returns shape `(1, 384)`
- [ ] `git status` does not list `.env`

**Commit:** `chore: python env, deps, gitignore, package skeleton`

---

## Phase 1 — Config + Source Registry

**Goal.** `config.py` (all tunables) and `sources.py` (the 15-URL registry).

**Why first.** Every later phase imports these. The registry is the single place a URL exists —
no URL is ever hardcoded elsewhere in the codebase.

### Files to create

#### `mf_rag/config.py`

```python
"""Configuration — every tunable in the system lives here. Not a pipeline stage."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central settings, overridable by environment / .env."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Paths
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    chunks_dir: Path = ROOT / "data" / "chunks"
    chroma_path: Path = ROOT / "chroma"

    # STAGE 3 — embedding
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    # STAGE 4 — vector store
    chroma_collection: str = "mf_facts_hdfc"

    # STAGE 2 — chunking
    chunk_strategy: str = "b"           # "a" | "b" — decided by Phase 5
    chunk_size: int = 700
    chunk_overlap: int = 100
    chunk_min_chars: int = 200
    chunk_hard_drop_chars: int = 80

    # STAGE 5 — retrieval
    top_k: int = 6
    mmr_lambda: float = 0.7
    min_score: float = 0.25             # calibrated in Phase 5

    # STAGE 6 — answering
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 250
    max_sentences: int = 3

    # STAGE 1 — loading
    http_timeout: int = 30
    http_retries: int = 3
    fetch_delay_seconds: float = 1.5
    user_agent: str = "mf-facts-rag-demo/1.0 (+class project; contact: student@example.com)"


settings = Settings()
for _d in (settings.data_dir, settings.raw_dir, settings.processed_dir, settings.chunks_dir):
    _d.mkdir(parents=True, exist_ok=True)
```

> **Note.** Replace the placeholder contact in `user_agent` with a real one before the live
> fetch. A fake contact address in a scraper's UA string is poor practice even in a demo.

#### `mf_rag/sources.py`

```python
"""Source registry — the ONLY place a corpus URL is defined. Not a pipeline stage."""
from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from .config import ROOT


@dataclass(frozen=True)
class SourceSpec:
    """One public source page in the corpus."""
    slug: str
    url: str
    title: str
    scheme: str          # "" for non-scheme pages
    category: str        # "" for non-scheme pages
    page_role: str       # primary|variant|amc|category|tool|regulatory|education


SOURCES: tuple[SourceSpec, ...] = (
    # ── Primary: the 5 in-scope schemes, all Direct-Growth ──────────────
    SourceSpec("hdfc-large-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "HDFC Large Cap Fund – Direct Growth", "HDFC Large Cap Fund", "large_cap", "primary"),
    SourceSpec("hdfc-equity-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        "HDFC Equity (Flexi Cap) Fund – Direct Growth", "HDFC Equity (Flexi Cap) Fund", "flexi_cap", "primary"),
    SourceSpec("hdfc-elss-tax-saver-fund-direct-plan-growth",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        "HDFC ELSS Tax Saver Fund – Direct Plan Growth", "HDFC ELSS Tax Saver Fund", "elss", "primary"),
    SourceSpec("hdfc-small-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "HDFC Small Cap Fund – Direct Growth", "HDFC Small Cap Fund", "small_cap", "primary"),
    SourceSpec("hdfc-balanced-advantage-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        "HDFC Balanced Advantage Fund – Direct Growth", "HDFC Balanced Advantage Fund", "hybrid", "primary"),

    # ── Context: plan variants and disambiguation ───────────────────────
    SourceSpec("hdfc-large-cap-fund-regular-growth",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth",
        "HDFC Large Cap Fund – Regular Growth", "HDFC Large Cap Fund", "large_cap", "variant"),
    SourceSpec("hdfc-large-cap-fund-direct-idcw",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw",
        "HDFC Large Cap Fund – Direct IDCW", "HDFC Large Cap Fund", "large_cap", "variant"),
    SourceSpec("hdfc-focused-large-cap-direct-plan-growth",
        "https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth",
        "HDFC Focused Large Cap Fund – Direct Growth", "HDFC Focused Large Cap Fund", "large_cap", "variant"),

    # ── Context: AMC, categories, tool, regulatory, education ───────────
    SourceSpec("hdfc-mutual-funds",
        "https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
        "HDFC Mutual Funds (AMC)", "", "", "amc"),
    SourceSpec("best-large-and-midcap-mutual-funds",
        "https://groww.in/mutual-funds/category/best-large-and-midcap-mutual-funds",
        "Best Large and Mid Cap Mutual Funds", "", "", "category"),
    SourceSpec("best-flexi-cap-mutual-fund",
        "https://groww.in/mutual-funds/category/best-flexi-cap-mutual-fund",
        "Best Flexi Cap Mutual Funds", "", "", "category"),
    SourceSpec("sip-calculator",
        "https://groww.in/calculators/sip-calculator",
        "SIP Calculator", "", "", "tool"),
    SourceSpec("riskometer",
        "https://groww.in/p/riskometer",
        "Riskometer (SEBI)", "", "", "regulatory"),
    SourceSpec("what-is-the-difference-between-elss-and-sip",
        "https://groww.in/questions/what-is-the-difference-between-elss-and-sip",
        "What is the difference between ELSS and SIP?", "", "", "education"),
    SourceSpec("tax-on-mutual-funds",
        "https://groww.in/blog/tax-on-mutual-funds/",
        "Tax on mutual funds", "", "", "education"),
)

SOURCES_CSV = ROOT / "data" / "sources.csv"

# Schemes the assistant is allowed to give scheme-level facts about.
PRIMARY_SCHEMES = tuple(s for s in SOURCES if s.page_role == "primary")
```

Also implement in `sources.py`:
- `load_sources_csv() -> list[dict]` — read `sources.csv` (empty list if missing)
- `valid_urls() -> set[str]` — URLs with `http_status == 200`
- `write_sources_csv(rows: list[dict]) -> None` — writes the header
  `slug,url,title,scheme,category,page_role,fetched_at,http_status,char_count`

### Verification

```powershell
python -c "from mf_rag.sources import SOURCES, PRIMARY_SCHEMES; print(len(SOURCES)); print(len(PRIMARY_SCHEMES)); [print(s.slug, s.page_role) for s in SOURCES]"
```
Expect: `15`, `5`, and 15 slug/role lines.

### Exit criteria

- [ ] `len(SOURCES) == 15`, `len(PRIMARY_SCHEMES) == 5`
- [ ] All 5 primary slugs match PRD §5.1 exactly
- [ ] Every scheme is a Direct–Growth plan
- [ ] `settings.embed_model` and `settings.embed_dim` read from config, not literals elsewhere

**Commit:** `feat: config module and 15-URL source registry`

---

## Phase 2 — Stage 1: Loader / Ingestion

**Goal.** `loaders.py` turns 15 URLs into clean markdown + `sources.csv`.

**Why now.** The corpus must exist before anything can be chunked, embedded, or retrieved.

### Contract to implement

```python
"""STAGE 1 — Loading: fetch, extract, clean."""

@dataclass
class FetchResult:
    spec: SourceSpec
    ok: bool
    http_status: int
    html: str
    from_cache: bool

def fetch_page(spec: SourceSpec, refresh: bool = False) -> FetchResult: ...
def extract_main_text(html: str, url: str) -> str: ...
def clean_text(text: str) -> str: ...
def process_page(spec: SourceSpec, html: str) -> str: ...   # returns markdown
def ingest_all(refresh: bool = False) -> IngestSummary: ...
```

### Implementation requirements

**1. Cache-first fetch.** If `data/raw/<slug>.html` exists and `refresh=False`, reuse it
(ADR-7 — makes re-runs instant and offline-reproducible).

**2. Retry with backoff.** `http_retries=3`, sleeps `1s, 2s, 4s`. Use
`httpx.Client(timeout=..., headers={"User-Agent": settings.user_agent})`, `follow_redirects=True`.

**3. Extraction order.**
- First: `trafilatura.extract(html, url=url, include_tables=True)` —
  `include_tables=True` is **load-bearing**: Groww renders expense ratio and exit load in
  tables, which naive extraction discards.
- Fallback: BeautifulSoup — decompose `nav, header, footer, aside, script, style, noscript`,
  then take the richest remaining container's text.

**4. `clean_text()`.**
- Normalise unicode dashes/quotes to ASCII
- Strip zero-width characters (`\u200b`–`\u200f`, `\ufeff`)
- Collapse runs of whitespace/blank lines
- **Remove boilerplate lines** (case-insensitive substring match), defined once as a module
  constant `BOILERPLATE_PATTERNS`:
  ```
  "groww app download", "download the groww app", "trusted by", "referral",
  "cookie", "privacy policy", "terms of use", "track your investment",
  "app store", "google play", "open in app", "log in", "sign up",
  "disclaimer: mutual fund investments are subject to market risks",
  "mutual fund investments are subject to market risks",
  ```
- Preserve numeric values and `%` exactly — never rewrite, reformat, or round a number.

**5. Output per page.** `data/processed/<slug>.md` with YAML front-matter:
```yaml
---
title: <spec.title>
url: <spec.url>
scheme: <spec.scheme>
category: <spec.category>
page_role: <spec.page_role>
fetched_at: <YYYY-MM-DD>
---
<body text>
```

**6. `data/sources.csv`.** Write one row per source with
`fetched_at` (snapshot date, ISO), `http_status`, `char_count`.

**7. Failure is data, not an exception** (FR-1). A failed page gets its row with the real
`http_status`; processing continues. If **zero** pages succeed, raise with a clear message.

**8. Rate limiting.** `await`-style `time.sleep(settings.fetch_delay_seconds)` between network
requests (not between cache hits).

**9. `IngestSummary`.** Dataclass with `requested, fetched, cached, failed, low_text` and a
`__str__` producing the FR-15 stage line:
```
STAGE 1 · pages requested 15 · fetched 12 · cached 3 · failed 0 · low_text 2
```

### Verification

```powershell
python -m mf_rag.cli ingest
Get-Content data\sources.csv | Select-Object -First 4
(Get-ChildItem data\processed\*.md).Count   # expect 15
```

Then **manually open 2 processed files** and confirm the text reads as scheme content — not
navigation, not a cookie banner, and with tables intact.

### Exit criteria

- [ ] 15 files in `data/processed/`, 15 rows in `data/sources.csv`
- [ ] `char_count` recorded for every row
- [ ] No nav/footer/app-download text in any processed file
- [ ] `STAGE 1` stage line printed
- [ ] Re-running is fast (all cache hits) and produces identical `sources.csv`

**Commit:** `feat: stage 1 loader — fetch, extract, clean, sources.csv`

---

## Phase 3 — ⛔ GATE 1: Data Inspection

**Goal.** Confirm the corpus actually contains the facts the demo is graded on.

**Why this gate exists.** Groww pages are client-rendered (risk **F4** in architecture §14).
If expense ratio and riskometer are absent from the static HTML, **no amount of correct
chunking or retrieval will recover them.** This must be discovered now, not at M6.

### Do this

1. Print a `char_count` summary per page, sorted ascending:
   ```powershell
   Import-Csv data\sources.csv | Sort-Object {[int]$_.char_count} |
     Select-Object slug,page_role,char_count,http_status | Format-Table -AutoSize
   ```
2. Flag any page with `char_count < 500` as `low_text`.
3. For each of the 5 primary schemes, search the processed text for the graded facts:

   ```powershell
   Select-String -Path data\processed\*.md -Pattern 'expense ratio','exit load',
     'minimum (sip|lump)','lock-in','riskometer','benchmark','3 year|three year'
   ```

4. Build a coverage table and report it:

   | Scheme | Expense ratio | Exit load | Min SIP | Lock-in | Riskometer | Benchmark |
   | --- | --- | --- | --- | --- | --- | --- |
   | HDFC Large Cap | present/absent | … | … | n/a | … | … |
   | HDFC ELSS | … | … | … | … | … | … |

### Decision table

| Finding | Action |
| --- | --- |
| All primary facts present | Continue to Phase 4 |
| Some facts missing, secondary page has them | Keep going; the bot links the page and says the field is missing (PRD §10.7) |
| **Expense ratio or riskometer missing from all 5 schemes** | **STOP. Report to the user.** Options: (a) find the public factsheet/SID PDF for those fields, (b) add a richer extractor, (c) narrow the demo Q&A to fields actually present. **Do not proceed and do not invent values.** |

### Exit criteria

- [ ] Coverage table produced from real fetched text
- [ ] Every `low_text` page identified and explained
- [ ] User informed of any missing graded fact **before** Phase 4 starts
- [ ] Decision recorded in the Phase 13 README ("corpus snapshot notes")

**Commit:** `docs: corpus coverage report from gate 1`

---

## Phase 4 — Stage 2: Chunking (Candidates A and B)

**Goal.** `chunkers.py` implements both chunking candidates behind one protocol.

**Why both.** PRD §7 requires inspecting the data before choosing. Phase 5 measures them;
this phase only implements.

### Contract to implement

```python
"""STAGE 2 — Chunking."""
from dataclasses import dataclass, field
from typing import Protocol, Literal

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    title: str
    scheme: str
    category: str
    page_role: str
    section: str
    char_len: int
    fetched_at: str
    also_seen_at: list[str] = field(default_factory=list)   # deduped duplicates

@dataclass
class ProcessedDoc:
    slug: str; url: str; title: str; scheme: str; category: str
    page_role: str; fetched_at: str; text: str

class Chunker(Protocol):
    name: str
    def split(self, doc: ProcessedDoc) -> list[Chunk]: ...
```

**Chunker A — `RecursiveChunker`** (name `"a"`):
`RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100,
separators=["\n## ", "\n\n", "\n", ". ", " "])`

**Chunker B — `StructureAwareChunker`** (name `"b"`, provisional default):
1. Parse section headings from the markdown (`^#{1,4}\s`, and `**Bold**` pseudo-headings) and
   FAQ question boundaries (lines ending in `?`).
2. Emit each section as one chunk if `len <= 700`; otherwise recurse into Chunker A **within
   that section only**.
3. Prefix every chunk with `"{scheme} — {section}\n"` (empty scheme → use `title`).
4. **Numeric-block protection:** if a chunk starts with a bare number or `%`
   (`^[\d(]|\d+(\.\d+)?\s?%`), rejoin it to the previous chunk. A percentage must never be
   orphaned from its label.
5. Post-filters: drop `< 80` chars; keep `80–200` only if it is a standalone FAQ answer;
   clamp final chunks to `chunk_size + 200` to tolerate a prefix.

**Shared post-processing (both candidates):**
- Dedupe by `sha256(normalized_text)` (lowercase, collapse whitespace). Keep one chunk; union
  the duplicate URLs into `also_seen_at` so citations survive (P1).
- `chunk_id = f"{slug}#{index:04d}"`, index assigned **after** dedupe, per page.
- **Every chunk must have a `source_url`.** Assert it; raise if violated (P1).

**Outputs:**
- `data/chunks/chunks.jsonl` — one JSON object per line, all `Chunk` fields
- `data/chunks/chunks.txt` — human-readable mirror (deliverable D8). Format:
  ```
  ─────────────────────────────────────────────────────────
  chunk_id : hdfc-large-cap-fund-direct-growth#0007
  scheme   : HDFC Large Cap Fund
  section  : Fees and charges
  source   : https://groww.in/...
  chars    : 612
  ─────────────────────────────────────────────────────────
  <text>
  ```

**`build_chunks(strategy: Literal["a","b"]) -> list[Chunk]`** — dispatches on
`settings.chunk_strategy`, reads `data/processed/*.md`, writes both outputs, and logs:
```
STAGE 2 · strategy=b · chunks 187 · dropped_short 24 · deduped 9 · mean_len 648
```

### Verification

```powershell
python -m mf_rag.cli chunk --strategy a
python -m mf_rag.cli chunk --strategy b
Get-Content data\chunks\chunks.jsonl | Measure-Object -Line
Get-Content data\chunks\chunks.txt | Select-Object -First 30
```

**Human check (required, P5):** open `chunks.txt` and confirm — (a) chunks carry scheme and
section context, (b) no chunk is pure navigation, (c) no fee table is split mid-way,
(d) percentages sit with their labels.

### Exit criteria

- [ ] Both strategies run and write both output files
- [ ] Every chunk has a non-empty `source_url` (assert this in code and prove it)
- [ ] `chunks.txt` inspected and looks correct by all four human checks
- [ ] Stage log line printed with real counts

**Commit:** `feat: stage 2 chunking — candidate A and B with dedupe and numeric protection`

---

## Phase 5 — ⛔ GATE 2: Chunking Evaluation & Decision

**Goal.** Measure both candidates on a 20-question labelled set, pick a winner with evidence,
calibrate `MIN_SCORE`, and set `chunk_strategy` in config.

**Why this gate exists.** PRD §7.2 pre-commits to a decision rule so the demo has an answer
either way. Shipping a chunking strategy without numbers would fail acceptance criterion 11.

### Step 1 — Build the eval set: `eval/chunking_eval.json`

20 rows, **4 per primary scheme**, labelled **by reading the fetched corpus** (never from
memory — Open Question #4):

```json
[
  {
    "id": "q01",
    "query": "expense ratio of HDFC Large Cap Fund Direct Growth",
    "expect_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "key_fact": "<exact phrase copied from data/processed/…>",
    "scheme": "HDFC Large Cap Fund",
    "fact_type": "numeric"
  }
]
```

Coverage rule — 4 per scheme spanning the taxonomy:
1. **numeric** fact (expense ratio, exit load, minimum SIP, minimum lump sum)
2. **attribute** fact (benchmark, riskometer, plan type, lock-in)
3. **concept/education** fact (from `education`/`regulatory` pages)
4. **procedural** fact (documents, how-to)

### Step 2 — Implement `mf_rag/evalkit.py`

> Note: this phase needs embeddings, so run it **after** Phase 6, or implement `evalkit` here
> and execute the evaluation at the end of Phase 6. Recommended: implement `evalkit.py` now
> against a `store` interface, execute in Phase 6.

```python
"""Evaluation — Recall@k, citation accuracy, orphan rate for chunking A vs B."""
@dataclass
class EvalResult:
    strategy: str
    recall_at_5: float
    citation_accuracy: float
    orphan_rate: float
    per_query: list[dict]

def load_eval_set(path: Path) -> list[dict]: ...
def score_strategy(strategy: Literal["a", "b"], eval_set: list[dict], k: int = 5) -> EvalResult: ...
def compare_strategies(results: list[EvalResult]) -> str: ...   # markdown table
```

**Metric definitions — implement exactly:**
- **Recall@5** — fraction of queries where `key_fact` (normalised, whitespace-collapsed) is a
  substring of at least one of the top-5 retrieved chunks.
- **Citation accuracy** — fraction of queries where the **top-1** chunk's `source_url` equals
  `expect_url`.
- **Orphan rate** — fraction of chunks whose text does **not** begin with the
  `"{scheme} — {section}"` prefix (Candidate A should score high, B ~0).

Each strategy must be indexed into its **own** Chroma collection
(`mf_facts_hdfc_eval_a`, `mf_facts_hdfc_eval_b`) so the comparison is not contaminated.

### Step 3 — Run and decide

```powershell
python -m mf_rag.cli eval
```

Write `docs/eval_report.md` containing: both scorecards as a markdown table, the chosen
strategy, the reasoning in one paragraph, **and the score distribution** used to calibrate
`MIN_SCORE`.

### Step 4 — Calibrate `MIN_SCORE` (§9.4 of architecture.md)

| Observed | Action |
| --- | --- |
| In-scope > 0.5, out-of-scope < 0.15 | Keep `0.25` |
| In-scope questions being dropped | Lower to `0.20` |
| Out-of-scope scoring > 0.30 | Raise to `0.35` |

Also test one out-of-scope probe — *"What is the weather in Delhi?"* — and confirm it falls
below the threshold. **Write the chosen value into `config.py` and set `chunk_strategy` to the
winner.**

### Exit criteria

- [ ] `eval/chunking_eval.json` has 20 rows; every `key_fact` copied from fetched text
- [ ] Both strategies scored on Recall@5, citation accuracy, orphan rate
- [ ] `docs/eval_report.md` written with both scorecards
- [ ] `chunk_strategy` and `min_score` set in `config.py` from evidence
- [ ] Out-of-scope probe correctly falls below threshold
- [ ] **User informed of the chosen strategy and the numbers**

**Commit:** `feat: chunking eval harness — decision + min_score calibration`

---

## Phase 6 — Stages 3 & 4: Embedding + Vector Store

**Goal.** `embedder.py` and `store.py`. Chunks become searchable vectors on disk.

**Why now.** Phases 8–9 both depend on retrieval working; this is the last gate before the
answering path can be built and tested.

### `mf_rag/embedder.py`

```python
"""STAGE 3 — Embedding: the only path from text to a 384-dim vector."""
from sentence_transformers import SentenceTransformer
from .config import settings

_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy singleton — ~80MB, ~2s load. Only the first query pays for it (NFR-2)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(settings.embed_model)
    return _MODEL


def embed(texts: list[str]) -> list[list[float]]:
    """Embed chunks AND queries. Normalised so cosine distance == dot product."""
    vectors = get_model().encode(
        texts, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    assert vectors.shape[1] == settings.embed_dim, "embedding dim mismatch"
    return vectors.tolist()
```

> P6: this is the **only** embedding entry point. Never add a second path — model drift
> between corpus and query is the classic silent RAG bug.

### `mf_rag/store.py`

```python
"""STAGE 4 — Vector Store: ChromaDB, persisted, idempotent."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from .config import settings

_client = None

def get_client() -> chromadb.ClientAPI:
    """Persistent client — the index survives restarts, so ingest runs once (P10)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client

def get_collection(name: str | None = None): ...
def upsert_chunks(chunks: list[Chunk], rebuild: bool = False) -> int: ...
def search(query_vec: list[float], top_k: int, collection: str | None = None) -> list[ScoredChunk]: ...
def collection_size(collection: str | None = None) -> int: ...
```

**Requirements:**
- Collection metadata must declare `{"hnsw:space": "cosine"}` — changing distance later would
  silently invalidate every stored distance (ADR-8).
- `upsert_chunks` uses `collection.upsert(ids=[c.chunk_id], documents=[c.text], metadatas=[...])`.
  `chunk_id` is deterministic, so re-embedding **replaces** rather than duplicates (P10, FR-3).
- `rebuild=True` drops and recreates the collection.
- Metadata per chunk: `chunk_id, source_url, title, scheme, category, page_role, section,
  char_len, fetched_at`. **`text` must NOT be in metadata** — Chroma stores it as `documents`;
  duplicating it bloats the metadata payload (§7.4).
- Log lines:
  ```
  STAGE 3 · model=all-MiniLM-L6-v2 · dim=384 · encoded 187 in 3.1s
  STAGE 4 · collection=mf_facts_hdfc · upserted 187 · total 187
  ```

### If Phase 5 was deferred

Run `python -m mf_rag.cli eval` now that embeddings work, produce
`docs/eval_report.md`, and set `chunk_strategy` + `min_score` in `config.py` before continuing.

### Verification

```powershell
python -m mf_rag.cli embed
python -m mf_rag.cli embed          # run twice — total must be identical (idempotent)
```

### Exit criteria

- [ ] `STAGE 3` and `STAGE 4` lines printed with real counts
- [ ] Second `embed` run shows the same `total` (proves idempotency)
- [ ] `chroma/` exists on disk and persists across process restarts
- [ ] Eval executed and `docs/eval_report.md` written (if deferred from Phase 5)
- [ ] `chunk_strategy` and `min_score` reflect the eval outcome

**Commit:** `feat: stages 3-4 — shared embedder and persistent Chroma store`

---

## Phase 7 — Stage 5a: Guards

**Goal.** `guards.py` — the compliance layer. This is the component that makes the
facts-only claim true.

**Why built before the retriever.** Guards must be verifiably upstream of generation (P2). A
reviewer can confirm this by reading import lines: `answerer.py` never imports `guards`,
`guards` imports nothing downstream.

### Contract to implement

```python
"""STAGE 5a — Guards: PII scrub and intent refusal. Leaf module — imports nothing downstream."""
from typing import Literal
from dataclasses import dataclass, field

GuardKind = Literal["pii", "advice", "returns", "comparative"]

@dataclass
class GuardVerdict:
    kind: GuardKind
    message: str
    link: str | None = None
    redacted_question: str = ""     # safe to log — PII already removed (P9)

def check_guards(question: str) -> GuardVerdict | None:
    """Return a verdict to short-circuit, or None to proceed to retrieval."""
```

### Order of operations — this ordering is load-bearing

```
1. PII scrub     → match?  privacy notice; DO NOT LOG the raw text; return
2. advice        → match?  refusal + educational link; return
3. returns       → match?  refusal + factsheet link; return
4. comparative   → match?  refusal + factsheet link; return
5. → None (proceed)
```

PII first, deliberately: *"my PAN is ABCDE1234F, should I buy?"* must return the **privacy
notice**, and the PAN must never reach `eval/queries.jsonl` (§10.1).

### PII patterns (module constant `PII_PATTERNS`)

| Name | Regex |
| --- | --- |
| PAN | `\b[A-Z]{5}\d{4}[A-Z]\b` |
| Aadhaar | `\b\d{4}\s?\d{4}\s?\d{4}\b` |
| Email | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` |
| Phone | `\b(?:\+91[\-\s]?)?[6-9]\d{9}\b` |
| OTP | 6 digits when adjacent to `otp\|code\|verify` |
| Account no. | 8–18 digits when adjacent to `account\|a/c\|acc` |

**Context-sensitive patterns must be implemented as lookaround**, not bare digit runs. A bare
`\d{8,18}` will match scheme codes and section numbers; over-eager PII matching makes the bot
look broken in a demo about mutual funds.

### Intent patterns (`ADVICE_PATTERNS`, `RETURNS_PATTERNS`, `COMPARATIVE_PATTERNS`)

- **advice:** `should i`, `is it good for me`, `recommend`, `best fund for`, `safe to`,
  `allocate`, `portfolio`, `suitable`, `worth buying`, `which should i`
- **returns:** `returns`, `performance`, `cagr`, `best performing`, `% gain`, `how much did it give`
- **comparative:** `better than`, `vs`, `versus`, `compare`, `ranking`, `top 10`

`comparative` is folded in because *"is HDFC Flexi Cap better than HDFC Large Cap?"* is a
returns question in a comparison costume.

### Verbatim copy — these strings must match `architecture.md` §10.4 exactly

**Privacy notice:**
> Please don't share personal details. This assistant doesn't accept or store PAN, Aadhaar
> numbers, account numbers, OTPs, email addresses, or phone numbers. Your question was not
> saved. For account-specific queries, use the Groww app or your registered adviser.

**Advice/returns refusal:**
> I can share published facts about HDFC Mutual Fund schemes — fees, exit load, minimum SIP,
> lock-in, riskometer, benchmark and how to get documents. I can't give investment advice or
> compare returns. For that, please read the official factsheet:
> `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`

### Also implement

```python
PII_REDACTION = "[redacted]"
def scrub_for_log(question: str) -> str:   # replaces PII spans; the ONLY thing logged (P9)
```

### Verification

```powershell
python -c "from mf_rag.guards import check_guards; \
  [print(repr(q), '->', (check_guards(q).kind if check_guards(q) else 'proceed')) \
   for q in ['my PAN is ABCDE1234F','call me on 9876543210','Should I buy HDFC Small Cap?', \
             'which fund has best returns','is Flexi Cap better than Large Cap', \
             'expense ratio of HDFC Large Cap?']]"
```

### Exit criteria

- [ ] All 6 PII classes detected; each returns the privacy notice
- [ ] A raw PAN never appears in any log written during the above run
- [ ] `should I buy` → `advice`; `best returns` → `returns`; `better than` → `comparative`
- [ ] `expense ratio of HDFC Large Cap?` → `None` (proceeds to retrieval)
- [ ] Verbatim copy matches §10.4
- [ ] No imports from `retriever`, `store`, or `answerer`

**Commit:** `feat: stage 5a guards — PII scrub and advice/returns refusal`

---

## Phase 8 — Stage 5b: Retriever

**Goal.** `retriever.py` — question → vector → top-k → MMR → threshold → prompt context.

### Contract to implement

```python
"""STAGE 5b — Retrieval."""
@dataclass
class ScoredChunk:
    chunk_id: str; text: str; source_url: str; scheme: str
    section: str; page_role: str; score: float

def retrieve(question: str, top_k: int | None = None,
             min_score: float | None = None) -> list[ScoredChunk]: ...
def mmr_rerank(candidates: list[ScoredChunk], lambda_: float, top_k: int) -> list[ScoredChunk]: ...
def build_context(chunks: list[ScoredChunk]) -> str: ...
```

### Implementation requirements

1. **Guard first.** `retrieve` assumes guards already passed (the caller in `cli.py`/`app.py`
   runs `check_guards` first). Keep this visible in the call order.
2. **Embed the query with `embedder.embed([question])`** — same model, same normalisation (P6).
3. **Cosine top-k** from `store.search(query_vec, top_k=settings.top_k)`.
   `score = 1 − distance`, so the score is interpretable against `MIN_SCORE` (§9.4).
4. **MMR re-rank** (λ = `settings.mmr_lambda` = 0.7), selecting 4 from the 6 candidates so a
   question about exit load doesn't get three near-identical fee chunks:
   ```
   MMR = argmax_{c ∉ S}  λ·sim(c,q) − (1−λ)·max_{s ∈ S} sim(c,s)
   ```
5. **Threshold.** Drop chunks below `settings.min_score`. If **nothing** survives, return
   `[]` — the caller takes the "not in my sources" path (P4). Never return a weak chunk just to
   have something to say.
6. **`build_context`** — numbered blocks, the exact shape in architecture §11.2:
   ```
   [1] source_url: … | page_role: primary | scheme: … | section: …
       <chunk text>
   ```
   `page_role` **must** appear in the prompt text (P3 / ADR-10).
7. **Scheme boost is OFF by default.** Only enable with Phase 5 eval evidence of cross-scheme
   confusion (§9.5).
8. Log: `STAGE 5 · q="…" · top_k=6 · mmr=4 · kept 3 · best_score=0.61`

### Verification

Temporarily script (or add a `--show-chunks` flag to the CLI) and confirm:

```powershell
python -c "from mf_rag.retriever import retrieve; \
  [print(f'{c.score:.3f}  {c.scheme}  {c.section}') for c in retrieve('exit load on HDFC Equity Flexi Cap')]"
python -c "from mf_rag.retriever import retrieve; print(retrieve('What is the weather in Delhi?'))"
```

### Exit criteria

- [ ] A real corpus question returns 3–4 chunks with sensible scores
- [ ] Top chunk's section is genuinely relevant (e.g. `Exit load` for the exit-load query)
- [ ] The weather probe returns `[]`
- [ ] `build_context` output includes `page_role` for every block
- [ ] `STAGE 5` line printed

**Commit:** `feat: stage 5b retriever — cosine, MMR, threshold, context assembly`

---

## Phase 9 — Stage 6: Prompts + Answerer

**Goal.** `prompts.py` and `answerer.py` — the LLM call, the post-checks, and the extractive
fallback.

### `mf_rag/prompts.py`

```python
"""STAGE 6 — Prompt construction and wording."""

DISCLAIMER = (
    "Facts-only. No investment advice. Answers are generated from public HDFC "
    "Mutual Fund pages on Groww and may be outdated. Verify every number on the "
    "linked source page before acting. Mutual fund investments are subject to "
    "market risks; read all scheme-related documents carefully."
)

FACTSHEET_LINK = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
EDUCATION_LINK = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
AS_OF_TEMPLATE = "Last updated from sources: {date}"

def system_prompt() -> str: ...
def user_prompt(question: str, chunks: list[ScoredChunk], as_of: str) -> str: ...
def extractive_template(sentences: list[str], url: str, as_of: str) -> str: ...
```

**The system prompt must contain all 8 rules** from `architecture.md` §11.1. Two matter most
beyond the obvious:

- **Rule 6** — context blocks with `page_role: education` explain concepts only; never take a
  scheme number from them (P3).
- **Rule 7** — if a field is genuinely missing, *say it is missing and link the page*. Never
  guess a number. This is what pushes the model toward P4 instead of fabricating.

The user prompt must **restate the format instruction after the question** as well as in the
system prompt — recency matters in prompts, and the last thing the model reads should be the
format it must follow.

### `mf_rag/answerer.py`

```python
"""STAGE 6 — Answering: Groq call, post-checks, deterministic fallback."""
from typing import Literal
from dataclasses import dataclass

AnswerMode = Literal["llm", "extractive", "refusal", "not_in_sources", "pii_refusal"]

@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    as_of: str
    chunks: list[ScoredChunk]
    mode: AnswerMode
    guard: str | None = None
    disclaimer: str = prompts.DISCLAIMER

def answer_question(question: str, top_k: int | None = None) -> Answer: ...
def post_check(text: str, chunks: list[ScoredChunk]) -> Answer | None: ...
def extractive_fallback(chunks: list[ScoredChunk]) -> Answer: ...
def call_groq(messages: list[dict]) -> str | None: ...
```

### `answer_question` flow

```
1. verdict = check_guards(question)
   if verdict: return Answer(mode=refusal|pii_refusal, guard=verdict.kind, text=verdict.message)
2. chunks = retrieve(question, top_k)
   if not chunks: return Answer(mode="not_in_sources",
       text="I don't have that in my sources. I cover …" + coverage list from sources.csv)
3. messages = build_messages(question, chunks)
4. raw = call_groq(messages)
      None (no key / error / timeout / 429 after one retry) → extractive_fallback
5. answer = post_check(raw, chunks)
      None → extractive_fallback
6. append "Last updated from sources: <fetched_at date>"
7. return Answer
```

### `post_check` — implement all 7 checks **in this order**

| # | Check | Method | On failure |
| --- | --- | --- | --- |
| 1 | Citation present | URL regex over the answer | → `extractive_fallback` |
| 2 | Citation is real | `url ∈ valid_urls()` from `sources.csv` **and** `http_status == 200` | → `extractive_fallback` |
| 3 | ≤ 3 sentences | Split on `[.!?]` | → truncate to 3 |
| 4 | No advice language | blacklist: `you should`, `i recommend`, `buy this`, `good for`, `advisable`, `must invest` | → refusal |
| 5 | No performance claims | `%` **and** a return-context word (`returns`, `cagr`, `gained`, `yield`) in the **same sentence** | → refusal + factsheet link |
| 6 | Footer present | contains `Last updated from sources:` | → append |
| 7 | Disclaimer present | equals `prompts.DISCLAIMER` | → append |

**Check 5 must be sentence-scoped and conservative.** A bare `%` is never blocked — expense
ratio *is* a percentage and Q1 depends on it passing. Only a percentage co-occurring with a
return-context word is a claim, so `0.63% expense ratio` passes and `18% returns` does not.

**Ordering matters:** checks 1→2 first. An ungrounded citation is a trust failure that
invalidates everything else, so fall back rather than patch.

### `extractive_fallback` (ADR-2, principle P8)

No LLM, no network, always works. Pick the sentences from the top chunk with highest
query-term overlap, trim to `max_sentences`, append `Source: <url>` and the as-of footer.
**This is what guarantees the demo completes with no `GROQ_API_KEY`.**

### `call_groq`

```python
from groq import Groq
client = Groq(api_key=settings.groq_api_key)
resp = client.chat.completions.create(
    model=settings.llm_model,
    messages=messages,
    temperature=settings.llm_temperature,   # 0.0 (P7)
    max_tokens=settings.llm_max_tokens,
)
```
Catch all exceptions, log a one-line notice, return `None`. Never let a Groq failure become a
traceback in a live demo.

### Verification

```powershell
python -m mf_rag.cli query "expense ratio of HDFC Large Cap Fund Direct Growth"
python -m mf_rag.cli query "Is HDFC ELSS Tax Saver Fund locked in?"
python -m mf_rag.cli query "Should I buy HDFC Small Cap Fund?"
python -m mf_rag.cli query "What is the weather in Delhi?"
python -m mf_rag.cli query "my PAN is ABCDE1234F"
```

Then unset `GROQ_API_KEY` and re-run query #1 — the `mode` must become `extractive` and the
answer must still carry a valid citation.

### Exit criteria

- [ ] All five probes above return the correct `mode`
- [ ] Factual answers are ≤ 3 sentences, end with the as-of footer, and carry a real citation
- [ ] Advice/returns/PII probes never call the LLM
- [ ] Weather probe returns "not in my sources" + coverage list
- [ ] With no API key, `extractive` mode still produces a cited, ≤ 3-sentence answer
- [ ] Check 5 allows an expense-ratio percentage and blocks a returns percentage

**Commit:** `feat: stage 6 — prompts, groq answering, 7 post-checks, extractive fallback`

---

## Phase 10 — CLI Wiring

**Goal.** `cli.py` exposes all seven commands and prints stage counts (FR-15).

### Commands

| Command | Runs | Options |
| --- | --- | --- |
| `ingest` | Stage 1 | `--refresh` to bypass cache |
| `chunk` | Stage 2 | `--strategy a\|b` |
| `embed` | Stages 3–4 | `--rebuild` |
| `query "<question>"` | Stages 5–6 | `--show-chunks`, `--top-k`, `--min-score` |
| `eval` | Stages 2 & 5 | `--strategy` |
| `app` | Stage 7 | `--port` |
| `all` | Stages 1–4 | `--refresh`, `--strategy`, `--rebuild` |

**Requirements:**
- `all` is the acceptance command (PRD §11.1). It must print one line per stage with counts.
- `query` **must** run `check_guards` before `retrieve` (P2) and print the `Answer.mode`.
- Clear one-line errors, never a raw traceback, for: index not built ("run
  `python -m mf_rag.cli embed` first"), ingest never run, missing `GROQ_API_KEY` (notice, not
  error — the fallback handles it).
- Every module gets a `main()`; `python -m mf_rag.cli <cmd>` works.

### Verification

```powershell
python -m mf_rag.cli all
python -m mf_rag.cli query "benchmark of HDFC Equity Flexi Cap" --show-chunks
```

Expected stage output (counts will differ; the shape must not):
```
STAGE 1 · pages requested 15 · fetched 15 · cached 0 · failed 0 · low_text 0
STAGE 2 · strategy=b · chunks 187 · dropped_short 24 · deduped 9 · mean_len 648
STAGE 3 · model=all-MiniLM-L6-v2 · dim=384 · encoded 187 in 3.1s
STAGE 4 · collection=mf_facts_hdfc · upserted 187 · total 187
```

### Exit criteria

- [ ] All 7 commands run
- [ ] `all` prints all 4 stage lines with real counts
- [ ] `query --show-chunks` shows chunks + scores + the final answer
- [ ] No traceback on any error path

**Commit:** `feat: cli — ingest/chunk/embed/query/eval/app/all with stage logging`

---

## Phase 11 — Stage 7: Streamlit UI

**Goal.** `app.py` — the demo surface, which makes the pipeline visible (P5, FR-13/FR-14).

### Required layout

```
┌───────────────────────────────────────────────────────────────┐
│  MF Facts Assistant — HDFC Mutual Funds                       │
│  Facts-only. No investment advice.                           │
│  Corpus: 15 public pages · 5 schemes · snapshot 2026-09-28    │
├───────────────────────────────────────────────────────────────┤
│  Try one of these:                                            │
│   • Expense ratio of HDFC Large Cap – Direct Growth?         │
│   • Is HDFC ELSS Tax Saver Fund locked in?                    │
│   • Should I buy HDFC Small Cap Fund?                        │
├───────────────────────────────────────────────────────────────┤
│  [ transcript ]                                               │
│   ▸ answer (≤3 sentences)                                    │
│   ▸ Source: [clickable URL]                                  │
│   ▸ Last updated from sources: 2026-09-28                    │
│   ▸ ▾ Retrieved context (3 chunks, with scores)   ← FR-14     │
├───────────────────────────────────────────────────────────────┤
│  [________________________] [Ask]                             │
│  ⚠ Facts-only. No investment advice.                         │
│    Don't enter PAN, Aadhaar, account numbers, OTPs, email,    │
│    phone.                                                     │
└───────────────────────────────────────────────────────────────┘
```

### Requirements

1. `st.set_page_config` first. Read the snapshot date and corpus counts from `sources.csv` —
   don't hardcode them.
2. `st.chat_input` + `st.chat_message`. The 3 example questions are clickable buttons that
   populate the input.
3. **Three example questions**, exactly as above (Q1 factual, Q4 factual/yes-no, Q9 refusal) —
   they demo a fact, a fact, and a refusal.
4. Per answer render: text, `Source:` as a real markdown link, as-of footer, and
   `st.expander("Retrieved context")` showing each chunk's scheme, section, score, and a
   truncated preview. **Retrieval must be demonstrable** (FR-14).
5. Colour the guard states distinctly: refusal, PII notice, and not-in-sources should be
   visually obvious.
6. `prompts.DISCLAIMER` in the sidebar/footer, verbatim, plus the PII warning.
7. `st.session_state` only. **No chat persistence to disk** (PRD §3).
8. Set `page_config` layout wide enough that the expander is usable.

### Verification

```powershell
streamlit run app.py
```
Click all three example questions, then type: `what is the weather in delhi` and
`my email is a@b.com`. Confirm the four distinct visual states.

### Exit criteria

- [ ] Welcome line, 3 clickable examples, disclaimer all present
- [ ] Answers show a clickable citation, the as-of footer, and an expandable chunk view
- [ ] Refusal / PII / not-in-sources states render distinctly
- [ ] Nothing persisted to disk

**Commit:** `feat: stage 7 — streamlit chat UI with citations and chunk disclosure`

---

## Phase 12 — Tests

**Goal.** Unit tests that make the compliance claims verifiable rather than asserted.

**Strategy: three layers, cheapest first** (architecture §16). Build L1 and L3 now; L2 already
exists as `evalkit.py` from Phase 5.

### `tests/test_guards.py` — the compliance layer (highest priority)

- Every PII class → `pii` verdict: PAN, Aadhaar, email, phone, OTP-in-context,
  account-number-in-context
- `should I buy` → `advice`; `best returns` → `returns`; `better than` → `comparative`
- **PII precedence:** `"my PAN is ABCDE1234F, should I buy HDFC Small Cap?"` → `pii`, not
  `advice`
- `scrub_for_log` removes the PAN
- Factual question → `None`
- **False-positive check:** a query containing a scheme code or a long number that is *not*
  labelled account/OTP context must **not** trigger the PII guard

### `tests/test_loaders.py`

- `clean_text` strips every `BOILERPLATE_PATTERNS` entry
- `clean_text` preserves `0.63%` and `1.25` exactly (no rounding, no reformatting)
- Zero-width characters removed
- Cached fetch does not hit the network (monkeypatch `httpx` to raise if called)

### `tests/test_chunkers.py`

- Both chunkers respect `chunk_size` within the documented tolerance
- No chunk is empty or whitespace-only
- **Every chunk has a non-empty `source_url`** (P1)
- `StructureAwareChunker` prefixes chunks with `scheme — section`; orphan rate ≈ 0
- A percentage never begins a chunk (numeric-block protection)
- Dedupe unions `also_seen_at`

### `tests/test_answerer.py` — the post-checks

- Check 1: no URL in the answer → fallback
- Check 2: a plausible-but-invented URL (e.g. `https://groww.in/mutual-funds/does-not-exist`)
  → fallback
- Check 3: 5-sentence answer → truncated to 3
- Check 4: "You should buy this" → refusal
- Check 5: `0.63% expense ratio` **passes**; `18% annual returns` **fails**
- Check 6/7: footer and disclaimer are appended
- `extractive_fallback` works with no API key and produces a valid citation

### `tests/test_sources.py`

- `len(SOURCES) == 15`, `len(PRIMARY_SCHEMES) == 5`
- All 5 primary schemes are Direct–Growth
- Every slug is unique and URL-safe

### Verification

```powershell
pytest -q
```

### Exit criteria

- [ ] All tests pass
- [ ] `test_guards.py` covers all 6 PII classes **and** the precedence case
- [ ] `test_answerer.py` proves the invented-URL case is caught
- [ ] `test_chunkers.py` asserts the `source_url` invariant on every chunk
- [ ] A PII false-positive test exists

**Commit:** `test: unit coverage for guards, cleaning, chunking, and post-checks`

---

## Phase 13 — Deliverables & Acceptance Pass

**Goal.** Produce every deliverable and prove acceptance criterion by acceptance criterion.

### Files to write

| File | Content |
| --- | --- |
| `README.md` | Setup steps, scope (AMC + 5 schemes), architecture summary, **chunking rationale with the Phase 5 scorecard**, corpus snapshot notes from Gate 1, known limits |
| `docs/sources.md` | The 15-row source table mirroring `data/sources.csv` (Appendix B of the PRD) |
| `docs/sample_qa.md` | **D5 — 5–10 queries generated by the real build**, each with answer + link + as-of footer |
| `docs/disclaimer.md` | The disclaimer snippet verbatim (D6) |
| `docs/eval_report.md` | Both chunking scorecards + chosen strategy + `MIN_SCORE` calibration (from Phase 5) |
| `docs/demo_script.md` | The 3-minute run-of-show, timed |

> **Do not hand-write `docs/sample_qa.md`.** Acceptance criterion 8 requires rows produced by
> the actual build. Generate them by piping the CLI over the canonical queries and pasting the
> real output. If a bot answers a fact incorrectly, fix the pipeline — do not edit the answer.

### Acceptance pass (PRD §11)

Run all 12 criteria and record pass/fail with evidence.

```powershell
# 1 — clean build
python -m mf_rag.cli all
# 2 — the six graded fact queries
foreach ($q in @("expense ratio of HDFC Large Cap Fund Direct Growth",
                 "exit load on HDFC Equity Flexi Cap Fund",
                 "minimum SIP for HDFC Small Cap Fund",
                 "is HDFC ELSS Tax Saver Fund locked in",
                 "riskometer level of HDFC Balanced Advantage Fund",
                 "benchmark of HDFC Equity Flexi Cap Fund")) {
  python -m mf_rag.cli query $q --show-chunks }
# 3–6 — refusals
python -m mf_rag.cli query "Should I buy HDFC Small Cap Fund?"
python -m mf_rag.cli query "Which HDFC fund has given the best returns?"
# 7 — PII, then grep the logs
python -m mf_rag.cli query "my PAN is ABCDE1234F"
Select-String -Path eval\queries.jsonl -Pattern 'ABCDE1234F|@[a-z]+\.'   # expect NO match
# 8 — every answer ≤ 3 sentences and carries the footer (inspect output)
# 9 — out of scope
python -m mf_rag.cli query "What is the weather in Delhi?"
# 11 — generate the sample Q&A from real output
# 12 — secret scan
git log --all -p | Select-String 'GROQ_API_KEY=' | Where-Object { $_ -notmatch 'your_key_here' }
```

### Exit criteria

- [ ] All 7 deliverable files exist and are accurate
- [ ] All 12 acceptance criteria pass, with evidence recorded
- [ ] `docs/sample_qa.md` rows came from real CLI output
- [ ] `docs/eval_report.md` contains both scorecards and the winner
- [ ] README documents scope, setup, chunking rationale, and known limits
- [ ] PII grep returns no match
- [ ] Secret scan is clean
- [ ] The whole pipeline runs end to end from a clean checkout

**Commit:** `docs: README, sources, sample QA, disclaimer, eval report, acceptance pass`

---

## Appendix A — Copy-Paste Agent Prompts

Use one prompt per phase. Paste the phase body as context.

**Phase 0**
> Set up a Python 3.11+ virtual environment for a RAG chatbot project. Create
> `requirements.txt` with the exact contents from the guide, `.env` / `.env.example` /
> `.gitignore` as specified, and the `mf_rag/` package skeleton with an empty `__init__.py`
> containing only a docstring and `__version__`. Verify the embedding model returns shape
> (1, 384) and that `.env` is not tracked by git. Stop and report if Python is missing.

**Phase 1**
> Implement `mf_rag/config.py` and `mf_rag/sources.py` exactly as specified in the guide.
> All 15 sources must be present with the exact slugs, URLs, and `page_role` values shown.
> Add the `load_sources_csv` / `valid_urls` / `write_sources_csv` helpers. Verify
> `len(SOURCES) == 15` and `len(PRIMARY_SCHEMES) == 5`.

**Phase 2**
> Implement `mf_rag/loaders.py` (STAGE 1) exactly as specified: cache-first fetch, 3 retries
> with backoff, `trafilatura` with `include_tables=True` and a BeautifulSoup fallback, the
> boilerplate-stripping `clean_text`, front-matter markdown output, and `data/sources.csv`
> with `char_count`. Failures must be recorded, not raised. Print the FR-15 stage line.

**Phase 3**
> Stop and run the Gate 1 data inspection from the guide. Print `char_count` per page sorted
> ascending, flag `low_text` pages, and build the coverage table for the six graded facts
> across the 5 primary schemes using `Select-String` over `data/processed/*.md`. **Report the
> table to me and do not proceed to Phase 4 until I confirm.** Do not invent or infer any
> missing value.

**Phase 4**
> Implement `mf_rag/chunkers.py` (STAGE 2) with both Candidate A (`RecursiveChunker`) and
> Candidate B (`StructureAwareChunker`) behind a `Chunker` protocol, plus the shared
> dedupe/numeric-protection/post-filter pipeline. Write `chunks.jsonl` and the human-readable
> `chunks.txt`. Every chunk must carry a `source_url` — assert it. Print the stage line.

**Phase 5**
> Implement `mf_rag/evalkit.py` and `eval/chunking_eval.json` (20 queries, 4 per primary
> scheme). **Every `key_fact` must be copied verbatim from `data/processed/` — do not write
> any value from memory.** Index each strategy into its own collection, compute Recall@5,
> citation accuracy, and orphan rate, write `docs/eval_report.md`, and recommend a
> `chunk_strategy` and `min_score` with the score-distribution evidence.

**Phase 6**
> Implement `mf_rag/embedder.py` and `mf_rag/store.py` (STAGES 3–4) as specified. The
> embedder must be a lazy singleton and the single embedding entry point with
> `normalize_embeddings=True` and a dim assertion. The store must use a persistent client,
> declare cosine space in collection metadata, upsert by `chunk_id` for idempotency, and keep
> `text` out of metadata. Print both stage lines. Prove idempotency by running twice.

**Phase 7**
> Implement `mf_rag/guards.py` (STAGE 5a) as a leaf module that imports nothing downstream.
> Order of operations is PII → advice → returns → comparative. Implement context-sensitive PII
> patterns with lookaround so bare digit runs don't false-positive. Use the verbatim copy
> strings from the guide. `scrub_for_log` is the only thing that may be persisted. Verify all
> 7 probe queries produce the expected verdicts.

**Phase 8**
> Implement `mf_rag/retriever.py` (STAGE 5b) as specified: embed the query with the shared
> embedder, cosine top-k, MMR re-rank at lambda 0.7 selecting 4 of 6, drop chunks below
> `min_score`, and return `[]` when nothing survives. `build_context` must emit numbered
> blocks including `page_role` for each. Print the stage line. Leave scheme boost off.

**Phase 9**
> Implement `mf_rag/prompts.py` and `mf_rag/answerer.py` (STAGE 6). The system prompt must
> contain all 8 rules verbatim from the guide, including rule 6 (education pages supply no
> scheme numbers) and rule 7 (say a missing field is missing). Implement all 7 post-checks in
> the specified order; check 5 must be sentence-scoped so an expense-ratio percentage passes
> and a returns percentage fails. Implement `extractive_fallback` so the demo works with no
> API key.

**Phase 10**
> Implement `mf_rag/cli.py` with the 7 commands and their options. `all` must print the 4
> stage lines. `query` must run guards before retrieval and print `Answer.mode`. Convert all
> error paths to clear one-line messages, never tracebacks.

**Phase 11**
> Implement `app.py` (STAGE 7) as a Streamlit chat UI per the layout in the guide. Read
> corpus counts and the snapshot date from `sources.csv`. Render the answer, a clickable
> citation, the as-of footer, and an expander showing retrieved chunks with scores. Show the
> 3 example questions as buttons. Use `st.session_state` only — no persistence.

**Phase 12**
> Write the unit tests listed in the guide under `tests/`. Priority order: guards (all 6 PII
> classes, precedence, and a false-positive case), answerer post-checks (including the
> invented-URL case and the check-5 percentage distinction), chunkers (the `source_url`
> invariant), loaders (boilerplate stripping and numeric preservation), sources (counts).

**Phase 13**
> Produce all deliverables: `README.md`, `docs/sources.md`, `docs/sample_qa.md`,
> `docs/disclaimer.md`, `docs/eval_report.md`, `docs/demo_script.md`. **Generate the sample Q&A
> by piping real CLI output — do not hand-write any answer.** Then run the full 12-criterion
> acceptance pass, including the PII log grep and the git secret scan, and report pass/fail
> with evidence for each.

---

## Appendix B — Definition of Done (whole build)

- [ ] `python -m mf_rag.cli all` builds from a clean checkout and prints 4 stage lines
- [ ] All 10 canonical queries return the correct `Answer.mode`
- [ ] All 6 graded fact queries return a correct fact with a working citation
- [ ] Every answer ≤ 3 sentences, with one citation and the as-of footer
- [ ] Advice / returns / comparative queries are refused **before** the LLM
- [ ] PII queries are refused and never written to disk
- [ ] Out-of-scope queries take the "not in my sources" path
- [ ] No emitted URL is absent from `sources.csv`
- [ ] `pytest -q` passes
- [ ] Both chunking candidates scored; the winner documented with numbers
- [ ] `docs/sample_qa.md` generated from the real build
- [ ] `GROQ_API_KEY` absent from the working tree and from git history
- [ ] The demo completes in under 3 minutes

## Appendix C — Quick Reference

```powershell
# Setup
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Build
python -m mf_rag.cli all                    # ingest → chunk → embed → store
python -m mf_rag.cli ingest --refresh       # force re-fetch
python -m mf_rag.cli chunk --strategy a     # candidate A
python -m mf_rag.cli embed --rebuild        # clean index

# Query
python -m mf_rag.cli query "expense ratio of HDFC Large Cap Fund Direct Growth" --show-chunks
python -m mf_rag.cli query "Should I buy HDFC Small Cap Fund?"

# Evaluate
python -m mf_rag.cli eval

# Run the UI
streamlit run app.py

# Test
pytest -q
```

| Config knob | Default | Set in |
| --- | --- | --- |
| `chunk_strategy` | `b` | Phase 5 (eval) |
| `min_score` | `0.25` | Phase 5 (calibration) |
| `llm_model` | `llama-3.3-70b-versatile` | Phase 9 / latency |
| `top_k` / `mmr_lambda` | `6` / `0.7` | Phase 8 |
