# Source list

Generated 2026-09-28 by `tools/make_sources.py` from `data/sources.csv` and `mf_rag/sources.py` (15 pages).

Scope is **one AMC, 5 Direct-Growth schemes**. The 5 primary pages are the only pages a scheme-specific answer may cite; the other 10 are retrieved for concept and process questions, and exist largely so the entity filter has something to reject.

Every URL the bot can emit appears in this table. That is checked mechanically by acceptance criterion 10, not by eye.

## The 5 in-scope schemes

| Scheme | URL | Last fetched |
| --- | --- | --- |
| HDFC Large Cap Fund —  | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | 2026-09-28 |
| HDFC Equity (Flexi Cap) Fund —  | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth | 2026-09-28 |
| HDFC ELSS Tax Saver Fund —  | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth | 2026-09-28 |
| HDFC Small Cap Fund —  | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth | 2026-09-28 |
| HDFC Balanced Advantage Fund —  | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth | 2026-09-28 |

## All 15 pages

| # | Role | Page | Slug |
| --- | --- | --- | --- |
| 1 | `primary` | [HDFC Large Cap Fund – Direct Growth](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth) | `hdfc-large-cap-fund-direct-growth` |
| 2 | `primary` | [HDFC Equity (Flexi Cap) Fund – Direct Growth](https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth) | `hdfc-equity-fund-direct-growth` |
| 3 | `primary` | [HDFC ELSS Tax Saver Fund – Direct Plan Growth](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth) | `hdfc-elss-tax-saver-fund-direct-plan-growth` |
| 4 | `primary` | [HDFC Small Cap Fund – Direct Growth](https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth) | `hdfc-small-cap-fund-direct-growth` |
| 5 | `primary` | [HDFC Balanced Advantage Fund – Direct Growth](https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth) | `hdfc-balanced-advantage-fund-direct-growth` |
| 6 | `variant` | [HDFC Large Cap Fund – Regular Growth](https://groww.in/mutual-funds/hdfc-large-cap-fund-regular-growth) | `hdfc-large-cap-fund-regular-growth` |
| 7 | `variant` | [HDFC Large Cap Fund – Direct IDCW](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-idcw) | `hdfc-large-cap-fund-direct-idcw` |
| 8 | `variant` | [HDFC Focused Large Cap Fund – Direct Growth](https://groww.in/mutual-funds/hdfc-focused-large-cap-direct-plan-growth) | `hdfc-focused-large-cap-direct-plan-growth` |
| 9 | `amc` | [HDFC Mutual Funds (AMC)](https://groww.in/mutual-funds/amc/hdfc-mutual-funds) | `hdfc-mutual-funds` |
| 10 | `category` | [Best Large and Mid Cap Mutual Funds](https://groww.in/mutual-funds/category/best-large-and-midcap-mutual-funds) | `best-large-and-midcap-mutual-funds` |
| 11 | `category` | [Best Flexi Cap Mutual Funds](https://groww.in/mutual-funds/category/best-flexi-cap-mutual-funds) | `best-flexi-cap-mutual-funds` |
| 12 | `tool` | [SIP Calculator](https://groww.in/calculators/sip-calculator) | `sip-calculator` |
| 13 | `regulatory` | [Riskometer (SEBI)](https://groww.in/p/riskometer) | `riskometer` |
| 14 | `education` | [What is the difference between ELSS and SIP?](https://groww.in/questions/what-is-the-difference-between-elss-and-sip) | `what-is-the-difference-between-elss-and-sip` |
| 15 | `education` | [Tax on mutual funds](https://groww.in/blog/tax-on-mutual-funds/) | `tax-on-mutual-funds` |

### What each role means

- **`primary`** (5 pages) — In scope. One of the 5 Direct-Growth schemes this bot answers about.
- **`plan-variant`** (0 pages) — Same scheme, different plan. Present to prove the entity filter works, not to be answered.
- **`context`** (0 pages) — Concept or process page. Retrieved for general questions, never for a scheme's numbers.

## Provenance

- Pages are fetched from `groww.in` and cleaned by `mf_rag/loaders.py`. Nothing is hand-typed; a fact that is not on a fetched page cannot be in the corpus.
- No third-party blogs, aggregators or forums are used as sources for scheme facts. Every number is HDFC's, as published on Groww.
- Performance tables are stripped at ingestion. Return figures are deliberately absent from the corpus so the bot cannot quote one, rather than relying on a prompt instruction not to.
- Re-fetch with `python -m mf_rag.cli ingest --refresh` (3 retries, 1.5s between pages).

## Verified current

All 15 entries in `mf_rag/sources.py` have a row here, and all 15 rows here have an entry in the registry. `tests/test_corpus.py` asserts this, so adding a URL in one place and not the other fails the suite.
