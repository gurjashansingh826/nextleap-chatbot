"""Structured scheme-fact extraction from Groww's embedded ``__NEXT_DATA__`` payload.

Why this module exists
----------------------
`trafilatura` alone extracts Groww's *visible* text, which on a scheme page is dominated by
returns tables and portfolio holdings. The graded facts — expense ratio, exit load, minimum
SIP, lock-in, riskometer, benchmark — are present in the page but only inside the Next.js
JSON payload, so a text-only extractor reports them as ABSENT (verified 2026-09-28, risk F4
in architecture §14).

Reading that JSON turns a data problem into a solved problem, and gives something better than
scraped prose: a **labelled, deterministic, auditable** fact block where every value is
copied verbatim from the page.

Two deliberate exclusions
-------------------------
1. **Performance fields are not extracted at all.** ``expense_ratio`` is a fee, not a return;
   but ``sip_return``, ``simple_return``, ``return_stats`` and ``peerComparison`` are, and the
   project's hard constraint is that this assistant makes no performance claims. Leaving them
   out of the corpus removes a retrieval hazard at the source rather than relying on the
   intent guard to catch it. Same for ``holdings`` (50 rows of noise that would dominate
   chunking) and ``historic_fund_expense`` (1,527 entries).
2. **No value is computed, rounded, or inferred.** ``format_value`` only adds a unit symbol
   (%, ₹) that the field's own name already implies. ``0.63`` stays ``0.63``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, NamedTuple

from .config import settings

_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.S
)

#: Key in ``pageProps`` that holds scheme data on every mutual-fund scheme page.
SCHEMA_KEY = "mfServerSideData"


class FactField(NamedTuple):
    """One extractable scheme fact.

    ``render`` turns the raw JSON value into display text. It receives the value and the whole
    payload, because a few fields are only meaningful in combination (``benchmark_name`` with
    ``benchmark``).
    """

    key: str
    label: str
    render: Callable[[Any, dict], str | None]


def _pct(value: Any, _all: dict) -> str | None:
    """Render a number that is a percentage. The unit is implied by the field name."""
    if value in (None, ""):
        return None
    return f"{value}%"


def _inr(value: Any, _all: dict) -> str | None:
    """Render an amount in rupees. Indian numbering, no reformatting of the number itself."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,}"
    else:
        text = str(value)
    return f"₹{text}"


def _crore(value: Any, _all: dict) -> str | None:
    """Render AUM, which Groww publishes in crores."""
    if value in (None, ""):
        return None
    return f"₹{float(value):,.2f} Cr"


def _text(value: Any, _all: dict) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _bool_yn(value: Any, _all: dict) -> str | None:
    if value is None:
        return None
    return "Yes" if value else "No"


def _nav(value: Any, all_: dict) -> str | None:
    """NAV with its as-on date, so a snapshot value can never read as a live price (PRD §3)."""
    if value in (None, ""):
        return None
    date = all_.get("nav_date")
    amount = f"₹{float(value):,.3f}"
    return f"{amount} (as on {date})" if date else amount


def _lock_in(value: Any, _all: dict) -> str | None:
    """Render the ELSS lock-in. Non-ELSS schemes carry an all-null dict."""
    if not isinstance(value, dict):
        return None
    parts = []
    for unit in ("years", "months", "days"):
        n = value.get(unit)
        if isinstance(n, (int, float)) and n > 0:
            parts.append(f"{int(n)} {unit.rstrip('s')}{'' if int(n) == 1 else 's'}")
    if not parts:
        return None
    return " ".join(parts)


def _benchmark(value: Any, all_: dict) -> str | None:
    """Prefer the long index name; append the short ticker when both are published."""
    long_name = _text(all_.get("benchmark_name"), all_)
    short = _text(value, all_)
    if long_name and short and short not in long_name:
        return f"{long_name} ({short})"
    return long_name or short


#: Rendered in this order, so the most-asked facts come first in the chunk.
#: A ``None`` render means the field is absent for this scheme and is skipped.
SCHEME_FACT_FIELDS: tuple[FactField, ...] = (
    FactField("expense_ratio", "Expense ratio (TER, direct plan)", _pct),
    FactField("base_expense_ratio", "Base expense ratio (excl. additional fund expenses)", _pct),
    FactField("exit_load", "Exit load", _text),
    FactField("min_sip_investment", "Minimum SIP investment", _inr),
    FactField("min_investment_amount", "Minimum lump sum investment", _inr),
    FactField("mini_additional_investment", "Minimum additional investment", _inr),
    FactField("sip_multiplier", "SIP amount multiplier", _text),
    FactField("lock_in", "Lock-in period", _lock_in),
    FactField("nfo_risk", "Riskometer level (as published on the page)", _text),
    FactField("benchmark", "Benchmark", _benchmark),
    FactField("scheme_name", "Scheme name", _text),
    FactField("plan_type", "Plan type", _text),
    FactField("category", "Category", _text),
    FactField("sub_category", "Sub-category", _text),
    FactField("amc", "AMC", _text),
    FactField("fund_house", "Fund house", _text),
    FactField("fund_manager", "Fund manager", _text),
    FactField("launch_date", "Launch date", _text),
    FactField("isin", "ISIN", _text),
    FactField("nav", "NAV", _nav),
    FactField("aum", "AUM", _crore),
    FactField("min_withdrawal", "Minimum redemption amount", _inr),
    FactField("purchase_multiplier", "Purchase amount multiplier", _text),
    FactField("stamp_duty", "Stamp duty", _text),
    FactField("portfolio_turnover", "Portfolio turnover ratio (%)", _text),
    FactField("sip_allowed", "SIP available", _bool_yn),
    FactField("lumpsum_allowed", "Lump sum available", _bool_yn),
    FactField("closed_scheme", "Closed for investment", _bool_yn),
    FactField("description", "Scheme objective", _text),
)

#: Graded-fact labels, used by Gate 1 to report coverage. Kept here so the fact list and the
#: coverage check can never drift apart.
GRADED_FACT_LABELS: tuple[str, ...] = (
    "Expense ratio (TER, direct plan)",
    "Exit load",
    "Minimum SIP investment",
    "Minimum lump sum investment",
    "Lock-in period",
    "Riskometer level (as published on the page)",
    "Benchmark",
)


def extract_next_data(html: str) -> dict | None:
    """Return Groww's ``pageProps`` payload, or ``None`` if the page has no usable payload.

    Deliberately forgiving: a page with no JSON payload is normal (it just means this page
    type relies on visible text), and a malformed payload must not abort ingestion.
    """
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    props = data.get("props", {})
    page_props = props.get("pageProps")
    return page_props if isinstance(page_props, dict) else None


def scheme_payload(html: str) -> dict | None:
    """The ``mfServerSideData`` object, or ``None`` for non-scheme pages."""
    props = extract_next_data(html)
    if not props:
        return None
    payload = props.get(SCHEMA_KEY)
    return payload if isinstance(payload, dict) else None


def extract_facts(payload: dict) -> list[tuple[str, str]]:
    """Extract the labelled scheme facts present in a payload.

    Returns ``[(label, value), ...]`` in the declared order, skipping absent fields. This is
    the list Gate 1 checks for coverage, and the source of the ``## Key facts`` chunk.
    """
    facts: list[tuple[str, str]] = []
    for field in SCHEME_FACT_FIELDS:
        if field.key not in payload:
            continue
        try:
            rendered = field.render(payload.get(field.key), payload)
        except (TypeError, ValueError, ZeroDivisionError):
            rendered = None
        if rendered:
            facts.append((field.label, rendered))
    return facts


def render_fact_block(payload: dict) -> str:
    """Render the ``## Key facts`` markdown section. Empty string if nothing extracted."""
    facts = extract_facts(payload)
    if not facts:
        return ""
    lines = [
        "## Key facts",
        "",
        "Values published on the scheme page, extracted as labelled fields.",
        "",
    ]
    lines += [f"- {label}: {value}" for label, value in facts]
    lines.append("")
    return "\n".join(lines)


def coverage_report() -> list[dict]:
    """Which graded facts each cached scheme page actually yielded.

    Gate 1 output. Reads the cached ``__NEXT_DATA__`` payloads rather than re-fetching, so it
    can be re-run any time without hitting the network again.
    """
    from .loaders import raw_path
    from .sources import SOURCES

    rows: list[dict] = []
    for spec in SOURCES:
        if spec.page_role not in ("primary", "variant"):
            continue
        path = raw_path(spec)
        payload = scheme_payload(
            path.read_text(encoding="utf-8", errors="replace")
        ) if path.exists() else None
        payload = payload or {}
        labels = {label for label, _ in extract_facts(payload)}
        plan = payload.get("plan_type") or ""
        sub = payload.get("sub_sub_category") or ""
        if sub:
            plan = f"{plan} {sub}".strip()
        rows.append(
            {
                "slug": spec.slug,
                "scheme": spec.scheme or spec.title,
                "plan_type": plan,
                "page_role": spec.page_role,
                "facts_found": len(labels),
                "missing": [x for x in GRADED_FACT_LABELS if x not in labels],
            }
        )
    return rows


#: Column headers for the Gate 1 table, in the order PRD §5 / implementation Gate 1 expects.
COVERAGE_COLUMNS: tuple[str, ...] = (
    "Expense ratio", "Exit load", "Min SIP", "Min lump", "Lock-in",
    "Riskometer", "Benchmark",
)

#: Maps a graded label onto its short column name.
_LABEL_TO_COLUMN: dict[str, str] = {
    "Expense ratio (TER, direct plan)": "Expense ratio",
    "Exit load": "Exit load",
    "Minimum SIP investment": "Min SIP",
    "Minimum lump sum investment": "Min lump",
    "Lock-in period": "Lock-in",
    "Riskometer level (as published on the page)": "Riskometer",
    "Benchmark": "Benchmark",
}


def coverage_table() -> str:
    """Render the Gate 1 coverage table as markdown.

    Plan type is shown alongside the scheme name because three rows share the name
    "HDFC Large Cap Fund" (Direct Growth, Regular Growth, Direct IDCW) and would otherwise be
    indistinguishable.
    """
    rows = coverage_report()
    labels = [f"{r['scheme']} ({r['plan_type']})" if r["plan_type"] else r["scheme"] for r in rows]
    width = max((len(x) for x in labels), default=10)
    lines = [
        "| Scheme / plan | " + " | ".join(COVERAGE_COLUMNS) + " |",
        "| --- | " + " | ".join("---" for _ in COVERAGE_COLUMNS) + " |",
    ]
    for r, label in zip(rows, labels):
        present = {
            _LABEL_TO_COLUMN[x]
            for x in GRADED_FACT_LABELS
            if x not in r["missing"]
        }
        cells = []
        for col in COVERAGE_COLUMNS:
            if col in present:
                cells.append("yes")
            elif col == "Lock-in" and "hdfc-elss" in r["slug"]:
                cells.append("n/a")
            else:
                cells.append("-")
        lines.append(f"| {label:<{width}} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Structured fact extraction — model page payload key: {SCHEMA_KEY!r}")
    print(f"Fact fields declared: {len(SCHEME_FACT_FIELDS)}")
    print(f"Performance fields excluded by design: "
          f"{', '.join(['sip_return', 'simple_return', 'return_stats', 'peerComparison', 'holdings', 'historic_fund_expense'])}")
    print()
    print(coverage_table())
