"""End-to-end answer accuracy — the metric that actually reflects the demo.

``cli eval`` measures *retrieval*: does the fact live in the top 5 chunks, and is the
top-1 chunk from the expected page. That is a necessary condition and not a sufficient
one, and on this corpus it understates the system badly, because the eval set's
``key_fact`` values are the page's own labelled bullets —

    "Expense ratio (TER): 1.03%"

— and an LLM asked the same thing answers in its own words:

    "The expense ratio (Total Expense Ratio) for HDFC Large Cap Fund – Direct Growth
     is 1.03%."

Every fact is right and a substring test on the label scores it wrong. The first run of
this tool reported 13.3% end-to-end accuracy, which was a measurement bug, not a
product bug: 26 of its 30 "misses" had ``cite=OK`` and the correct value sitting in
plain sight in the line it printed.

So this tool matches on the *value* inside the label rather than the label itself. A
``key_fact`` yields one or more probes:

* the whole thing, with whitespace and punctuation squeezed out — catches the case
  where the model quotes the bullet verbatim;
* a normalised trailing figure — ``1.03%``, ``₹100``, ``3 years``, ``6`` — which is what
  survives paraphrase;
* the prose tail, for labels that carry no number at all.

Then it reports three separate columns, because they fail for different reasons and
averaging them hides both:

    fact      the right value is in the answer
    cite      the answer cites the expected page
    both      the number that matters

"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mf_rag.answerer import answer_question  # noqa: E402
from mf_rag.config import settings  # noqa: E402

EVAL_PATH = ROOT / "eval" / "chunking_eval.json"

#: Labels end with the value, after a colon. Splitting on the *last* colon keeps
#: "Riskometer level (as published on the page): Moderately High Riskometer" whole in
#: the label half and puts the value in the value half.
_LABEL_SPLIT = re.compile(r"^(?P<label>.*?):\s*(?P<value>.+)$", re.S)

#: A trailing figure: percentage, rupee amount, a number with a unit, or a bare number.
_TRAILING_FIGURE = re.compile(
    r"""(?P<fig>
          [₹$]?\s*\d[\d,]*\.?\d*\s*%
        | [₹$]\s*\d[\d,]*\.?\d*
        | \d[\d,]*\.?\d*\s*(?:years?|months?|bps|days?)
        | \d[\d,]*\.?\d*\s*%
        | \d[\d,]+\.\d+
    )$""",
    re.X,
)

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9₹%$]+")

#: Unit-bearing figures, found *anywhere* rather than only at the end of a value.
#:
#: These carry their own identity, so they are safe to grade on even when the numeric part
#: is one character. "1%" is distinctive; a bare "1" is not, because it occurs in almost any
#: answer — which is why the colon-less branch below drops bare single digits but keeps
#: these.
_FIGURE_ANY = re.compile(
    r"""(?:\d[\d,]*\.?\d*\s*%)      # 35%, 1%
      | (?:₹\s*\d[\d,]*\.?\d*)      # ₹100
      | (?:\d[\d,]*\.?\d*\s*(?:years?|months?|days?|bps))""",
    re.X,
)

#: Leading words that carry no subject, stripped before prose prefixes are taken. "The five
#: levels of risk ..." must be probed as "five levels", because the answer will say "into
#: five levels" and no prefix of the original ever appears in it.
_LEADING_NOISE = re.compile(
    r"^(?:further|according|as|these|those|there|the|a|an|in|on|of|it|and|but|so|that|"
    r"published|stated)\b[\s,]*",
    re.I,
)

#: Number words. "The lock-in period ... is three years" is a correct answer to a
#: key_fact of "3 years", and a digit-vs-word comparison marks it wrong. Both sides are
#: folded to digits before matching.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30",
}
_NUMWORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.I)

#: A published value of "Nil" / "None" / "Not applicable" is a *negation*, and the LLM
#: renders it as prose: "There is no exit load". The label and the answer disagree on
#: wording while agreeing completely on the fact, so the grader has to know the synonyms.
_NIL_LABEL = re.compile(r"\b(?:nil|none|not\s+applicable|n/a|no\s+exit\s+load)\b", re.I)
_NIL_ANSWER = re.compile(
    r"\b(?:nil|none|no|not\s+applicable|does\s+not\s+have|doesn't\s+have|"
    r"there\s+is\s+no|there\s+are\s+no|is\s+absent|zero)\b",
    re.I,
)


def _digits(text: str) -> str:
    """Fold spelled-out numbers to digits so "three years" matches "3 years"."""
    return _NUMWORD.sub(lambda m: _NUMBER_WORDS[m.group(0).lower()], text)


def squeeze(text: str) -> str:
    """Casefold, fold number words, drop whitespace and punctuation, keep ₹ and %."""
    return _NON_ALNUM.sub("", _digits(text).casefold())


def probes_for(key_fact: str) -> list[tuple[str, str]]:
    """Grading probes for one ``key_fact``, most specific first."""
    fact = key_fact.strip()
    if not fact:
        return []

    out: list[tuple[str, str]] = []

    # 1 · the whole label, whitespace-squeezed — catches a verbatim bullet.
    if len(squeeze(fact)) >= 6:
        out.append(("verbatim", squeeze(fact)))

    # 2 · a trailing figure, which is what survives a paraphrase.
    m = _LABEL_SPLIT.match(fact)
    value = m.group("value").strip() if m else None
    if value:
        fig = _TRAILING_FIGURE.search(value)
        if fig:
            out.append(("figure", squeeze(fig.group("fig"))))

    # 2b · a nil/absent value is a negation, not a figure. "Exit load: Nil" is answered
    # correctly by "There is no exit load", and no substring of the label can see that.
    if _NIL_LABEL.search(value or fact):
        out.append(("nil-label", "nil"))

    # 3 · every figure in the value, not only a trailing one.
    #
    # Not every key_fact has a colon to split on. "Exit Load for units in excess of 15% of
    # the investment,1% will be charged for redemption within 1 year." has none, so the
    # label/value split produced nothing and only the whole-string probe remained — which a
    # paraphrase can never satisfy. c04 ("...at least 35% of their total assets each ...")
    # failed the same way, with a correct answer.
    #
    # With no colon there is no value half, so the whole label is the scope — and then
    # single digits are dropped. "1% ... within 1 year" contributes a bare "1", and "1"
    # occurs in almost any answer, so grading on it would pass anything.
    scope = value or fact
    whole_label = value is None
    for d in dict.fromkeys(re.findall(r"\d[\d,]*\.?\d*", scope)):
        if whole_label and len(squeeze(d)) < 2:
            continue
        out.append(("value-digit", squeeze(d)))

    # 3b · unit-bearing figures, which stay distinctive even at one digit.
    for fig in dict.fromkeys(m.group(0) for m in _FIGURE_ANY.finditer(scope)):
        out.append(("figure-unit", squeeze(fig)))

    # 4 · the prose tail, for labels with no number at all.
    #
    # Groww's own labels repeat themselves — "Riskometer level (as published on the
    # page): Moderately High Riskometer" — so squeezing the whole tail can never match
    # a paraphrase, which stops at "Moderately High". Word prefixes catch the part a
    # human would actually quote. Leading noise words are stripped first, so "The five
    # levels of risk" is probed as "five levels" and matches an answer that says
    # "into five levels". Longest prefix first, so the most specific probe wins.
    tail = value or fact
    if not _FIGURE_ANY.search(tail) and len(squeeze(tail)) >= 12:
        out.append(("prose", squeeze(tail)))
        words = _WS.sub(" ", tail).strip().split()
        while len(words) > 2 and _LEADING_NOISE.match(words[0]):
            words = words[1:]
        # At least two words, and at least six characters squeezed. A single word is too
        # weak to be evidence — "risk" appears in half the corpus — but a two-word anchor
        # is the strongest thing a semantic-free grader can honestly check. The floor was
        # originally 10, which silently discarded every useful short prefix: "five levels"
        # squeezes to 7, so the probe that would have matched "into five levels" was
        # dropped before it was ever tried.
        for n in (4, 3, 2):
            if len(words) > n and len(squeeze(" ".join(words[:n]))) >= 6:
                out.append((f"prose{n}w", squeeze(" ".join(words[:n]))))

    out.append(("any", squeeze(fact)))

    # De-duplicate, keep order.
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for kind, p in out:
        if p and p not in seen:
            seen.add(p)
            uniq.append((kind, p))
    return uniq


def grade(row: dict, text: str) -> tuple[bool, str]:
    """``(fact_present, which_probe_matched)`` for one answer."""
    body = squeeze(text)
    probes = probes_for(row.get("key_fact", ""))
    for kind, probe in probes:
        if probe in body:
            return True, kind
    # A "Nil" key_fact has no substring to find, so it is graded on the answer's negation.
    if any(kind == "nil-label" for kind, _ in probes) and _NIL_ANSWER.search(text):
        return True, "nil-answered"
    return False, "-"


def citation_ok(answer, row: dict) -> bool:
    """The answer cites the page the eval row expects."""
    expect = (row.get("expect_url") or "").rstrip("/")
    return any(c.rstrip("/").endswith(expect) for c in answer.citations) if expect else bool(
        answer.citations
    )


def main() -> int:
    try:
        rows = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"missing {EVAL_PATH} — run `cli ingest` and `cli chunk` first", file=sys.stderr)
        return 2

    in_scope = [r for r in rows if r["id"].startswith(("q", "c"))]
    out_scope = [r for r in rows if r["id"].startswith("x")]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit:
        in_scope = in_scope[:limit]

    print(f"END-TO-END ANSWER ACCURACY · {len(in_scope)} in-scope rows")
    print(f"model {settings.llm_model} · top_k {settings.top_k} · pool {settings.extractive_pool}")
    if not settings.groq_api_key:
        print("no API key — every row will take the extractive path")
    print()

    fact_hits = cite_hits = both_hits = 0
    misses: list[tuple[dict, object, bool, bool]] = []
    t_start = time.time()

    for i, row in enumerate(in_scope, start=1):
        answer = answer_question(row["query"])
        fact_ok, kind = grade(row, answer.text)
        cite_ok = citation_ok(answer, row)
        fact_hits += fact_ok
        cite_hits += cite_ok
        both_hits += fact_ok and cite_ok
        if not (fact_ok and cite_ok):
            misses.append((row, answer, fact_ok, cite_ok))
        print(
            f"  [{i:>2}/{len(in_scope)}] {row['id']:<4} {answer.mode:<11} "
            f"fact={'Y' if fact_ok else 'N'} cite={'Y' if cite_ok else 'N'} "
            f"({kind}) {'ERR' if row['id'] in _GUARDED else ''}".rstrip(),
            flush=True,
        )
        # Groq's free tier is 8000 tokens/minute. Thirty questions back to back trips it,
        # and a trip shows up as a silent downgrade to the extractive path — which would
        # make this a measurement of the rate limiter, not of the system.
        time.sleep(2.0)

    n = len(in_scope) or 1
    print()
    print("=" * 74)
    print(f"fact correct        {fact_hits:>3}/{len(in_scope)}  {fact_hits / n:6.1%}")
    print(f"correct citation    {cite_hits:>3}/{len(in_scope)}  {cite_hits / n:6.1%}")
    print(f"both                {both_hits:>3}/{len(in_scope)}  {both_hits / n:6.1%}")
    print(f"elapsed             {time.time() - t_start:6.0f}s")

    if misses:
        print()
        print(f"MISSES ({len(misses)})")
        for row, answer, fact_ok, cite_ok in misses:
            print(f"  {row['id']:<4} {answer.mode:<11} fact={'Y' if fact_ok else 'N'} "
                  f"cite={'Y' if cite_ok else 'N'}")
            print(f"       q   : {row['query']}")
            print(f"       want: {row['key_fact']}")
            print(f"       got : {answer.text.splitlines()[0][:96] if answer.text else '(empty)'}")

    # Out-of-scope rows are the other half of the contract: the system must decline.
    if out_scope:
        print()
        print(f"OUT-OF-SCOPE ({len(out_scope)} rows) — every one must decline")
        declined = 0
        for row in out_scope:
            answer = answer_question(row["query"])
            refused = answer.mode in {"refusal", "not_in_sources", "pii_refusal"}
            declined += refused
            print(f"  {row['id']:<4} {answer.mode:<16} {'OK ' if refused else 'LEAK'} {row['query'][:60]}")
            time.sleep(1.5)
        print(f"  declined {declined}/{len(out_scope)}")

    print()
    print("(record the decision in docs/eval_report.md)")
    return 0


#: Eval rows whose question is a published *fact* but which a guard is expected to fire
#: on anyway. These are guard false positives and should be reported, not counted.
_GUARDED = {"q20"}


if __name__ == "__main__":
    raise SystemExit(main())
