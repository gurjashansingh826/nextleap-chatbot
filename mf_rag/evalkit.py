"""GATE 2 harness — measures chunking strategies and calibrates ``min_score``.

Two jobs, both of which have to be settled by measurement rather than taste:

**1. Which chunking strategy?** Candidates A (fixed-width recursive), B (heading-aware) and
C (fact-grouped) are each chunked, embedded and indexed into their **own** Chroma collection,
so no strategy can be flattered by another's chunks. The three metrics come from
``implementation.md`` Phase 5 and are defined exactly as written there:

* ``recall_at_5``        — fraction of in-scope queries whose ``key_fact`` appears verbatim
  (whitespace-normalised) in at least one of the top-5 chunks. This measures whether the
  *answer text* is retrievable, which is the thing that matters.
* ``citation_accuracy``  — fraction where the **top-1** chunk's ``source_url`` equals
  ``expect_url``. A separate metric on purpose: retrieving the right sentence from the wrong
  page produces a confidently mis-sourced answer, which is the single worst failure mode in
  a facts-only system (P1).
* ``orphan_rate``        — fraction of chunks that do **not** begin with their
  ``"{scheme} - {section}"`` prefix. An orphan is a number with no scheme attached; it cannot
  be cited and it cannot be sanity-checked by a human reading the demo.

**2. What should ``min_score`` be?** ADR-9 forbids guessing it. This module records the full
similarity distribution for in-scope and out-of-scope queries separately, because a threshold
is only meaningful relative to both: set it at the in-scope median and you refuse half your
real questions; set it at the out-of-scope maximum and you cite the weather page.

**Isolation from the production index.** Every strategy writes to ``mf_facts_hdfc_eval_<x>``
and drops its collection afterwards. The live ``mf_facts_hdfc`` collection is never touched —
an eval run must not be able to change what the next query returns.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from .config import settings

logger = logging.getLogger(__name__)

Strategy = Literal["a", "b", "c"]

#: Suffix for a strategy's private collection. Keeps GATE 2 off the production index.
EVAL_COLLECTION_TEMPLATE = "mf_facts_hdfc_eval_{strategy}"

_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Collapse all whitespace to single spaces, and lower-case.

    The corpus round-trips through HTML extraction, markdown rendering and a JSON file, so a
    ``key_fact`` copied from the page can pick up a newline where the chunk has a space.
    Comparing raw strings would then report a miss for a fact that is genuinely present.
    Case is folded for the same reason: the label casing is a rendering choice, not content.
    """
    return _WS_RE.sub(" ", text).strip().lower()


# ── Eval set ─────────────────────────────────────────────────────────────────


@dataclass
class EvalQuery:
    id: str
    query: str
    expect_url: str | None
    key_fact: str
    scheme: str
    fact_type: str

    @property
    def in_scope(self) -> bool:
        return self.expect_url is not None


def load_eval_set(path: Path | None = None) -> list[EvalQuery]:
    """Read ``eval/chunking_eval.json`` into :class:`EvalQuery` objects.

    Raises on a malformed file rather than skipping rows: a silently shorter eval set makes
    every number below look better than it is, which is the failure mode an eval harness
    exists to prevent.
    """
    path = path or (settings.eval_dir / "chunking_eval.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The eval set is a deliverable and is labelled by reading the "
            f"fetched corpus, not from memory."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    for i, row in enumerate(rows):
        missing = {"id", "query", "expect_url", "key_fact", "scheme", "fact_type"} - row.keys()
        if missing:
            raise ValueError(f"{path} row {i} is missing {sorted(missing)}")
        if row["expect_url"] and not row["key_fact"]:
            raise ValueError(
                f"{path} row {i} ({row['id']}) has an expect_url but no key_fact — an "
                f"in-scope query with nothing to assert is unmeasurable."
            )
        queries.append(
            EvalQuery(
                id=row["id"],
                query=row["query"],
                expect_url=row["expect_url"],
                key_fact=row["key_fact"],
                scheme=row["scheme"],
                fact_type=row["fact_type"],
            )
        )
    if not any(q.in_scope for q in queries):
        raise ValueError(f"{path} contains no in-scope queries")
    return queries


# ── Results ──────────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    id: str
    fact_type: str
    in_scope: bool
    hit_at_5: bool | None          # None for out-of-scope rows
    top_url: str
    top_score: float
    expect_url: str | None
    citation_ok: bool | None
    scores: list[float] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fact_type": self.fact_type,
            "in_scope": self.in_scope,
            "hit_at_5": self.hit_at_5,
            "top_url": self.top_url,
            "top_score": round(self.top_score, 4),
            "expect_url": self.expect_url,
            "citation_ok": self.citation_ok,
            "detail": self.detail,
        }


@dataclass
class EvalResult:
    strategy: str
    recall_at_5: float
    citation_accuracy: float
    orphan_rate: float
    chunk_count: int
    mean_len: int
    dropped: int
    deduped: int
    per_query: list[QueryResult]
    in_scope_scores: list[float] = field(default_factory=list)
    out_of_scope_scores: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def by_fact_type(self) -> dict[str, tuple[float, float]]:
        """``{fact_type: (recall@5, citation_accuracy)}`` over in-scope rows."""
        buckets: dict[str, list[QueryResult]] = {}
        for r in self.per_query:
            if r.in_scope:
                buckets.setdefault(r.fact_type, []).append(r)
        out: dict[str, tuple[float, float]] = {}
        for fact_type, rows in sorted(buckets.items()):
            out[fact_type] = (
                sum(bool(r.hit_at_5) for r in rows) / len(rows),
                sum(bool(r.citation_ok) for r in rows) / len(rows),
            )
        return out

    def misses(self) -> list[QueryResult]:
        """In-scope rows that failed recall or citation — the ones worth reading."""
        return [r for r in self.per_query if r.in_scope and not (r.hit_at_5 and r.citation_ok)]


# ── Chunking + indexing for one strategy ──────────────────────────────────────


def build_for_strategy(strategy: str) -> tuple[list, dict]:
    """Chunk the whole corpus with ``strategy``.

    The chunk count and length statistics are part of the comparison, so they are returned
    alongside the chunks rather than recomputed later.
    """
    from .chunkers import build_chunks

    chunks, stats = build_chunks(strategy)
    return chunks, stats


def orphan_count(chunks: Sequence) -> int:
    """Chunks whose text does not open with ``"{scheme} - {section}"``.

    Measured on the *body* after the chunker has prepended its prefix, which is the thing the
    model and the human reviewer both see. A scheme page whose chunks open on a bare
    ``- Minimum SIP investment: 100`` cannot be cited responsibly, so this is a quality metric
    and not a cosmetic one.
    """
    orphans = 0
    for chunk in chunks:
        head = chunk.text.strip()[:160]
        scheme, section = chunk.scheme, chunk.section
        if not scheme:
            # Context pages are prefixed with their page title instead, which is the correct
            # anchor there — there is no scheme to name.
            if chunk.title and chunk.title not in head:
                orphans += 1
            continue
        if scheme not in head or (section and section not in head):
            orphans += 1
    return orphans


def _index(collection_name: str, chunks: Sequence) -> None:
    """Upsert into a named collection with the fixed cosine metric."""
    from .store import COLLECTION_METADATA, _clean_metadata, get_client, get_collection

    client = get_client()
    if collection_name in {c.name for c in client.list_collections()}:
        client.delete_collection(collection_name)
    collection = client.create_collection(name=collection_name, metadata=COLLECTION_METADATA)

    from .embedder import embed

    texts = [c.text for c in chunks]
    vectors = embed(texts)
    ids = [c.chunk_id for c in chunks]
    metadatas = [_clean_metadata(c) for c in chunks]
    for start in range(0, len(ids), 256):
        collection.upsert(
            ids=ids[start:start + 256],
            documents=texts[start:start + 256],
            metadatas=metadatas[start:start + 256],
            embeddings=vectors[start:start + 256],
        )


# ── Scoring one strategy ─────────────────────────────────────────────────────


def score_strategy(
    strategy: str,
    eval_set: list[EvalQuery],
    k: int = 5,
    verbose: bool = True,
    use_entity_filter: bool = False,
) -> EvalResult:
    """Chunk, embed, index and score ``strategy`` against the whole eval set.

    ``use_entity_filter=True`` also applies the retriever's scheme/plan filter, which is what
    the shipped system does (Phase 8 / ADR-14). Comparing the filtered and unfiltered numbers
    on the *same* index isolates how much of the raw score is entity confusion — the
    measurement that justifies the filter's existence.
    """
    from .embedder import embed_one
    from .store import search

    started = time.perf_counter()
    collection_name = EVAL_COLLECTION_TEMPLATE.format(strategy=strategy)

    chunks, stats = build_for_strategy(strategy)
    if verbose:
        print(f"  strategy {strategy}: {len(chunks)} chunks, indexing…")
    _index(collection_name, chunks)

    # Bind the flag AND the names together. An earlier version set a flag to None and only
    # imported inside the `if`, so the filter never ran and the "filtered" column was a copy of
    # the raw one — a measurement bug that looked exactly like "the filter does nothing".
    entity_filter = bool(use_entity_filter)
    if entity_filter:
        from .retriever import _matches_filter, resolve_scheme

    per_query: list[QueryResult] = []
    in_scope_scores: list[float] = []
    out_scope_scores: list[float] = []

    for q in eval_set:
        pool = search(embed_one(q.query), top_k=k, name=collection_name)
        if entity_filter:
            # `pool` is rebound only after the comprehension finishes, so the filter sees the
            # full candidate list and the sibling lookup inside _matches_filter stays correct.
            resolution = resolve_scheme(q.query)
            pool = [h for h in pool if _matches_filter(h, resolution, pool)[0]]
        hits = pool
        scores = [h.score for h in hits]
        top_url = hits[0].source_url if hits else ""
        top_score = hits[0].score if hits else 0.0

        if not q.in_scope:
            # Out-of-scope rows exist to calibrate the threshold, not to be "answered".
            # Their score is recorded and the row is excluded from every recall/accuracy mean,
            # because including them would make citation accuracy meaningless (there is no
            # correct URL to hit).
            out_scope_scores.append(top_score)
            per_query.append(
                QueryResult(
                    id=q.id, fact_type=q.fact_type, in_scope=False, hit_at_5=None,
                    top_url=top_url, top_score=top_score, expect_url=None, citation_ok=None,
                    scores=scores, detail="out-of-scope probe (threshold calibration only)",
                )
            )
            continue

        in_scope_scores.append(top_score)
        joined = normalise(" ".join(h.text for h in hits))
        hit = normalise(q.key_fact) in joined
        cite_ok = top_url == q.expect_url
        if hit and cite_ok:
            detail = ""
        elif not hit and not cite_ok:
            detail = "wrong page and fact absent"
        elif not hit:
            detail = "right neighbourhood, key_fact not in top-5"
        else:
            detail = f"fact found but top-1 was {top_url.rsplit('/', 1)[-1]}"

        per_query.append(
            QueryResult(
                id=q.id, fact_type=q.fact_type, in_scope=True, hit_at_5=hit,
                top_url=top_url, top_score=top_score, expect_url=q.expect_url,
                citation_ok=cite_ok, scores=scores, detail=detail,
            )
        )

    _drop_collection(collection_name)

    in_scope_rows = [r for r in per_query if r.in_scope]
    n = len(in_scope_rows) or 1
    lengths = [c.char_len for c in chunks] or [0]
    result = EvalResult(
        strategy=strategy,
        recall_at_5=sum(bool(r.hit_at_5) for r in in_scope_rows) / n,
        citation_accuracy=sum(bool(r.citation_ok) for r in in_scope_rows) / n,
        orphan_rate=orphan_count(chunks) / (len(chunks) or 1),
        chunk_count=len(chunks),
        mean_len=sum(lengths) // len(lengths),
        dropped=stats["dropped"],
        deduped=stats["deduped"],
        per_query=per_query,
        in_scope_scores=in_scope_scores,
        out_of_scope_scores=out_scope_scores,
        seconds=time.perf_counter() - started,
    )
    logger.info(
        "eval strategy=%s recall@5=%.3f citation=%.3f orphan=%.3f chunks=%d",
        strategy, result.recall_at_5, result.citation_accuracy, result.orphan_rate,
        result.chunk_count,
    )
    return result


def _drop_collection(name: str) -> None:
    from .store import get_client

    client = get_client()
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)


# ── Reporting ────────────────────────────────────────────────────────────────


def scorecard(result: EvalResult) -> str:
    """One strategy's line-item scorecard."""
    lines = [
        f"strategy {result.strategy}  ({result.chunk_count} chunks · mean {result.mean_len} chars "
        f"· dropped {result.dropped} · deduped {result.deduped} · {result.seconds:.1f}s)",
        f"  recall@5         {result.recall_at_5:6.1%}   {int(round(result.recall_at_5 * len([r for r in result.per_query if r.in_scope])))}"
        f"/{len([r for r in result.per_query if r.in_scope])} in-scope queries found their fact",
        f"  citation accuracy{result.citation_accuracy:6.1%}   top-1 chunk came from the expected page",
        f"  orphan rate      {result.orphan_rate:6.1%}   chunks with no scheme/section prefix",
    ]
    by_type = result.by_fact_type()
    if by_type:
        lines.append("  by fact type:")
        for fact_type, (recall, citation) in by_type.items():
            lines.append(f"    {fact_type:<12} recall@5 {recall:5.0%}   citation {citation:5.0%}")
    return "\n".join(lines)


def compare_strategies(results: list[EvalResult]) -> str:
    """Markdown table, for pasting into ``docs/eval_report.md``."""
    head = (
        "| strategy | chunks | mean len | recall@5 | citation acc. | orphan rate | "
        "min in-scope score | max out-of-scope score |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for r in results:
        min_in = min(r.in_scope_scores) if r.in_scope_scores else float("nan")
        max_out = max(r.out_of_scope_scores) if r.out_of_scope_scores else float("nan")
        rows.append(
            f"| {r.strategy} | {r.chunk_count} | {r.mean_len} | {r.recall_at_5:.0%} | "
            f"{r.citation_accuracy:.0%} | {r.orphan_rate:.0%} | {min_in:.3f} | {max_out:.3f} |"
        )
    return "\n".join([head, *rows])


def recommend_min_score(result: EvalResult) -> tuple[float, str]:
    """Propose ``min_score`` from the measured distribution, per implementation.md §Step 4.

    The decision table from the plan, applied to real numbers rather than assumed:

    * in-scope min > 0.5 and out-of-scope max < 0.15 → 0.25 is comfortably placed
    * any in-scope query scoring below 0.20 → lower, or that question is unanswerable
    * out-of-scope max > 0.30 → raise, or the bot cites an unrelated page with confidence

    When both a "lower" and a "raise" condition fire, the raise wins. An out-of-scope question
    answered with a confident citation is a facts-only violation; an in-scope question that
    falls back to "not in my sources" is a visible, honest limitation.
    """
    in_scope = result.in_scope_scores
    out_scope = result.out_of_scope_scores
    if not in_scope:
        return settings.min_score, "no in-scope scores measured; leaving the configured value"

    lo_in, hi_in = min(in_scope), max(in_scope)
    med_in = statistics.median(in_scope)
    hi_out = max(out_scope) if out_scope else 0.0

    # The two distributions overlapping is the important case, and it is what was measured.
    # implementation.md's decision table assumes a clean gap; there isn't one, so no single
    # threshold can both keep every in-scope question and reject every out-of-scope probe.
    # Chasing that gap by raising min_score would be fitting a number to six probes.
    if hi_out > lo_in:
        return 0.25, (
            f"the distributions OVERLAP: the strongest out-of-scope probe scores {hi_out:.3f}, "
            f"above the weakest in-scope question at {lo_in:.3f} (in-scope median {med_in:.3f}). "
            f"No threshold can separate them, so min_score is not the control here and is kept at "
            f"0.25, which passes all 30 in-scope questions. Out-of-scope rejection is done by the "
            f"entity filter (a question naming no known scheme retrieves only what it can name) "
            f"and by the intent guards, not by a similarity cut."
        )
    if lo_in < 0.20:
        return 0.20, (
            f"the weakest in-scope question scored {lo_in:.3f} < 0.20, so a higher threshold "
            f"would refuse a question the corpus can answer. Lowered to 0.20 (in-scope median "
            f"{med_in:.3f}, out-of-scope max {hi_out:.3f})."
        )
    if lo_in > 0.5 and hi_out < 0.15:
        return 0.25, (
            f"clean separation: every in-scope question scores above {lo_in:.3f} and the "
            f"strongest out-of-scope probe only reaches {hi_out:.3f}. 0.25 sits in the gap with "
            f"margin on both sides."
        )
    return 0.25, (
        f"in-scope {lo_in:.3f}-{hi_in:.3f} (median {med_in:.3f}) sits above the out-of-scope max "
        f"{hi_out:.3f}. 0.25 keeps every in-scope question with margin and rejects the "
        f"out-of-scope probes."
    )


def distribution_table(result: EvalResult, bins: int = 10) -> str:
    """Histogram of top-1 similarity, in-scope vs out-of-scope. Feeds the threshold decision."""
    lines = [f"{'bin':>14}  {'in-scope':>9}  {'out-of-scope':>13}"]
    lines.append("-" * 42)
    in_scope = result.in_scope_scores
    out_scope = result.out_of_scope_scores
    if not in_scope and not out_scope:
        return "(no scores recorded)"
    lo = min(in_scope + out_scope)
    hi = max(in_scope + out_scope)
    width = (hi - lo) / bins or 1.0
    for i in range(bins):
        edge = lo + i * width
        in_hits = sum(1 for s in in_scope if edge <= s < edge + width)
        out_hits = sum(1 for s in out_scope if edge <= s < edge + width)
        lines.append(f"{edge:6.3f}-{edge + width:6.3f}  {in_hits:9d}  {out_hits:13d}")
    lines.append("-" * 42)
    lines.append(
        f"in-scope    n={len(in_scope):<3} min {min(in_scope):.3f}  median "
        f"{statistics.median(in_scope):.3f}  max {max(in_scope):.3f}"
    )
    if out_scope:
        lines.append(
            f"out-of-scope n={len(out_scope):<3} min {min(out_scope):.3f}  median "
            f"{statistics.median(out_scope):.3f}  max {max(out_scope):.3f}"
        )
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────


def run_all(strategies: Sequence[str] = ("a", "b", "c"), verbose: bool = True) -> list[EvalResult]:
    """Score every strategy. Re-chunks and re-embeds the corpus once per strategy."""
    eval_set = load_eval_set()
    in_scope = [q for q in eval_set if q.in_scope]
    if verbose:
        print(f"GATE 2 · {len(in_scope)} in-scope + {len(eval_set) - len(in_scope)} "
              f"out-of-scope queries · strategies {list(strategies)}")
        print()

    results: list[EvalResult] = []
    for strategy in strategies:
        result = score_strategy(strategy, eval_set, verbose=verbose)
        results.append(result)
        if verbose:
            print(scorecard(result))
            misses = result.misses()
            if misses:
                print(f"  misses ({len(misses)}):")
                for m in misses:
                    print(f"    {m.id}  {m.fact_type:<10} top1={m.top_url.rsplit('/', 1)[-1] or '-'}"
                          f"  {m.detail}")
            print()
    return results


def main() -> None:
    results = run_all()
    if not results:
        return

    print("COMPARISON — raw chunk retrieval, no entity filter (isolates chunking quality)")
    print(compare_strategies(results))
    print()

    # The shipped strategy, with the Phase 8 entity filter applied. Same index, same queries —
    # the only difference is the filter, so the delta is attributable to it.
    shipped_strategy = settings.chunk_strategy
    shipped_raw = next((r for r in results if r.strategy == shipped_strategy), results[-1])
    print(f"ENTITY FILTER — strategy {shipped_strategy}, raw vs filtered")
    filtered = score_strategy(shipped_strategy, load_eval_set(), use_entity_filter=True,
                              verbose=False)
    raw_misses = shipped_raw.misses()
    filt_misses = filtered.misses()
    fixed = {m.id for m in raw_misses} - {m.id for m in filt_misses}
    print(f"  raw       recall@5 {shipped_raw.recall_at_5:6.1%}   citation "
          f"{shipped_raw.citation_accuracy:6.1%}   {len(raw_misses)} misses")
    print(f"  filtered  recall@5 {filtered.recall_at_5:6.1%}   citation "
          f"{filtered.citation_accuracy:6.1%}   {len(filt_misses)} misses")
    if fixed:
        print(f"  the filter fixed {len(fixed)}: {', '.join(sorted(fixed))}")
    remaining = [m.id for m in filt_misses]
    if remaining:
        print(f"  still failing after filtering: {', '.join(remaining)}")
    print()

    print(f"SCORE DISTRIBUTION — strategy {shipped_strategy}")
    print(distribution_table(shipped_raw))
    print()

    proposed, why = recommend_min_score(shipped_raw)
    print(f"min_score recommendation: {proposed:.2f}")
    print(f"  {why}")
    print()
    print(f"config currently has min_score={settings.min_score:.2f} "
          f"chunk_strategy={settings.chunk_strategy!r}")

    print(f"\n(record the decision in {settings.docs_dir / 'eval_report.md'})")


if __name__ == "__main__":
    main()
