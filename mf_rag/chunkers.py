"""STAGE 2 — Chunking.

Strategy selection, and the evidence behind it
----------------------------------------------
Three candidates are implemented. The default is **Candidate C (fact-grouped)**, chosen after
inspecting the real corpus rather than by intuition. The evidence:

1. **8 of 15 pages are scheme pages whose entire content is a ``## Key facts`` block of ~28
   labelled bullets, ~1,100 chars, in an identical shape on every page.** That is ~1.6x
   ``chunk_size`` (700), so *any* fixed-size splitter — recursive or not — cuts the list
   mid-way. A chunk that begins ``- Minimum lump sum investment: ...`` with no scheme context
   cannot answer "what is the minimum SIP" on its own, and cannot be cited accurately either.
2. **The graded questions cluster into clean field groups.** Fees, minimum investments,
   lock-in, risk+benchmark, identity, NAV/AUM, objective. Every group is 200-450 chars, so
   grouping keeps each one *whole* and under the size cap, and each group answers a complete
   question class on its own.
3. **The other 7 pages are narrative prose** (2,000-7,200 chars) with real markdown headings
   and no field structure. Those want heading-aware splitting, not fact grouping.

So the effective method is a **three-tier hybrid**: fact grouping where the data is labelled
facts, heading-aware splitting where the data is prose, and a recursive character splitter
only as the inner fallback for anything still oversized. Every chunk is prefixed with its
scheme and section so it is self-describing and citable (principle P5).

Why not an LLM-based semantic chunker? It would be slower, non-deterministic, cost money, and
add a failure mode — for a 15-page corpus with a known field structure, grouping on the
labels already in the data is strictly better and fully auditable. Phase 5's eval measures
Recall@5 and citation accuracy for all three and confirms or overturns this default.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .loaders import ProcessedDoc, load_processed_docs
from .scheme_facts import SCHEME_FACT_FIELDS

Strategy = Literal["a", "b", "c"]


@dataclass
class Chunk:
    """One retrievable unit. ``source_url`` is mandatory — provenance rule P1."""

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
    also_seen_at: list[str] = field(default_factory=list)
    strategy: str = ""
    #: "facts" for a labelled fact group, "prose" for narrative. Fact groups are exempt from
    #: the generic minimum-length filter — see :func:`_filter_chunks`.
    kind: str = "prose"

    def to_dict(self) -> dict:
        return asdict(self)


class Chunker(Protocol):
    name: str

    def split(self, doc: ProcessedDoc) -> list[Chunk]: ...


# ── Fact groups ──────────────────────────────────────────────────────────────

#: Field groups for the ``## Key facts`` block, in emission order. The boundaries are chosen
#: so each group answers one class of question completely — "what is the expense ratio" and
#: "what is the exit load" both land in Fees, never split across two chunks.
FACT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Fees and charges", (
        "Expense ratio (TER, direct plan)",
        "Base expense ratio (excl. additional fund expenses)",
        "Exit load",
        "Stamp duty",
        "Portfolio turnover ratio (%)",
    )),
    ("Minimum investments", (
        "Minimum SIP investment",
        "Minimum lump sum investment",
        "Minimum additional investment",
        "SIP amount multiplier",
        "Minimum redemption amount",
        "Purchase amount multiplier",
    )),
    ("Lock-in and availability", (
        "Lock-in period",
        "SIP available",
        "Lump sum available",
        "Closed for investment",
    )),
    ("Risk and benchmark", (
        "Riskometer level (as published on the page)",
        "Benchmark",
    )),
    ("Scheme identity", (
        "Scheme name",
        "Plan type",
        "Category",
        "Sub-category",
        "AMC",
        "Fund house",
        "Fund manager",
        "Launch date",
        "ISIN",
    )),
    ("NAV and fund size", (
        "NAV",
        "AUM",
    )),
    ("Scheme objective", (
        "Scheme objective",
    )),
)

#: Every declared fact field must land in exactly one group, or facts would go missing.
_GROUPED_LABELS: set[str] = {label for _, labels in FACT_GROUPS for label in labels}
_DECLARED_LABELS: set[str] = {f.label for f in SCHEME_FACT_FIELDS}
if _DECLARED_LABELS - _GROUPED_LABELS:  # pragma: no cover - guards a developer edit
    raise ValueError(
        "fact fields missing from FACT_GROUPS: "
        f"{sorted(_DECLARED_LABELS - _GROUPED_LABELS)}"
    )
if _GROUPED_LABELS - _DECLARED_LABELS:  # pragma: no cover - guards a developer edit
    raise ValueError(
        "FACT_GROUPS references unknown labels: "
        f"{sorted(_GROUPED_LABELS - _DECLARED_LABELS)}"
    )


# ── Shared helpers ───────────────────────────────────────────────────────────

_FACT_LINE_RE = re.compile(r"^-\s+(?P<label>[^:]+):\s*(?P<value>.+)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_STARTS_WITH_NUMBER_RE = re.compile(r"^\(?\s*\d|\d+(?:\.\d+)?\s?%")
#: Lines that are only scaffolding, never a standalone FAQ answer.
_TRIVIAL_PREFIXES = ("values published on the scheme page",)


def _make_chunk(
    doc: ProcessedDoc, section: str, body: str, strategy: str, kind: str = "prose"
) -> Chunk | None:
    """Build a chunk with the ``scheme — section`` prefix, or ``None`` if it is too small."""
    prefix = f"{doc.scheme or doc.title} — {section}"
    text = f"{prefix}\n{body.strip()}".strip()
    if not body.strip():
        return None
    return Chunk(
        chunk_id="",  # assigned in build_chunks, after dedupe
        text=text,
        source_url=doc.url,  # P1: mandatory, never empty
        title=doc.title,
        scheme=doc.scheme,
        category=doc.category,
        page_role=doc.page_role,
        section=section,
        char_len=len(text),
        fetched_at=doc.fetched_at,
        strategy=strategy,
        kind=kind,
    )


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )


def _split_headings(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into ``(heading, content)`` pairs.

    Content before the first heading gets the section name "Overview".
    """
    sections: list[tuple[str, list[str]]] = []
    current_name = "Overview"
    current: list[str] = []
    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((current_name, current))
            current_name, current = m.group(2).strip(), []
        else:
            current.append(line)
    sections.append((current_name, current))
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "\n".join(lines).strip()]


def _parse_fact_block(body: str) -> list[tuple[str, str]] | None:
    """Return ``(label, value)`` pairs if this body is a ``## Key facts`` block, else ``None``."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not any(ln.lstrip().startswith("- ") for ln in lines):
        return None
    facts: list[tuple[str, str]] = []
    for line in lines:
        m = _FACT_LINE_RE.match(line.strip())
        if m:
            facts.append((m.group("label").strip(), m.group("value").strip()))
    return facts or None


# ── Candidate A — recursive character splitting ──────────────────────────────


class RecursiveChunker:
    """Candidate A. Robust, never loses content, but blind to structure."""

    name = "a"

    def split(self, doc: ProcessedDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        for piece in _recursive_splitter().split_text(doc.body):
            chunk = _make_chunk(doc, "Text", piece, self.name)
            if chunk:
                chunks.append(chunk)
        return chunks


# ── Candidate B — structure-aware ────────────────────────────────────────────


class StructureAwareChunker:
    """Candidate B. Split on headings, recurse inside an oversized section."""

    name = "b"

    def split(self, doc: ProcessedDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        for heading, content in _split_headings(doc.body):
            if len(content) <= settings.chunk_size:
                chunk = _make_chunk(doc, heading, content, self.name)
                if chunk:
                    chunks.append(chunk)
                continue
            for piece in _recursive_splitter().split_text(content):
                chunk = _make_chunk(doc, heading, piece, self.name)
                if chunk:
                    chunks.append(chunk)
        return chunks


# ── Candidate C — fact-grouped (default) ─────────────────────────────────────


class FactGroupedChunker:
    """Candidate C, the default. Groups labelled facts, splits prose by heading.

    Tiers, applied per document:
      1. ``## Key facts`` block  -> group fields by :data:`FACT_GROUPS`
      2. any other section      -> heading-aware, recurse only if oversized
      3. anything still oversized -> recursive character fallback
    """

    name = "c"

    def split(self, doc: ProcessedDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        for heading, content in _split_headings(doc.body):
            facts = _parse_fact_block(content) if heading.lower().startswith("key facts") else None
            if facts:
                chunks.extend(self._group_facts(doc, facts))
                continue
            # Prose section: keep whole if it fits, otherwise recurse.
            if len(content) <= settings.chunk_size:
                chunk = _make_chunk(doc, heading, content, self.name)
                if chunk:
                    chunks.append(chunk)
                continue
            for piece in _recursive_splitter().split_text(content):
                chunk = _make_chunk(doc, heading, piece, self.name)
                if chunk:
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _group_facts(doc: ProcessedDoc, facts: list[tuple[str, str]]) -> list[Chunk]:
        """Emit one chunk per fact group, in declaration order, skipping empty groups."""
        by_label = dict(facts)
        chunks: list[Chunk] = []
        for group_name, labels in FACT_GROUPS:
            present = [(lb, by_label[lb]) for lb in labels if lb in by_label]
            if not present:
                continue
            body = "\n".join(f"- {lb}: {val}" for lb, val in present)
            section = f"Key facts: {group_name}"
            chunk = _make_chunk(doc, section, body, "c", kind="facts")
            if chunk:
                chunks.append(chunk)
        return chunks


CHUNKERS: dict[str, type] = {
    "a": RecursiveChunker,
    "b": StructureAwareChunker,
    "c": FactGroupedChunker,
}


# ── Post-processing shared by all candidates ─────────────────────────────────


def _normalise_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _rejoin_numeric_orphans(chunks: list[Chunk]) -> list[Chunk]:
    """Merge a chunk that begins with a bare number into its predecessor.

    A percentage or a fee figure must never be separated from the label that introduces it
    (PRD §7.3, numeric-block protection).
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        first_line = chunk.text.split("\n", 1)[-1].lstrip()
        starts_numeric = bool(_STARTS_WITH_NUMBER_RE.match(first_line))
        if merged and starts_numeric:
            prev = merged[-1]
            if prev.source_url == chunk.source_url:
                prev.text = f"{prev.text}\n{chunk.text.split(chr(10), 1)[-1]}"
                prev.char_len = len(prev.text)
                continue
        merged.append(chunk)
    return merged


#: Floor for a labelled fact group. Fact groups are complete answers to a question class by
#: construction, so they are exempt from the narrative minimum-length rule: "Risk and
#: benchmark" is 131 chars and "NAV and fund size" is 58, and dropping them would silently
#: lose the riskometer, the benchmark and the NAV. The 30-char floor only catches degenerate
#: output. Verified 2026-09-28 — before this exemption, Gate 1's facts were being discarded.
FACT_MIN_CHARS = 30


def _filter_chunks(chunks: Iterable[Chunk]) -> tuple[list[Chunk], int, int]:
    """Apply length rules, dedupe, and the provenance check. Returns ``(kept, dropped, deduped)``.

    Two different length regimes, because the two kinds of chunk serve different purposes:

    * **prose** — the minimum-length filter exists to drop nav crumbs, cookie banners and
      disclaimer tails, so it applies normally.
    * **facts** — a labelled fact group is a *complete* answer to a question class. Applying
      the prose thresholds to it discarded Minimum investments, Risk and benchmark, NAV and
      fund size, Lock-in and availability, and Scheme objective — i.e. every graded fact
      except fees and identity.
    """
    kept: list[Chunk] = []
    seen: dict[str, Chunk] = {}
    dropped = 0
    deduped = 0

    for chunk in chunks:
        if not chunk.source_url:
            # P1: a chunk without provenance is never emitted.
            dropped += 1
            continue
        body = chunk.text.split("\n", 1)[-1].strip()
        if not body or body.lower().startswith(_TRIVIAL_PREFIXES):
            dropped += 1
            continue

        if chunk.kind == "facts":
            if len(body) < FACT_MIN_CHARS:
                dropped += 1
                continue
        else:
            if len(body) < settings.chunk_hard_drop_chars:
                dropped += 1
                continue
            if len(body) < settings.chunk_min_chars and "?" not in body:
                # Short but not an FAQ answer — drop it.
                dropped += 1
                continue

        key = _normalise_for_hash(body)
        # Fact chunks are never deduped across pages. Two different schemes legitimately share
        # a value ("Moderately High" riskometer, "NIFTY 100 TRI" benchmark) but the fact is
        # *about that scheme*, and merging them left "Minimum SIP investment" appearing twice
        # in an 8-scheme corpus instead of 8 times. Dedupe exists for repeated prose
        # boilerplate, which is a narrative phenomenon, not a factual one.
        if chunk.kind != "facts":
            digest = hashlib.sha256(
                f"{chunk.page_role}|{key}".encode("utf-8")
            ).hexdigest()
            prior = seen.get(digest)
            if prior is not None:
                # Dedupe by content, but union the URLs so citation options survive (P1).
                if (
                    chunk.source_url not in prior.also_seen_at
                    and chunk.source_url != prior.source_url
                ):
                    prior.also_seen_at.append(chunk.source_url)
                deduped += 1
                continue
            seen[digest] = chunk
        kept.append(chunk)

    return kept, dropped, deduped


def build_chunks(strategy: Strategy | None = None) -> tuple[list[Chunk], dict]:
    """Chunk every processed document and write ``chunks.jsonl`` + ``chunks.txt``.

    Returns the chunk list and a stats dict for the FR-15 stage line.
    """
    strategy = strategy or settings.chunk_strategy
    chunker_cls = CHUNKERS.get(strategy)
    if chunker_cls is None:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(CHUNKERS)}")

    chunker = chunker_cls()
    raw_chunks: list[Chunk] = []
    docs = load_processed_docs()
    for doc in docs:
        raw_chunks.extend(chunker.split(doc))

    # Numeric-block protection runs within a document, before global dedupe.
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in raw_chunks:
        by_doc.setdefault(chunk.source_url, []).append(chunk)
    regrouped: list[Chunk] = []
    for url_chunks in by_doc.values():
        regrouped.extend(_rejoin_numeric_orphans(url_chunks))

    kept, dropped, deduped = _filter_chunks(regrouped)

    # Deterministic ids: doc order, then position. Re-running replaces, never duplicates.
    counters: dict[str, int] = {}
    for chunk in kept:
        slug = chunk.source_url.rstrip("/").split("/")[-1]
        idx = counters.get(slug, 0)
        counters[slug] = idx + 1
        chunk.chunk_id = f"{slug}#{idx:04d}"

    # P1 assertion: no chunk exists without a source_url.
    for chunk in kept:
        assert chunk.source_url, f"chunk {chunk.chunk_id} has no source_url"

    jsonl = settings.chunks_dir / "chunks.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for chunk in kept:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    _write_readable_dump(kept, strategy)

    lengths = [c.char_len for c in kept] or [0]
    stats = {
        "strategy": strategy,
        "chunks": len(kept),
        "dropped": dropped,
        "deduped": deduped,
        "mean_len": sum(lengths) // len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "docs": len(docs),
    }
    return kept, stats


def _write_readable_dump(chunks: list[Chunk], strategy: str) -> None:
    """Human-readable mirror of the chunk set — deliverable D8, and what Gate 1 inspects."""
    rule = "─" * 78
    out = [
        rule,
        f"CHUNK DUMP — strategy {strategy} · {len(chunks)} chunks",
        "Each chunk is prefixed with its scheme and section, so it is self-describing.",
        rule,
        "",
    ]
    for chunk in chunks:
        out += [
            rule,
            f"chunk_id : {chunk.chunk_id}",
            f"scheme   : {chunk.scheme or '-'}",
            f"role     : {chunk.page_role}",
            f"section  : {chunk.section}",
            f"source   : {chunk.source_url}",
            f"chars    : {chunk.char_len}",
        ]
        if chunk.also_seen_at:
            out.append(f"also at  : {', '.join(chunk.also_seen_at)}")
        out += [rule, chunk.text, ""]
    (settings.chunks_dir / "chunks.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    import sys

    choice = sys.argv[1] if len(sys.argv) > 1 else settings.chunk_strategy
    _, stats = build_chunks(choice)  # type: ignore[arg-type]
    print(
        f"STAGE 2 · strategy={stats['strategy']} · chunks {stats['chunks']} "
        f"· dropped_short {stats['dropped']} · deduped {stats['deduped']} "
        f"· mean_len {stats['mean_len']} · min {stats['min_len']} · max {stats['max_len']}"
    )
