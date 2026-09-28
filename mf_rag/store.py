"""STAGE 4 — Vector Store: ChromaDB, persisted to disk, idempotent upserts.

Why ChromaDB (ADR-8): the corpus is 15 pages / ~107 chunks. A Postgres+pgvector or Qdrant
deployment would be speculative generality. Chroma is local, free, zero-config, and persists
to disk, which is what makes the "ingest once, don't re-embed on every restart" requirement
(P10) achievable without any infrastructure.

Three properties this module is responsible for:

1. **Persistence (P10).** ``PersistentClient`` writes to ``chroma/``, so the index survives
   process restarts and the model loads only when a query actually needs it.
2. **Idempotency (FR-3).** ``chunk_id`` is deterministic (``{slug}#{index:04d}``), so
   re-embedding *replaces* rather than duplicates. Running the ingest twice must leave the
   collection size unchanged — asserted by ``tests/test_store.py``.
3. **A fixed distance metric.** Collection metadata declares ``{"hnsw:space": "cosine"}``
   once, at creation. Changing the metric later would silently invalidate every stored
   distance without any error, which is the failure this line exists to prevent.

**Metadata does not contain ``text``.** Chroma already stores the chunk body as ``documents``;
copying it into metadata would double the stored payload for no benefit. See architecture §7.4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import settings

#: Declared once, at collection creation. Never mutated afterwards (see module docstring).
COLLECTION_METADATA: dict[str, str] = {"hnsw:space": settings.chroma_space}

#: Fields mirrored into Chroma metadata. Deliberately excludes ``text`` and ``also_seen_at``
#: (a list, which Chroma metadata cannot hold).
METADATA_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "source_url",
    "title",
    "scheme",
    "category",
    "page_role",
    "plan",
    "section",
    "char_len",
    "fetched_at",
)

_client = None


@dataclass
class ScoredChunk:
    """A retrieved chunk with its similarity score.

    ``score`` is cosine *similarity* in ``[-1, 1]``, already converted from Chroma's cosine
    *distance*. Higher is better. Keeping the sign convention right here means STAGE 5's
    ``min_score`` threshold is a similarity, matching how the value is reasoned about in
    architecture §9.4.
    """

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
    score: float
    distance: float
    plan: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_url": self.source_url,
            "title": self.title,
            "scheme": self.scheme,
            "category": self.category,
            "page_role": self.page_role,
            "plan": self.plan,
            "section": self.section,
            "char_len": self.char_len,
            "fetched_at": self.fetched_at,
            "score": round(self.score, 4),
            "distance": round(self.distance, 4),
        }


def get_client():
    """Open (or reuse) the persistent Chroma client. Creates ``chroma/`` on first use."""
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str | None = None, create: bool = True):
    """Get the collection by name, creating it with the fixed metric if it is absent.

    ``create=False`` is for read-only paths (STAGE 5 retrieval) so that a missing index
    raises a clear "run embed first" error rather than silently creating an empty one that
    would look like a legitimate zero-result retrieval.
    """
    name = name or settings.chroma_collection
    client = get_client()
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        return client.get_collection(name)
    if not create:
        raise FileNotFoundError(
            f"Chroma collection {name!r} does not exist at {settings.chroma_path}. "
            f"Run `python -m mf_rag.embed_index` to build the index first."
        )
    return client.create_collection(name=name, metadata=COLLECTION_METADATA)


def _clean_metadata(chunk) -> dict[str, Any]:
    """Project a Chunk onto Chroma's metadata rules.

    Chroma accepts only str/int/float/bool, rejects ``None``, and rejects lists. Non-scheme
    pages legitimately have an empty ``scheme``/``category``, so those become ``""`` rather
    than being dropped — a missing key would break the Stage 5 filters that read them.
    """
    meta: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = getattr(chunk, field, None)
        if field == "char_len":
            meta[field] = int(value or 0)
        else:
            meta[field] = "" if value is None else str(value)
    return meta


def collection_size(name: str | None = None) -> int:
    """Number of chunks in the collection. 0 if the collection does not exist yet."""
    name = name or settings.chroma_collection
    client = get_client()
    if name not in {c.name for c in client.list_collections()}:
        return 0
    return int(get_collection(name).count())


def upsert_chunks(chunks: Sequence, rebuild: bool = False, verbose: bool = True) -> int:
    """Embed chunks and upsert them into Chroma. Returns the number written.

    ``rebuild=True`` drops and recreates the collection, which is the only way to change the
    distance metric or the embedding model — see the warning on :data:`COLLECTION_METADATA`.

    Batched to keep peak memory flat. The batch size is a Chroma/practical limit, not a
    tuning knob: passing 107 vectors at once is fine, but a 5,000-chunk corpus should not
    try.
    """
    from .embedder import embed  # imported here to keep STAGE 3 optional at import time

    if not chunks:
        raise ValueError("upsert_chunks called with no chunks — run the chunker first")

    name = settings.chroma_collection
    client = get_client()

    if rebuild:
        if name in {c.name for c in client.list_collections()}:
            client.delete_collection(name)
        collection = client.create_collection(name=name, metadata=COLLECTION_METADATA)
    else:
        collection = get_collection(name)

    before = collection.count()
    texts = [c.text for c in chunks]

    started = time.perf_counter()
    vectors = embed(texts)
    embed_seconds = time.perf_counter() - started

    if verbose:
        print(
            f"STAGE 3 · model={settings.embed_model} · dim={settings.embed_dim} · "
            f"encoded {len(vectors)} in {embed_seconds:.2f}s"
        )

    ids = [c.chunk_id for c in chunks]
    metadatas = [_clean_metadata(c) for c in chunks]

    # P1: refuse to index anything that cannot be cited.
    missing = [i for i, m in zip(ids, metadatas) if not m["source_url"]]
    if missing:
        raise ValueError(f"refusing to index chunks without a source_url: {missing[:5]}")

    batch = 256
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start:start + batch],
            documents=texts[start:start + batch],
            metadatas=metadatas[start:start + batch],
            embeddings=vectors[start:start + batch],
        )

    after = collection.count()
    # Stamp what the index was built from, here rather than at the call site, so the
    # fingerprint cannot be skipped by a programmatic caller. verify_index_fingerprint()
    # reads this at query time to catch a model or chunking-strategy change.
    #
    # `hnsw:space` is deliberately excluded: Chroma rejects a modify() that mentions it,
    # treating it as an attempt to change the distance function of a live collection.
    collection.modify(metadata=dict(index_fingerprint()))

    if verbose:
        print(
            f"STAGE 4 · collection={name} · upserted {len(ids)} · total {after} "
            f"(was {before}{', rebuilt' if rebuild else ''})"
        )
    return len(ids)


def search(
    query_vec: list[float],
    top_k: int | None = None,
    name: str | None = None,
) -> list[ScoredChunk]:
    """Return the ``top_k`` nearest chunks, highest similarity first.

    With ``hnsw:space:cosine`` and L2-normalised vectors, Chroma returns a *distance* where
    ``similarity = 1 - distance``. Converting here keeps ``min_score`` a similarity
    everywhere else in the system.
    """
    top_k = top_k or settings.top_k
    collection = get_collection(name or settings.chroma_collection, create=False)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    scored: list[ScoredChunk] = []
    for chunk_id, doc, meta, distance in zip(ids, docs, metas, dists):
        meta = meta or {}
        scored.append(
            ScoredChunk(
                chunk_id=chunk_id,
                text=doc or "",
                source_url=meta.get("source_url", ""),
                title=meta.get("title", ""),
                scheme=meta.get("scheme", ""),
                category=meta.get("category", ""),
                page_role=meta.get("page_role", ""),
                section=meta.get("section", ""),
                char_len=int(meta.get("char_len") or 0),
                fetched_at=meta.get("fetched_at", ""),
                score=1.0 - float(distance),
                distance=float(distance),
                plan=meta.get("plan", ""),
            )
        )
    return scored


def drop_collection(name: str | None = None) -> None:
    """Delete the collection. Used by ``rebuild`` and by the test for a clean rebuild."""
    name = name or settings.chroma_collection
    client = get_client()
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)


def index_fingerprint(name: str | None = None) -> dict:
    """What the index was built from. Persisted as collection metadata.

    Two things silently corrupt retrieval if they change without a rebuild: the embedding
    model and the chunking strategy. A query vector from a different model, or a query
    against chunks from a different strategy, returns plausible-looking nonsense. Recording
    both on the collection makes the mismatch detectable instead.
    """
    return {"embed_model": settings.embed_model, "chunk_strategy": settings.chunk_strategy}


def verify_index_fingerprint(strict: bool = True) -> None:
    """Raise if the stored index was built with a different model or chunking strategy.

    Called at the start of every query. This is the check that turns a silent failure into a
    loud one.
    """
    collection = get_collection(create=False)
    stored = collection.metadata or {}
    current = index_fingerprint()

    problems = [
        f"{key}: index={stored.get(key)!r} current={value!r}"
        for key, value in current.items()
        if stored.get(key) not in (None, value)
    ]
    if not problems:
        return
    message = (
        "The vector index was built with different settings than the current config:\n  "
        + "\n  ".join(problems)
        + f"\nRe-run the embed step to rebuild {settings.chroma_collection!r}."
    )
    if strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}")


if __name__ == "__main__":
    import sys

    from .chunkers import build_chunks

    rebuild = "--rebuild" in sys.argv
    import json

    # Read the chunk file from disk rather than re-chunking: this stage must never silently
    # build the index from a different strategy than the one whose output was reviewed.
    path = settings.chunks_dir / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python -m mf_rag.chunkers` first.")

    from .chunkers import Chunk

    chunks = [Chunk(**json.loads(line)) for line in
              path.read_text(encoding="utf-8").splitlines() if line.strip()]
    upsert_chunks(chunks, rebuild=rebuild)
    print(f"index fingerprint: {index_fingerprint()}")
    print(f"collection_size: {collection_size()}")
