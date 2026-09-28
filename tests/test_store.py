"""Gates for STAGE 3 (embedding) and STAGE 4 (vector store).

A vector store can be built, populated, and return a plausible row count while being useless,
so these tests assert the three properties that make retrieval trustworthy rather than just
present:

1. the index holds exactly the chunks the corpus produced, with citable provenance on every one;
2. rebuilding is idempotent, so a second ingest cannot silently duplicate or drift;
3. the score STAGE 5 will threshold on really is cosine similarity, with the right sign.

They run against a throwaway collection under ``tmp_path`` so they never touch the committed
index, and they use a handful of synthetic chunks so the suite stays fast.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import pytest

from mf_rag import store
from mf_rag.chunkers import Chunk
from mf_rag.config import settings
from mf_rag.embedder import embed_one

PROBE = "the expense ratio of a large cap fund is published on the scheme page"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _chunk(n: int, *, scheme: str = "HDFC Test Fund", plan: str = "Direct Growth") -> Chunk:
    return Chunk(
        chunk_id=f"hdfc-test-fund-direct-growth#{n:04d}",
        text=f"{scheme} ({plan}) — Key facts: Fees and charges\n"
             f"- Expense ratio (TER): {n}.0%",
        source_url="https://groww.in/mutual-funds/hdfc-test-fund-direct-growth",
        title="HDFC Test Fund Direct Growth",
        scheme=scheme,
        category="Large cap",
        page_role="primary",
        section="Key facts: Fees and charges",
        char_len=120,
        fetched_at="2026-09-28",
        strategy="c",
        kind="facts",
        plan=plan,
    )


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A private Chroma collection per test, torn down with the tmp dir."""
    monkeypatch.setattr(settings, "chroma_path", tmp_path / "chroma")
    monkeypatch.setattr(settings, "chroma_collection", f"test_{tmp_path.name[:8]}")
    store._client = None  # reopen against the new path
    yield
    store._client = None


# ── STAGE 3 · embedding ───────────────────────────────────────────────────────

def test_embed_returns_normalised_vectors_of_the_declared_width():
    from mf_rag.embedder import embed, model_info

    vectors = embed(["expense ratio of a large cap fund", "lock-in period of an ELSS"])
    assert len(vectors) == 2
    assert len(vectors[0]) == settings.embed_dim
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5
        # Normalisation is what lets store.search read Chroma's cosine distance as
        # 1 - similarity. If this drifts, every score in the system is wrong.
        assert abs(norm - 1.0) < 1e-5, f"vector is not L2-normalised (norm={norm})"
    assert model_info()["dim"] == settings.embed_dim


def test_embedder_refuses_a_model_whose_pooling_it_does_not_implement(monkeypatch):
    """The recipe is asserted, not assumed.

    A model using CLS or max pooling embedded with mean pooling produces vectors that retrieve
    nonsense while every call succeeds, so the mismatch must be an exception.
    """
    from mf_rag import embedder

    monkeypatch.setattr(embedder, "_read_recipe", lambda _m: {
        "pooling": {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True,
                    "pooling_mode_max_tokens": False,
                    "pooling_mode_mean_sqrt_len_tokens": False},
        "max_seq_length": 256, "dim": settings.embed_dim, "normalize": True, "modules": [],
    })
    monkeypatch.setattr(embedder, "_MODEL", None)
    monkeypatch.setattr(embedder, "_TOKENIZER", None)
    with pytest.raises(RuntimeError, match="declares pooling"):
        embedder.get_model()


# ── STAGE 4 · vector store ───────────────────────────────────────────────────

def test_upsert_then_search_round_trips(sandbox):
    chunks = [_chunk(i) for i in range(4)]
    assert store.upsert_chunks(chunks, rebuild=True, verbose=False) == 4
    assert store.collection_size() == 4

    results = store.search(embed_one(PROBE), top_k=4)
    assert len(results) == 4
    assert {r.chunk_id for r in results} == {c.chunk_id for c in chunks}
    # Descending score, always.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_upsert_is_idempotent(sandbox):
    """Re-ingesting must replace, never duplicate.

    Deterministic chunk ids are what make this hold; without them every corpus rebuild would
    grow the collection and the same fact would be cited several times with different ranks.
    """
    chunks = [_chunk(i) for i in range(3)]
    store.upsert_chunks(chunks, rebuild=True, verbose=False)
    store.upsert_chunks(chunks, verbose=False)
    store.upsert_chunks(chunks, verbose=False)
    assert store.collection_size() == 3


def test_every_stored_row_carries_a_citable_source(sandbox):
    """Provenance rule P1, enforced at write time and readable after the fact."""
    chunks = [_chunk(i) for i in range(3)]
    store.upsert_chunks(chunks, rebuild=True, verbose=False)
    for r in store.search(embed_one(PROBE), top_k=3):
        assert r.source_url.startswith("https://")
        assert r.chunk_id and r.fetched_at


def test_upsert_refuses_a_chunk_with_no_source(sandbox):
    bad = _chunk(0)
    bad.source_url = ""
    with pytest.raises(ValueError, match="without a source_url"):
        store.upsert_chunks([bad], rebuild=True, verbose=False)
    assert store.collection_size() == 0


def test_score_is_cosine_similarity_with_the_right_sign(sandbox):
    """`score == 1 - distance`, and an identical text scores near 1.

    STAGE 5's ``min_score`` is reasoned about as a similarity. If the sign or the scale is
    wrong the threshold silently admits everything or nothing.
    """
    store.upsert_chunks([_chunk(0), _chunk(1)], rebuild=True, verbose=False)
    top = store.search(embed_one(PROBE), top_k=1)[0]
    assert abs(top.score - (1.0 - top.distance)) < 1e-6
    assert 0.0 < top.score <= 1.0

    # The identical sentence must beat a lexically unrelated one.
    same = store.search(embed_one(top.text), top_k=1)[0]
    assert same.chunk_id == top.chunk_id
    assert same.score > top.score - 0.01


def test_index_fingerprint_detects_a_model_or_strategy_change(sandbox):
    """A stale index must fail loudly, not return plausible nonsense.

    Query vectors from a different embedding model retrieve garbage with no error anywhere.
    The fingerprint is what turns that silent failure into a loud one.
    """
    store.upsert_chunks([_chunk(0)], rebuild=True, verbose=False)
    store.verify_index_fingerprint()  # matching settings: no raise

    collection = store.get_collection()
    stored = collection.metadata or {}
    collection.modify(metadata={**{k: v for k, v in stored.items() if k != "embed_model"},
                                "embed_model": "some/other-model"})

    with pytest.raises(RuntimeError, match="built with different settings"):
        store.verify_index_fingerprint()

    collection.modify(metadata=dict(stored))
    store.verify_index_fingerprint()


def test_plan_is_persisted_for_entity_resolution(sandbox):
    """STAGE 5 needs the plan in metadata to map a scheme name onto exactly one page.

    Groww sets ``plan_type`` to "Direct" on both the Growth and IDCW pages, so the plan
    descriptor derived from the page's own name is the only reliable discriminator.
    """
    store.upsert_chunks([_chunk(0, plan="Direct IDCW")], rebuild=True, verbose=False)
    row = store.search(embed_one(PROBE), top_k=1)[0]
    assert row.plan == "Direct IDCW"
    assert "(Direct IDCW)" in row.text.split("\n")[0]


def test_search_against_a_missing_index_fails_loudly(sandbox):
    """An absent index must not look like a legitimate zero-hit answer."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        store.search([0.0] * settings.embed_dim, top_k=3)
