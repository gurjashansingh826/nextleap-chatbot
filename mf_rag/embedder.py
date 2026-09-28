"""STAGE 3 — Embedding: the only path from text to a 384-dim vector.

Model: ``sentence-transformers/all-MiniLM-L6-v2`` — a HuggingFace sentence-embedding model,
384-dim, ~80 MB, CPU-only, no API key and no cost. Chosen by the brief.

**Why this module talks to ``transformers`` + ``torch`` directly**
The obvious implementation is ``from sentence_transformers import SentenceTransformer``.
On this machine that import fails: ``sentence_transformers/__init__.py`` eagerly pulls in its
own ``evaluation`` module, which imports ``sklearn.metrics``, whose compiled extension
``_expected_mutual_info_fast`` is blocked by a Windows Application Control policy. That is a
deliberate security control and is not something to work around.

scikit-learn is genuinely irrelevant to embedding a sentence, so the fix is to do what
``sentence-transformers`` does for this model and no more:

    tokenize (truncate to 256) → BERT encoder → attention-mask mean pooling → L2 normalise

That recipe is not assumed. It is read from the model repo's own config at load time
(``1_Pooling/config.json`` and ``sentence_bert_config.json``) and asserted, so if the model is
ever swapped for one that uses CLS or max pooling, this fails loudly instead of silently
producing wrong vectors. A convenience-only wrapper library became a hard dependency for no
functional gain.

**P6 — this module is the only embedding entry point in the codebase.** Corpus chunks and user
queries must go through the same model, the same pooling and the same normalisation. The
classic silent RAG bug is a corpus embedded one way and queries another: retrieval still
"works", it just quietly returns nonsense, and nothing errors. Do not add a second embedding
path, and do not let any other module load the model directly.

**Normalisation.** Vectors are L2-normalised, so cosine similarity equals the dot product and
a distance of 0 means identical text. That is what lets ``store.search`` read Chroma's cosine
distance as ``1 - similarity``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import settings

#: Populated on first load.
_TOKENIZER = None
_MODEL = None
_RECIPE: dict = {}
_LOAD_SECONDS = 0.0

#: The pooling this module implements. Checked against the model repo at load time. All four
#: keys must be listed: an absent key would compare as a mismatch on a dict that declares it.
_EXPECTED_RECIPE = {
    "pooling_mode_mean_tokens": True,
    "pooling_mode_cls_token": False,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
}


def _read_recipe(model_id: str) -> dict:
    """Read the sentence-transformers recipe that ships with the model repo.

    Returns the pooling mode, the sequence limit and the embedding width, so the
    implementation is driven by the model's own declaration rather than by recollection.
    """
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(model_id, allow_patterns=["*.json"]))
    pooling_file = root / "1_Pooling" / "config.json"
    st_file = root / "sentence_bert_config.json"
    modules_file = root / "modules.json"
    if not pooling_file.is_file():
        raise RuntimeError(
            f"{model_id} does not ship 1_Pooling/config.json, so the pooling recipe cannot "
            f"be verified. Refusing to guess a pooling mode — an unverified one silently "
            f"degrades every retrieval score."
        )
    pooling = json.loads(pooling_file.read_text(encoding="utf-8"))
    limits = json.loads(st_file.read_text(encoding="utf-8")) if st_file.is_file() else {}

    # Read normalisation from modules.json, not from the presence of a 2_Normalize directory:
    # the Normalize module has no config file, so an *.json-only snapshot never creates it.
    modules = json.loads(modules_file.read_text(encoding="utf-8")) if modules_file.is_file() else []
    normalize = any(str(m.get("type", "")).endswith(".Normalize") for m in modules)

    return {
        "pooling": pooling,
        "max_seq_length": int(limits.get("max_seq_length", 256)),
        "dim": int(pooling.get("word_embedding_dimension", settings.embed_dim)),
        "normalize": normalize,
        "modules": [m.get("type", "") for m in modules],
    }


def get_model():
    """Load tokenizer + encoder once per process and reuse them (NFR-2).

    Lazy rather than import-time so that ``import mf_rag.chunkers`` stays instant and the
    model load is paid only by commands that actually embed.
    """
    global _TOKENIZER, _MODEL, _RECIPE, _LOAD_SECONDS
    if _MODEL is not None:
        return _TOKENIZER, _MODEL

    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    _RECIPE = _read_recipe(settings.embed_model)

    # Verify rather than assume: this module implements mean pooling + L2 normalise, nothing
    # else. If the model's declared recipe differs, embedding it this way would produce
    # vectors that retrieve nonsense, and no exception would ever be raised.
    declared = {k: bool(v) for k, v in _RECIPE["pooling"].items() if k.startswith("pooling_")}
    if declared != _EXPECTED_RECIPE:
        raise RuntimeError(
            f"{settings.embed_model} declares pooling {declared}, but embedder.py implements "
            f"{_EXPECTED_RECIPE}. Update the pooling in this module to match the model, or "
            f"set MF_RAG_EMBED_MODEL to a mean-pooled model."
        )
    if not _RECIPE["normalize"]:
        raise RuntimeError(
            f"{settings.embed_model} does not declare a Normalize module. This module always "
            f"L2-normalises, which the rest of the pipeline depends on for cosine == dot."
        )
    if _RECIPE["dim"] != settings.embed_dim:
        raise RuntimeError(
            f"model width {_RECIPE['dim']} != settings.embed_dim {settings.embed_dim}. "
            f"Set MF_RAG_EMBED_DIM to match, or the index will be built at the wrong width."
        )

    _TOKENIZER = AutoTokenizer.from_pretrained(settings.embed_model)
    _MODEL = AutoModel.from_pretrained(settings.embed_model)
    _MODEL.eval()  # disables dropout, so the same text always embeds identically
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    _LOAD_SECONDS = time.perf_counter() - started
    return _TOKENIZER, _MODEL


def embed(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed a batch of strings. Used for BOTH chunks and queries (P6).

    Attention-mask-weighted mean pooling over the last hidden state, then L2 normalisation —
    the recipe the model repo declares (see module docstring).
    """
    if not texts:
        return []
    import torch
    import torch.nn.functional as F

    tokenizer, model = get_model()

    encoded = tokenizer(
        texts,
        padding=True,              # pad to the longest item in the batch
        truncation=True,
        max_length=_RECIPE["max_seq_length"],
        return_tensors="pt",
    )

    with torch.no_grad():  # inference only; no graph is needed and memory is wasted if built
        output = model(**encoded)

    hidden = output.last_hidden_state                      # (batch, tokens, dim)
    mask = encoded["attention_mask"].unsqueeze(-1)         # (batch, tokens, 1)

    # Sum only real tokens, then divide by how many there were. Without the mask, padding
    # tokens would be averaged into the vector and dilute every embedding by batch shape.
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)               # guard a fully-masked row
    pooled = summed / counts

    vectors = F.normalize(pooled, p=2, dim=1)              # unit length => cosine == dot

    assert vectors.shape[1] == settings.embed_dim, (
        f"embedding dim mismatch: produced {vectors.shape[1]}, "
        f"config expects {settings.embed_dim}"
    )
    if show_progress:
        print(f"  embedded {len(vectors)} texts at dim {vectors.shape[1]}")
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    """Embed a single string. Same model, same path (P6)."""
    return embed([text])[0]


def model_info() -> dict:
    """Describe the loaded model and the recipe in force. Used by the STAGE 3 log line."""
    get_model()
    return {
        "model": settings.embed_model,
        "dim": _RECIPE["dim"],
        "expected_dim": settings.embed_dim,
        "max_seq_length": _RECIPE["max_seq_length"],
        "pooling": "mean + L2 normalise",
        "normalize": _RECIPE["normalize"],
        "load_seconds": round(_LOAD_SECONDS, 2),
    }


if __name__ == "__main__":
    info = model_info()
    print(
        f"STAGE 3 · model={info['model']} · dim={info['dim']} · "
        f"pooling={info['pooling']} · max_seq={info['max_seq_length']} · "
        f"load={info['load_seconds']}s"
    )
    probe = embed_one("expense ratio of HDFC Large Cap Fund")
    norm = sum(x * x for x in probe) ** 0.5
    print(f"probe: {len(probe)} floats · L2 norm {norm:.6f} (must be 1.0)")
    assert len(probe) == info["dim"] == settings.embed_dim
    assert abs(norm - 1.0) < 1e-5, "vector is not normalised"

    # A second, independent check that the model is doing semantic work rather than hashing:
    # a paraphrase must land closer to the query than an unrelated sentence does.
    import math

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    q = embed_one("What is the expense ratio of HDFC Large Cap Fund?")
    near = embed_one("The expense ratio is 1.03% for the direct plan.")
    far = embed_one("The weight of an elephant is approximately five tonnes.")
    print(f"cosine(query, paraphrase)   = {cos(q, near):.4f}")
    print(f"cosine(query, unrelated)    = {cos(q, far):.4f}")
    assert cos(q, near) > cos(q, far), "embeddings do not capture meaning"
    print("OK - embeddings are semantic and normalised.")
