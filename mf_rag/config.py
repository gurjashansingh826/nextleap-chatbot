"""Central configuration for the MF Facts RAG chatbot. Not a pipeline stage.

Every tunable in the system lives here (rule R4 / architecture §13). No other module may read
``os.environ`` directly — override anything via environment variables or the ``.env`` file.

The values that are *derived from measurement* rather than chosen up front are marked below:

* ``chunk_strategy`` — decided in Phase 5 by scoring Candidates A and B
* ``min_score``      — calibrated in Phase 5 against the observed score distribution

Both are provisional defaults so the system runs before the eval exists; Phase 5 must
overwrite them with evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# mf_rag/config.py -> mf_rag/ -> <repo root>
ROOT = Path(__file__).resolve().parent.parent

ChunkStrategy = Literal["a", "b"]


class Settings(BaseSettings):
    """Settings, overridable by environment variable or ``.env``."""

    # env_prefix namespaces every override as MF_RAG_*. Without it, generic names like
    # TOP_K or MIN_SCORE could be silently picked up from an unrelated system variable and
    # change retrieval behaviour with no visible cause.
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="MF_RAG_", extra="ignore"
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    chunks_dir: Path = ROOT / "data" / "chunks"
    chroma_path: Path = ROOT / "chroma"
    eval_dir: Path = ROOT / "eval"
    docs_dir: Path = ROOT / "docs"

    # ── STAGE 3 · embedding (fixed by the brief) ───────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    # ── STAGE 4 · vector store ─────────────────────────────────────────────
    chroma_collection: str = "mf_facts_hdfc"
    chroma_space: str = "cosine"

    # ── STAGE 2 · chunking ─────────────────────────────────────────────────
    # chunk_strategy is PROVISIONAL — Phase 5 eval sets the real value.
    chunk_strategy: ChunkStrategy = "b"
    chunk_size: int = 700
    chunk_overlap: int = 100
    chunk_min_chars: int = 200
    chunk_hard_drop_chars: int = 80

    # ── STAGE 5 · retrieval ────────────────────────────────────────────────
    # min_score is PROVISIONAL — Phase 5 calibration sets the real value
    # (architecture §9.4). It is the fabrication gate: chunks below it are dropped and the
    # bot takes the "not in my sources" path instead of guessing.
    top_k: int = 6
    mmr_k: int = 4
    mmr_lambda: float = 0.7
    min_score: float = 0.25
    scheme_boost: float = 0.0  # off until the eval shows cross-scheme confusion (§9.5)

    # ── STAGE 6 · answering ────────────────────────────────────────────────
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0  # determinism (NFR-5)
    llm_max_tokens: int = 250
    max_sentences: int = 3  # hard cap from the brief
    factsheet_link: str = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    education_link: str = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"

    # ── STAGE 1 · loading ──────────────────────────────────────────────────
    http_timeout: int = 30
    http_retries: int = 3
    fetch_delay_seconds: float = 1.5
    # Replace the placeholder contact before the live fetch (see guide, Phase 2).
    user_agent: str = (
        "mf-facts-rag-demo/1.0 (+class project; contact: student@example.com)"
    )

    # ── Thresholds for "page fetched but probably empty" (Gate 1) ──────────
    low_text_threshold: int = 500

    def ensure_dirs(self) -> None:
        """Create the data directory tree. Idempotent."""
        for d in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.chunks_dir,
            self.eval_dir,
            self.docs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()


def _inventory() -> str:
    """Human-readable config dump for `python -m mf_rag.config`."""
    lines = ["CONFIG", "-" * 60]
    for field, value in settings.model_dump().items():
        rendered = value
        if isinstance(value, Path):
            try:
                rendered = str(value.relative_to(ROOT))
            except ValueError:
                rendered = str(value)
        lines.append(f"  {field:<24} {rendered}")
    lines.append("-" * 60)
    lines.append(f"  derived: EMBED_MODEL   {settings.embed_model} ({settings.embed_dim}-dim)")
    lines.append(f"  derived: SRC_ROOT      {ROOT}")
    lines.append("-" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print(_inventory())
