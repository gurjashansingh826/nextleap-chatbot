"""Central configuration for the MF Facts RAG chatbot. Not a pipeline stage.

Every tunable in the system lives here (rule R4 / architecture §13). No other module may read
``os.environ`` directly — override anything via environment variables or the ``.env`` file.

The values that are *derived from measurement* rather than chosen up front are marked below:

* ``chunk_strategy`` — set to ``"c"`` (fact-grouped) from measurement on the real corpus,
  see the comment in ``Settings``; Phase 5's eval confirms or overturns it
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

ChunkStrategy = Literal["a", "b", "c"]


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
    # "c" = fact-grouped, CONFIRMED by Gate 2 measurement rather than assumed. On 30
    # in-scope queries, raw chunk retrieval scored recall@5 87% / citation accuracy 83% for
    # "c", against 83% / 77% for "a" (fixed-width recursive) and 73% / 80% for "b"
    # (heading-aware). On numeric facts — the fact type this corpus is mostly made of — "c"
    # scored 100% recall against 86% and 71%. See docs/eval_report.md.
    chunk_strategy: ChunkStrategy = "c"
    chunk_size: int = 700
    chunk_overlap: int = 100
    # Prose-only thresholds. Fact groups are exempt — see FACT_MIN_CHARS in chunkers.py.
    chunk_min_chars: int = 200
    chunk_hard_drop_chars: int = 80

    # ── STAGE 5 · retrieval ────────────────────────────────────────────────
    # min_score is NOT the out-of-scope control on this corpus, and Gate 2 is the reason.
    # The in-scope and out-of-scope score distributions OVERLAP: across 30 in-scope and 6
    # out-of-scope probes, the weakest in-scope top-1 was 0.603 while the strongest
    # out-of-scope probe reached 0.641. No threshold separates them, so this value drops
    # near-zero noise, it does not police relevance. Out-of-scope questions are rejected by
    # the entity filter (retriever._matches_filter) and by guards.py — deterministic checks
    # that do not depend on a similarity margin that does not exist here. It passes all 30
    # in-scope questions with wide margin (median 0.834).
    top_k: int = 6
    # How many chunks the extractive fallback scans, which is deliberately deeper than the
    # top_k the LLM is shown. The two need different pools: the model must be given a short,
    # focused context, while the fallback has to find the one group that actually mentions
    # the subject, and cosine similarity over short labelled bullets ranks that group almost
    # arbitrarily ("Who manages HDFC Large Cap Fund?" put "Fund manager: Prashant Jain" at
    # about 13th, behind four groups that say nothing about managers). Costs nothing — the
    # query is embedded once either way.
    extractive_pool: int = 12
    mmr_k: int = 4
    mmr_lambda: float = 0.7
    # Conversational memory: how many recent exchanges may inform query resolution. A scope
    # addition on request — the PRD excludes saved chat history, and this is not that: the
    # window lives in st.session_state, is dropped when the tab closes, and holds only
    # (question, slug, mode), never answer text. 10 is well past the 2-3 a follow-up needs;
    # the bound exists so the window cannot grow into an unbounded transcript.
    memory_turns: int = 10
    min_score: float = 0.25
    scheme_boost: float = 0.0  # superseded by the deterministic entity filter (ADR-14)

    # ── STAGE 6 · answering ────────────────────────────────────────────────
    groq_api_key: str = ""
    # The model must be one THIS KEY can reach — Groq's catalogue varies by account, and an
    # unreachable model returns 404 model_not_found, not an auth error. The value this
    # project originally shipped with, "llama-3.3-70b-versatile", 404s on the key it was
    # built against; see the warning in env.py and README for how to list your own.
    #
    # 20b rather than 120b because this is a live demo and the free tier allows 8000 tokens
    # per minute: measured, 120b answers in 8-16s and trips the rate limit on a handful of
    # questions, silently downgrading to the extractive path, while 20b answers the same
    # questions correctly in 1.4-2.0s. The guard and filter stages do the safety work and the
    # model only writes prose, so the larger model buys latency and nothing else. Set
    # MF_RAG_LLM_MODEL=openai/gpt-oss-120b if you would rather have it.
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0  # determinism (NFR-5)
    llm_max_tokens: int = 250
    max_sentences: int = 3  # hard cap from the brief
    factsheet_link: str = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    education_link: str = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"

    # ── STAGE 1 · loading ──────────────────────────────────────────────────
    http_timeout: int = 30
    http_retries: int = 3
    fetch_delay_seconds: float = 1.5
    # No fake contact address here on purpose. If your course or org requires a contact in
    # the UA string, set MF_RAG_USER_AGENT in .env to something like:
    #   "mf-facts-rag-demo/1.0 (+class project; you@university.edu)"
    user_agent: str = "mf-facts-rag-demo/1.0 (class project; non-commercial; single-user)"

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

#: The ``.env`` this process actually reads. Exposed so a diagnostic can print the exact
#: path rather than making someone go and find it — the most common cause of "I set the key
#: and nothing happened" is having edited a second copy of the file.
ENV_PATH = ROOT / ".env"

#: The line in ``.env.example`` holding the key, so the error message can point at a line
#: number instead of making the reader search the file.
KEY_LINE_NUMBER = next(
    (
        i
        for i, line in enumerate(
            (ROOT / ".env.example").read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip().startswith("MF_RAG_GROQ_API_KEY")
    ),
    7,
)

if not settings.groq_api_key:
    # Loud, specific, and at import time — because the alternative is a silent downgrade to
    # the extractive path that looks like a working demo. "Answers are just less fluent"
    # is not a symptom anyone would report, which is how a missing key survives eight
    # verification rounds. It names the file, the line and the fix.
    import logging

    logging.getLogger("mf_rag").warning(
        "No Groq API key loaded. Answers will come from the deterministic extractive "
        "fallback instead of the LLM — fully functional, but not the configured path.\n"
        "  file read : %s\n"
        "  edit line : %d  (MF_RAG_GROQ_API_KEY=<your gsk_... key>)\n"
        "  or run    : .venv\\Scripts\\python.exe tools\\set_groq_key.py\n"
        "  diagnose  : powershell -ExecutionPolicy Bypass -File tools\\check_key.ps1",
        ENV_PATH,
        KEY_LINE_NUMBER,
    )


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
