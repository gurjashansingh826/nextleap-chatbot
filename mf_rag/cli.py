"""Command line entry point — ``python -m mf_rag.cli <command>``.

Exists so the whole pipeline can be demonstrated from a terminal, which is what turns the
demo from "a Streamlit app that happens to work" into "a pipeline whose every stage is
visible and countable". The acceptance command is ``all``.

Two rules govern error handling here, and they are the reason this module is more than
argument parsing:

* **Never print a traceback.** Every anticipated failure — index not built, corpus not
  ingested, no API key — gets a one-line message naming the command that fixes it. A demo
  that dies with a stack trace teaches the audience nothing except that it crashed.
* **Print one line per stage with real counts.** The shape is fixed
  (``STAGE n · label value · label value``) so the numbers can be read aloud, but every value
  is measured at run time. Nothing here is hard-coded, because a hard-coded count in a demo
  is a claim rather than evidence.

Guards run before retrieval in ``query``, and that ordering is the whole point of the
``query`` command existing separately from ``app``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import settings

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s · %(message)s"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


def _fail(message: str, remedy: str) -> None:
    """One-line error + the command that fixes it. Never a traceback."""
    print(f"ERROR: {message}", file=sys.stderr)
    print(f"  fix: {remedy}", file=sys.stderr)
    raise SystemExit(1)


# ── STAGE 1 · ingest ─────────────────────────────────────────────────────────


def cmd_ingest(args: argparse.Namespace) -> None:
    from .loaders import ingest_all

    if not settings.processed_dir.exists() or not any(settings.processed_dir.glob("*.md")):
        _fail("no processed corpus found", "run `python -m mf_rag.cli ingest` first")

    summary = ingest_all(refresh=args.refresh, verbose=True)
    print(
        f"STAGE 1 · pages requested {summary.requested} · fetched {summary.fetched} · "
        f"cached {summary.cached} · failed {summary.failed} · low_text {summary.low_text}"
    )


# ── STAGE 2 · chunk ──────────────────────────────────────────────────────────


def _require_corpus() -> None:
    docs = list(settings.processed_dir.glob("*.md"))
    if not docs:
        _fail(
            f"no processed documents in {settings.processed_dir}",
            "run `python -m mf_rag.cli ingest` first",
        )


def cmd_chunk(args: argparse.Namespace) -> None:
    from .chunkers import build_chunks

    _require_corpus()
    strategy = args.strategy or settings.chunk_strategy
    chunks, stats = build_chunks(strategy)
    print(
        f"STAGE 2 · strategy={stats['strategy']} · chunks {stats['chunks']} · "
        f"dropped_short {stats['dropped']} · deduped {stats['deduped']} · "
        f"mean_len {stats['mean_len']}"
    )
    print(f"          wrote {settings.chunks_dir / 'chunks.jsonl'} and chunks.txt")


# ── STAGES 3 & 4 · embed ─────────────────────────────────────────────────────


def _load_chunks() -> list:
    """Read the committed chunk set (STAGE 2 output) back into Chunk objects."""
    from .chunkers import load_chunks

    return load_chunks()


def cmd_embed(args: argparse.Namespace) -> None:
    from .store import collection_size, index_fingerprint, upsert_chunks

    _require_corpus()
    chunks = _load_chunks()

    # Refuse to embed chunks that disagree with the configured strategy. The index
    # fingerprint would catch this later, but only at query time, as a confusing error.
    wrong = [c for c in chunks if c.strategy and c.strategy != settings.chunk_strategy]
    if wrong and not args.rebuild:
        _fail(
            f"chunks.jsonl holds strategy {wrong[0].strategy!r} but config says "
            f"{settings.chunk_strategy!r}",
            f"run `python -m mf_rag.cli chunk --strategy {settings.chunk_strategy}`",
        )

    upsert_chunks(chunks, rebuild=args.rebuild, verbose=True)
    print(f"STAGE 4 · collection={settings.chroma_collection} · total {collection_size()}")
    print(f"          fingerprint={index_fingerprint()}")


# ── STAGES 5 & 6 · query ─────────────────────────────────────────────────────


def cmd_query(args: argparse.Namespace) -> None:
    from .answerer import answer_question
    from .guards import check_guards
    from .store import collection_size

    question = args.question.strip()
    if not question:
        _fail("empty question", 'pass a question, e.g. `query "exit load of HDFC Large Cap"`')

    # P2, made visible: the guard is checked here, before any retrieval, and the CLI reports
    # which path was taken. A reviewer can confirm compliance by reading this function alone.
    verdict = check_guards(question)
    if verdict:
        print(f"MODE   {verdict.kind} (guard fired before retrieval, no LLM call)")
        if verdict.redacted_question and verdict.redacted_question != question:
            print(f"LOGGED {verdict.redacted_question!r}  (PII redacted)")
        else:
            print(f"LOGGED {verdict.redacted_question!r}")
        print("-" * 78)
        print(verdict.message)
        if verdict.link:
            print(f"\n{verdict.link}")
        return

    if collection_size() == 0:
        _fail(
            f"vector index {settings.chroma_collection!r} is empty or missing",
            "run `python -m mf_rag.cli embed` first",
        )

    if not settings.groq_api_key:
        print("NOTE   GROQ_API_KEY not set — using the deterministic extractive fallback")
        print("       set MF_RAG_GROQ_API_KEY in .env to use the LLM path")

    started = time.perf_counter()
    answer = answer_question(question, top_k=args.top_k)
    elapsed = time.perf_counter() - started

    if args.show_chunks:
        print("-" * 78)
        print("RETRIEVED CHUNKS")
        if not answer.chunks:
            print("  (none — nothing passed the filters)")
        for i, chunk in enumerate(answer.chunks, start=1):
            print(f"  [{i}] score {chunk.score:.4f}  {chunk.plan or '-'}  {chunk.section}")
            print(f"      {chunk.source_url}")
            print(f"      filter: {chunk.filter_reason}")
            body = chunk.text.strip().replace("\n", "\n      ")
            print(f"      {body[:300]}{'…' if len(body) > 300 else ''}")
        print("-" * 78)

    print(f"MODE   {answer.mode}" + (f"  (guard: {answer.guard})" if answer.guard else ""))
    print(f"AS_OF  {answer.as_of or 'n/a'}")
    if answer.citations:
        print(f"CITE   {answer.citations[0]}")
    for note in answer.notes:
        print(f"NOTE   {note}")
    print(f"TOOK   {elapsed:.2f}s")
    print("-" * 78)
    print(answer.text)


# ── Gate 2 · eval ────────────────────────────────────────────────────────────


def cmd_eval(args: argparse.Namespace) -> None:
    from .evalkit import main as evalkit_main

    evalkit_main()


# ── Stage 7 · app ────────────────────────────────────────────────────────────


def cmd_app(args: argparse.Namespace) -> None:
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    if not app_path.exists():
        _fail(f"{app_path} not found", "the Streamlit UI is not present in this checkout")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        _fail(
            "streamlit is not installed",
            "run `pip install -r requirements.txt`",
        )
    from streamlit.web import cli as st_cli

    sys.argv = ["streamlit", "run", str(app_path), "--server.port", str(args.port),
                "--server.headless", "true"]
    sys.exit(st_cli.main())


# ── all · the acceptance command ─────────────────────────────────────────────


def cmd_all(args: argparse.Namespace) -> None:
    """Run Stages 1-4 end to end and print one line per stage (PRD §11.1)."""
    from .chunkers import build_chunks
    from .loaders import ingest_all
    from .store import collection_size, index_fingerprint, upsert_chunks

    print("=" * 78)
    print("MF Facts RAG — full pipeline")
    print("=" * 78)

    summary = ingest_all(refresh=args.refresh, verbose=True)
    print(
        f"STAGE 1 · pages requested {summary.requested} · fetched {summary.fetched} · "
        f"cached {summary.cached} · failed {summary.failed} · low_text {summary.low_text}"
    )

    strategy = args.strategy or settings.chunk_strategy
    chunks, stats = build_chunks(strategy)
    print(
        f"STAGE 2 · strategy={stats['strategy']} · chunks {stats['chunks']} · "
        f"dropped_short {stats['dropped']} · deduped {stats['deduped']} · "
        f"mean_len {stats['mean_len']}"
    )

    upsert_chunks(chunks, rebuild=args.rebuild, verbose=True)
    # `upsert_chunks` already prints its own "STAGE 4 ·" line, so this adds only the
    # fingerprint. Printing the count again here produced two STAGE 4 lines and made the
    # one-line-per-stage summary ambiguous when reading it aloud.
    print(f"          fingerprint={index_fingerprint()}")

    print("=" * 78)
    print(f"done in {time.perf_counter() - started:.1f}s · try: "
          f'python -m mf_rag.cli query "expense ratio of HDFC Large Cap Fund Direct Growth"')


# ── Parser ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mf_rag.cli",
        description="MF Facts RAG — a facts-only chatbot over HDFC Mutual Fund pages on Groww.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="STAGE 1 — fetch and clean the corpus")
    p.add_argument("--refresh", action="store_true", help="bypass the cache and refetch")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("chunk", help="STAGE 2 — chunk the corpus")
    p.add_argument("--strategy", choices=["a", "b", "c"], help="override the configured strategy")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("embed", help="STAGES 3-4 — embed chunks and build the vector index")
    p.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("query", help="STAGES 5-6 — ask a question")
    p.add_argument("question", help="the question to answer")
    p.add_argument("--show-chunks", action="store_true", help="print retrieved chunks + scores")
    p.add_argument("--top-k", type=int, default=None, help="override the candidate pool size")
    p.add_argument("--min-score", type=float, default=None, help="override the similarity floor")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("eval", help="Gate 2 — score chunking strategies, calibrate min_score")
    p.add_argument("--strategy", choices=["a", "b", "c"], help="score one strategy only")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("app", help="STAGE 7 — launch the Streamlit UI")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("all", help="run STAGES 1-4 and print one line per stage")
    p.add_argument("--refresh", action="store_true", help="bypass the cache and refetch")
    p.add_argument("--strategy", choices=["a", "b", "c"], help="override the chunking strategy")
    p.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    global started
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "query" and args.min_score is not None:
        # A one-off override for a single query, not a silent mutation of global config.
        settings.min_score = args.min_score

    try:
        args.func(args)
    except FileNotFoundError as exc:
        _fail(str(exc), "see the README for the stage order")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
