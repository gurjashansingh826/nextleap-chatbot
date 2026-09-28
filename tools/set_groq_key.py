"""Write the Groq API key into ``.env`` without putting it in shell history.

Run this yourself, in a terminal you control::

    .venv\\Scripts\\python.exe tools\\set_groq_key.py

It exists because editing ``.env`` by hand kept silently failing: the file was byte-identical
after each attempt, and a key that never lands on disk is indistinguishable from a key that
was never entered. This script prompts, writes, re-reads through the real config loader, and
tells you which of those actually happened.

The key is never echoed back, never printed, and never written anywhere except the gitignored
``.env``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
KEY_LINE = "MF_RAG_GROQ_API_KEY="


def main() -> int:
    if not ENV.exists():
        print(f"ERROR: {ENV} not found.")
        print("  fix: copy .env.example to .env first")
        return 1

    current = ENV.read_text(encoding="utf-8")
    if KEY_LINE not in current:
        print(f"ERROR: no {KEY_LINE} line in {ENV.name}.")
        print("  fix: restore .env from .env.example and re-run")
        return 1

    key = input("Paste your Groq API key: ").strip()
    if not key:
        print("Nothing entered. .env unchanged.")
        return 1

    # Catch the two mistakes that make a key look present to the eye and absent to the code.
    if key.startswith(('"', "'")) and key.endswith(('"', "'")) and len(key) > 1:
        print("Stripping surrounding quotes.")
        key = key[1:-1]
    if " " in key:
        print("ERROR: the key contains a space, so it was probably pasted with a trailing")
        print("       'Groq' prefix or URL. Copy only the key itself from the console.")
        return 1

    updated = re.sub(
        rf"(?m)^{re.escape(KEY_LINE)}.*$", f"{KEY_LINE}{key}", current, count=1
    )
    ENV.write_text(updated, encoding="utf-8")

    # Verify through the real loader, not by re-reading the file we just wrote. A key that is
    # in the file but not in settings is exactly the failure this script exists to rule out.
    for module in [m for m in list(sys.modules) if m.startswith("mf_rag")]:
        del sys.modules[module]
    from mf_rag.config import settings

    loaded = settings.groq_api_key
    if not loaded:
        print("ERROR: wrote the file but settings.groq_api_key is still empty.")
        print("       Check for a second MF_RAG_GROQ_API_KEY line, or a .env override.")
        return 1
    if loaded != key:
        print("ERROR: settings.groq_api_key does not match what you typed.")
        return 1

    print(f"OK: .env updated and the key loaded ({len(loaded)} chars, {loaded[:3]}...).")
    print("    .env is gitignored, so the key will not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
