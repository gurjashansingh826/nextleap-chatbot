"""STAGE 7 — ChatBoat, the Streamlit UI.

A facts-only bot is a claim that it does not make things up, and that claim is only credible
if the audience can watch the machinery that prevents it. So the retrieved chunks, their
scores, and the ``filter_reason`` for each one are all still rendered — behind one checkbox,
off by default.

That default is the judgement this file encodes. A chunk from
``hdfc-large-cap-fund-regular-growth`` appearing under a "Direct Growth" question is precisely
the ADR-14 hazard, and the deterministic entity filter is the only defence against it, so the
verdict has to be auditable. But the user asking "who manages the fund" is not auditing
anything — they want the name. Six lines of scores and filter verdicts between the question
and the answer is a cost paid by every user to serve an auditor who asks for it once, and the
answer itself got shorter than the evidence for it. So the mechanism is one click away and
the answer is the first thing on screen.

Header counts are read from ``data/sources.csv`` rather than hardcoded, so the banner cannot
claim a corpus the index does not have.

Layout
------
``st.chat_input`` is pinned to the bottom of the page by Streamlit, so nothing can be laid out
*below* it. The example cards are therefore the last element before the input call, which puts
them immediately above the bar — as close to it as the widget allows without giving up the
native chat input and its Enter-to-send behaviour.

Colour carries meaning
----------------------
A refusal, an out-of-scope decline and a fact must never look alike, because "I can help you
with the factual details" and a one-line answer are near-identical strings on screen. So the
three states get three tints, and the rest of the page is deliberately desaturated so those
three are the only colourful things the eye lands on. The tints are defined once, in
``STATE_TINT``, and reused by the banner, the answer card and the mode help — so a state can
never be rose in one place and plain in another.

Structure
---------
    §1  constants      name, examples, per-state copy and colour
    §2  data helpers   pure functions, no Streamlit, unit-testable
    §3  styling        one place for the CSS, injected once
    §4  renderers      one function per piece of the page
    §5  state          session_state access, in one place
    §6  main           assembles the sections in order

Run with ``python -m mf_rag.cli app`` (or ``streamlit run app.py``).
"""

from __future__ import annotations

import logging

import streamlit as st

from mf_rag import prompts
from mf_rag.answerer import answer_question
from mf_rag.config import settings
from mf_rag.memory import ConversationMemory
from mf_rag.sources import EXPECTED_PRIMARY_COUNT, load_sources_csv

_LOGGER = logging.getLogger(__name__)

# ── §1  constants ─────────────────────────────────────────────────────────────

APP_NAME = "ChatBoat"
APP_TAGLINE = "Facts-only assistant for HDFC Mutual Funds on Groww"
APP_URL = "http://localhost:8501"

#: Three example questions, one per answer shape and one per scheme, so a first-time user
#: sees the bot do three visibly different things rather than three variations on one. They
#: are also three questions with a known expected source page, which means the demo is
#: scored by construction: a wrong number here is visible in the citation.
#:
#: "Who is the fund manager" is here on purpose. It is the hardest field in the corpus —
#: MiniLM ranks the "Scheme identity" group that publishes it around 13th, because short
#: labelled bullets are close to arbitrary for cosine similarity — so it is the clearest
#: demonstration that the on-topic group selection matters more than the raw ranking.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("👤", "Who is the fund manager of HDFC Small Cap Fund?"),
    ("📊", "What is the expense ratio of HDFC Large Cap Fund Direct Growth?"),
    ("🔒", "Is HDFC ELSS Tax Saver Fund locked in?"),
)

#: (left border, background, icon) per outcome. Used by the answer card and the legend so
#: the two can never disagree about what a refusal looks like.
STATE_TINT: dict[str, tuple[str, str, str]] = {
    "llm": ("#4F46E5", "#FFFFFF", "✅"),
    "extractive": ("#0EA5E9", "#F0F9FF", "📋"),
    "not_in_sources": ("#2563EB", "#EFF6FF", "🔍"),
    "refusal": ("#E11D48", "#FFF1F2", "🚫"),
    "pii_refusal": ("#E11D48", "#FFF1F2", "🔒"),
}

MODE_HELP: dict[str, str] = {
    "llm": "Groq wrote this from the chunks below. Every sentence was checked against them.",
    "extractive": (
        "No LLM was called. The answer is copied verbatim from a chunk that mentions the "
        "subject, so it cannot hallucinate — but it is a quote, not a summary."
    ),
    "not_in_sources": "Nothing in the corpus answered this. The bot declined rather than guess.",
    "refusal": "A guard fired before retrieval. Nothing was fetched and no LLM was called.",
    "pii_refusal": (
        "A PII guard fired before retrieval. The identifier was redacted and never logged, "
        "stored, or sent anywhere."
    ),
}

CHAT_PLACEHOLDER = "Ask a factual question about the 5 HDFC schemes…"

#: A chevron pointing down-right, into the chat bar.
#:
#: Inline SVG rather than an emoji or an icon font. An emoji renders at the font's colour and
#: cannot be sized to 15px without looking chunky; an icon font is a network request that can
#: fail, and a failed glyph is a blank box where the affordance should be. This is 4 lines of
#: path data that always renders, inherits `currentColor`, and scales cleanly.
_ARROW_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2.75" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M5 5l14 14"/><path d="M19 9v10H9"/></svg>'
)


# ── §2  data helpers (pure, no Streamlit) ──────────────────────────────────────


def corpus_banner() -> str:
    """Header counts, read from the source registry rather than hardcoded."""
    try:
        rows = load_sources_csv()
    except Exception:  # pragma: no cover - a status line must never take the app down
        return f"{EXPECTED_PRIMARY_COUNT} schemes"

    primary = sum(1 for r in rows if r.get("page_role") == "primary")
    dates = sorted(r.get("fetched_at", "") for r in rows if r.get("fetched_at"))
    snapshot = dates[-1][:10] if dates else "unknown"
    return f"{len(rows)} public pages · {primary} schemes · snapshot {snapshot}"


def index_state() -> str:
    """One line describing the index, so the audience knows what they are querying."""
    try:
        from mf_rag.store import collection_size, index_fingerprint
    except Exception:  # pragma: no cover - defensive
        return "index unavailable"

    n = collection_size()
    if n == 0:
        return "index EMPTY — run `python -m mf_rag.cli all`"
    fp = index_fingerprint() or {}
    model = (fp.get("embed_model") or "?").split("/")[-1]
    return f"{n} chunks · strategy {fp.get('chunk_strategy', '?')} · {model}"


# ── §3  styling ───────────────────────────────────────────────────────────────

#: Kept deliberately small and targeted at stable hooks. Streamlit renames its internal CSS
#: classes between releases, so a stylesheet built on them breaks on upgrade; the selectors
#: below are either documented attributes (`data-testid`) or plain HTML elements.
_CSS = """
<style>
  /* Bottom padding only has to clear the pinned chat bar, not reserve a comfortable gap.
     7rem pushed the "Type your question here" label ~90px away from the input it names, so
     the label looked like a heading for the examples and the bar looked unrelated. 2.6rem
     clears the bar itself and leaves the label sitting on top of it. */
  .block-container { padding-top: 2.2rem; padding-bottom: 2.6rem; max-width: 900px; }

  /* Wordmark. Rendered as HTML rather than st.title so the two-tone treatment is
     possible and the line height is under our control. */
  .cb-mark   { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;
              color: #0F172A; line-height: 1.1; }
  .cb-mark em{ font-style: normal; color: #4F46E5; }
  .cb-tag    { color: #64748B; font-size: 0.95rem; margin-top: .2rem; }

  /* Status pills under the wordmark. */
  .cb-pills  { display: flex; flex-wrap: wrap; gap: .4rem; margin: .9rem 0 .3rem; }
  .cb-pill   { background: #F4F6FB; border: 1px solid #E2E8F0; color: #475569;
              border-radius: 999px; padding: .18rem .7rem; font-size: .78rem;
              font-weight: 600; }

  /* Answer card. The left border is the state colour, so a refusal cannot be mistaken
     for a short answer at a glance. */
  .cb-card   { border: 1px solid #E2E8F0; border-left-width: 4px; border-radius: 10px;
              padding: .85rem 1rem; margin: .2rem 0 .6rem; }
  .cb-src    { font-size: .82rem; color: #64748B; margin-top: .5rem; }
  .cb-src a  { color: #4F46E5; text-decoration: none; font-weight: 600; }
  .cb-src a:hover { text-decoration: underline; }

  /* The prompt above the chat bar. Black, and larger than the surrounding copy so it reads
     as the line that owns the bottom of the page. The arrow is the only saturated element
     on the line, so that is still where the eye lands. */
  .cb-ask    { display: flex; align-items: center; gap: .55rem;
              color: #0B1220; font-size: 1.3rem; font-weight: 800;
              letter-spacing: -0.02em; margin: .2rem 0 .7rem; line-height: 1.2; }
  .cb-ask svg{ width: 20px; height: 20px; flex: none; }
  .cb-ask span{ color: #64748B; font-weight: 500; font-size: .85rem; }

  /* Centred label for the input itself. Rendered as the last element before the chat_input
     call, so it lands immediately above the pinned bar and names it. The margins are small
     on purpose: every extra rem here is dead space between the label and the thing it
     labels, and that is the gap this line is trying to close. */
  .cb-type   { text-align: center; color: #334155; font-size: 1.15rem; font-weight: 700;
              letter-spacing: .06em; text-transform: uppercase;
              margin: .85rem 0 .1rem; }

  /* Section rules, so the page reads as four blocks rather than one long scroll. */
  .cb-rule   { border: 0; border-top: 1px solid #E2E8F0; margin: 1.4rem 0 1rem; }

  /* The disclaimer, now the notice at the top of the page. Set as a tinted panel rather
     than as loose grey text: it is the one thing a reader must see before they type, so it
     should read as a distinct block, not as a caption that belongs to the header above it.
     The generous bottom margin is deliberate — the panel is legal text and the line below it
     is the invitation to type, and at 1.2rem the two sat close enough to read as one clump
     of instructions. 2.4rem is enough for the eye to change subject in between. */
  .cb-notice { background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #94A3B8;
               border-radius: 10px; color: #475569; font-size: .8rem; line-height: 1.55;
               padding: .8rem 1rem; margin: 0 0 2.4rem; }

  /* Example buttons: full width, tinted, left-aligned so they read as a list of
     suggestions rather than a row of buttons. */
  [data-testid="stButton"] button {
      background: #F8FAFF; border: 1px solid #DBE3F5; color: #1E293B;
      border-radius: 10px; text-align: left; font-weight: 500;
      transition: border-color .12s, background .12s; }
  [data-testid="stButton"] button:hover {
      background: #EEF2FF; border-color: #A5B4FC; color: #312E81; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def answer_card(answer) -> None:
    """The answer text, one source link, and the as-of date.

    One link, always. A second link would leave the reader unable to tell which page backs
    which figure, and this corpus is built from sibling pages that differ by one token — so
    the system has to commit to a single page rather than offer two and let the reader choose.

    The raw URL is the thing being cut here, not the citation itself. A 90-character
    ``groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth`` on its own line is wider than
    the answer it supports, and it is the same string on every reply, so it reads as furniture
    rather than evidence. The anchor text is the scheme name and the full URL stays in the
    ``href``, so the link is unchanged — a reviewer can still copy it, and the hover tooltip
    shows where it goes.

    The as-of date is kept because the brief requires it, but demoted to the same line as the
    link: the two are one thought ("this, as of then"), and splitting them across lines gave
    a two-line footer to a one-line answer.
    """
    border, background, _ = STATE_TINT.get(answer.mode, ("#94A3B8", "#FFFFFF", "•"))
    body = answer.text.replace("\n", "<br>")
    if answer.citations:
        url = answer.citations[0]
        # Named after the page it points at, which is the first chunk's scheme — the same
        # page the citation is drawn from. `Answer` has no `scheme` of its own because a
        # declined answer has no page, and adding one to carry a label would be a field that
        # is empty exactly when the answer is already showing nothing to label.
        label = answer.chunks[0].scheme if answer.chunks else "Source page"
        body += (
            f'<div class="cb-src">🔗 <a href="{url}" target="_blank" '
            f'rel="noopener" title="{url}">{label}</a> · '
            f'Last updated from sources: {answer.as_of or "unknown"}</div>'
        )
    st.markdown(
        f'<div class="cb-card" style="border-left-color:{border};'
        f'background:{background}">{body}</div>',
        unsafe_allow_html=True,
    )


# ── §4  renderers ─────────────────────────────────────────────────────────────


def render_header() -> None:
    st.markdown(
        f'<div class="cb-mark">⛵ Chat<em>Boat</em></div>'
        f'<div class="cb-tag">{APP_TAGLINE}</div>',
        unsafe_allow_html=True,
    )
    pills = ["Facts only", "No investment advice", corpus_banner(), index_state()]
    st.markdown(
        '<div class="cb-pills">'
        + "".join(f'<span class="cb-pill">{p}</span>' for p in pills)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="cb-rule">', unsafe_allow_html=True)


def render_sidebar() -> tuple[int, bool, bool]:
    """Answer settings, the memory note, the state legend, and the reset control."""
    with st.sidebar:
        st.markdown(f"### ⛵ {APP_NAME}")
        st.caption(APP_URL)

        st.subheader("Answer settings")
        top_k = st.slider("Chunks given to the LLM", 1, 15, settings.top_k)
        # Off by default: the provenance trail is evidence for someone who doubts the answer,
        # not part of the answer, and it pushed a 6-line block between every reply and the
        # next question. Available one click away for a demo that wants to show the mechanism.
        show_chunks = st.checkbox("Show retrieved context", value=False)
        show_debug = st.checkbox("Show filter log", value=False)

        st.divider()
        st.subheader("What each state means")
        for mode, (_, _, icon) in STATE_TINT.items():
            st.markdown(f"{icon} `{mode}`", help=MODE_HELP.get(mode, ""))
        st.caption("Colour on the answer card matches these states.")

        st.divider()
        st.subheader("Follow-up memory")
        st.caption(
            f"Last **{settings.memory_turns}** exchanges, this tab only. Stores the "
            "question, the scheme and the answer mode — never the answer text."
        )

        st.divider()
        st.subheader("Not investment advice")
        st.caption(prompts.DISCLAIMER)
        st.caption(
            "Don't enter PAN, Aadhaar, account numbers, OTPs, email or phone numbers. "
            "They are refused before retrieval and never stored."
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["transcript"] = []
            st.session_state["memory"] = ConversationMemory()
            st.rerun()
    return top_k, show_chunks, show_debug


def render_chunks(answer, *, show_debug: bool) -> None:
    """The retrieved context — the audit trail for ADR-14.

    Off by default. It is the answer to "where did this come from?", which nobody asks until
    they already doubt the answer, so leaving it expanded pushes the evidence below the fold
    where a reader has to go looking for it. The mechanism is worth showing to an engineer and
    worth hiding from a user, so it lives behind one checkbox.
    """
    if not answer.chunks:
        return
    st.caption(
        "**Retrieved context.** Each entry is a page section this answer was built from, "
        "with its match score. This is the audit trail for the plan-disambiguation hazard "
        "(ADR-14): cross-scheme and wrong-plan chunks do not survive the entity filter."
    )
    for i, c in enumerate(answer.chunks, start=1):
        with st.expander(f"[{i}]  {c.score:.3f}  ·  {c.section}  ·  {c.filter_reason}"):
            st.caption(f"{c.scheme}  ({c.plan or 'plan n/a'})  —  {c.source_url}")
            body = c.text[:900] + ("…" if len(c.text) > 900 else "")
            st.text(body)

    if show_debug:
        st.dataframe(
            [
                {
                    "score": round(c.score, 4),
                    "scheme": c.scheme,
                    "plan": c.plan,
                    "section": c.section,
                    "verdict": c.filter_reason,
                }
                for c in answer.chunks
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_answer(answer, *, show_chunks: bool, show_debug: bool) -> None:
    """One assistant turn: the answer, its provenance, and how it was produced."""
    _, _, icon = STATE_TINT.get(answer.mode, ("#94A3B8", "#FFFFFF", "•"))
    with st.chat_message("assistant"):
        answer_card(answer)

        # Mode is a caption rather than a metric tile. Four big-number tiles under every reply
        # is a dashboard, and a dashboard is what you read when you do not trust the thing in
        # front of you. "Chunks: 6" tells a user nothing about the expense ratio; it is one
        # more row between the answer and the next question. The mode still matters — it is
        # how you tell a fact from a decline at a glance — so it stays, just as a quiet line,
        # and the counts live in the expander for anyone who actually wants them.
        st.caption(f"{icon} {answer.mode}")

        with st.expander("How this answer was produced"):
            st.write(MODE_HELP.get(answer.mode, answer.mode))
            st.caption(
                f"{len(answer.chunks)} chunks retrieved · "
                f"{len(answer.citations)} source(s) · "
                f"{len(st.session_state['memory'])} turn(s) remembered"
            )
            for note in answer.notes:
                st.caption(f"• {note}")

        if show_chunks:
            render_chunks(answer, show_debug=show_debug)


def render_examples() -> None:
    """The prompt and the three examples, immediately above the input bar.

    A click fills the input rather than firing the question. The user still presses Ask, so
    the transcript only ever contains turns that were actually asked, and an example never
    spends an LLM call the user did not request.
    """
    st.markdown(
        f'<div class="cb-ask">{_ARROW_SVG}<div>How may I help you?'
        "<span>&nbsp;— ask a factual question, or pick one below</span></div></div>",
        unsafe_allow_html=True,
    )
    for i, (icon, question) in enumerate(EXAMPLES):
        if st.button(f"{icon}   {question}", key=f"example_{i}", use_container_width=True):
            st.session_state["ask"] = question
            st.rerun()

    # Rendered after the examples, so it is the last element before the input call and reads
    # as a label on the bar rather than as a fourth thing to click.
    st.markdown('<div class="cb-type">Type your question here</div>', unsafe_allow_html=True)


def render_transcript(*, show_chunks: bool, show_debug: bool) -> None:
    """Replay the whole conversation, oldest first."""
    for turn in st.session_state["transcript"]:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        render_answer(turn["answer"], show_chunks=show_chunks, show_debug=show_debug)


def render_notice() -> None:
    """The disclaimer, rendered at the top of the page.

    Renamed when it moved up from above the chat bar. It was a page footer emitted after
    `st.chat_input`, which Streamlit pins to the bottom of the viewport, so the text a
    reader is required to see was pushed off-screen below the one control they had come to
    use. The old name was the last thing still describing that position.
    """
    st.markdown(f'<div class="cb-notice">{prompts.DISCLAIMER}</div>', unsafe_allow_html=True)


# ── §5  state ─────────────────────────────────────────────────────────────────


def init_state() -> None:
    """Create the session's state once. Session state is the only place it lives."""
    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("memory", ConversationMemory())


def _bootstrap_index() -> None:
    """Make sure the vector index exists before a question can be answered.

    The Chroma store is gitignored — it is regenerable (P10) — so it is absent from a fresh
    checkout and from a deployed image whose build command skipped the embed step. Without
    this, the first question on such a host dies on ``get_collection(create=False)`` with a
    FileNotFoundError, which reads as "the chatbot is broken" at exactly the moment it is
    supposed to be demonstrating. Building the index here, from the committed
    ``data/chunks/chunks.jsonl``, makes the app self-sufficient on any host: the check is one
    cheap ``list_collections`` call when the index exists, and a one-time ~10 s embed when it
    does not. ``build_chunks`` (Stage 2) never needs to run on a deploy — the chunks it
    produces are committed, so this path is embed + upsert only (Stage 3/4), no scraping.

    Callers place this AFTER the page has rendered: Streamlit streams elements to the client
    while the script runs, so running this before any ``st.*`` element turns a cold host's
    first load into a blank page for the whole duration of the model download + embed. After
    the paint, the same work shows as a brief spinner and never blanks the app.
    """
    try:
        from mf_rag.store import ensure_index
    except ImportError:  # pragma: no cover - only reachable on a broken install
        return

    try:
        # Rendered *inside* the call: st.spinner paints its element while the block runs,
        # so a cold boot (model download + ~10 s embed) shows "Building the answer index…"
        # below the already-drawn page instead of blanking the first paint.
        with st.spinner("Building the answer index — the first question takes a few seconds longer."):
            size, built = ensure_index(verbose=True)
    except Exception as exc:  # model download blocked, disk full, HF unreachable, …
        # A readable failure beats a silent white page or a raw traceback in the browser.
        # The app still renders; every question will show the loud "index missing" message
        # from get_collection(create=False), which names the rebuild command.
        st.error(
            "Could not build the answer index at startup "
            f"({type(exc).__name__}: {exc}). Check the deploy log, then run "
            "`python -m mf_rag.cli embed` on the host, or redeploy."
        )
        _LOGGER.exception("index bootstrap failed")
        return

    if built:
        st.toast(
            f"Built the answer index ({size} chunks) — the first question takes a second "
            "longer than the rest."
        )


# ── §6  main ──────────────────────────────────────────────────────────────────


def main() -> None:
    # Must be the first Streamlit call in the script.
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_TAGLINE}",
        page_icon="⛵",
        layout="centered",
    )
    inject_css()
    init_state()

    top_k, show_chunks, show_debug = render_sidebar()
    render_header()
    render_notice()
    render_transcript(show_chunks=show_chunks, show_debug=show_debug)

    # Order matters: everything above the input call renders above the pinned bar.
    render_examples()

    # Bootstrap AFTER the page has painted, not before. Streamlit streams widgets to the
    # browser as the script runs, so a host whose build skipped the embed step (cold image,
    # no baked index) would otherwise block the FIRST paint inside the model download —
    # a blank white page on Render while the proxy waits. Here the whole UI draws first and
    # the build finishes behind it, with the spinner below the examples. On a host whose
    # image already has the index (baked at build, render.yaml) this is one cheap empty
    # check and there is no visual difference.
    _bootstrap_index()

    question = st.chat_input(CHAT_PLACEHOLDER, key="ask")

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    memory: ConversationMemory = st.session_state["memory"]
    with st.spinner("Retrieving and answering…"):
        # Guards run inside answer_question, before anything is fetched or sent anywhere.
        answer = answer_question(question, top_k=top_k, memory=memory)

    st.session_state["transcript"].append({"question": question, "answer": answer})
    # Recorded only for answerable turns, and only after the answer exists, so a refusal
    # never lends its subject to the next question.
    memory.add_from_answer(answer)
    render_answer(answer, show_chunks=show_chunks, show_debug=show_debug)


if __name__ == "__main__":
    main()
