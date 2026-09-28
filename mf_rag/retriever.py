"""STAGE 5b — Retrieval: question → vector → filter → top-k → MMR → threshold → context.

Four things happen here, in this order, and the order is the design:

1. **Entity resolution (scheme + plan) as a filter, not a score.** This is the single most
   important thing in this module, and it was added because measurement demanded it. See
   :func:`resolve_scheme` for the evidence — dense similarity *cannot* separate HDFC Large Cap
   Fund Direct Growth from the same fund's Regular Growth page (cosine 0.8985 vs 0.9009), so a
   "direct plan" answer could confidently cite the Regular plan's 1.57% TER as its 1.03%.
   A filter has no margin problem because it is not a margin.
2. **Cosine top-k** from the shared embedder (P6) and the persistent store.
3. **MMR re-rank** (λ = 0.7) to avoid returning three near-identical fee chunks.
4. **Threshold** at ``settings.min_score``; if nothing survives, return ``[]`` so the caller
   takes the "not in my sources" path (P4). Never return a weak chunk just to have something
   to say — a low-confidence citation is worse than an honest "I don't have that".

**Guards are not re-checked here.** ``retrieve`` assumes :func:`mf_rag.guards.check_guards`
already passed, and the callers (``answerer.answer_question``, ``cli``, ``app``) all run
guards first. That ordering is what makes P2 verifiable by reading import lines rather than by
trusting a comment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import settings
from .embedder import embed_one
from .store import search as vector_search
from .store import verify_index_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk selected for the answer, with the score it was selected on.

    ``score`` is cosine **similarity**, higher is better, already converted from Chroma's
    distance by ``store.search``. ``filter_reason`` records *how* the chunk survived the
    entity filter, which is what makes the scheme disambiguation auditable in the UI (FR-14)
    rather than a black box.
    """

    chunk_id: str
    text: str
    source_url: str
    title: str
    scheme: str
    category: str
    page_role: str
    plan: str
    section: str
    char_len: int
    fetched_at: str
    score: float
    filter_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_url": self.source_url,
            "title": self.title,
            "scheme": self.scheme,
            "plan": self.plan,
            "category": self.category,
            "page_role": self.page_role,
            "section": self.section,
            "char_len": self.char_len,
            "fetched_at": self.fetched_at,
            "score": round(self.score, 4),
            "filter_reason": self.filter_reason,
        }


# ── Entity resolution ───────────────────────────────────────────────────────
#
# Plan words as they appear in a user question, mapped onto the plan descriptors the corpus
# actually publishes ("Direct Growth", "Direct IDCW", "Growth", ...). Groww sets plan_type to
# "Direct" on BOTH the Growth and IDCW pages, so plan_type alone cannot identify a page — see
# scheme_facts.plan_label.

_PLAN_WORDS: dict[str, str] = {
    "idcw": "direct idcw",
    "dividend": "dividend",
    "direct growth": "direct growth",
    "direct-growth": "direct growth",
    "regular growth": "growth",
    "growth": "growth",
    "direct": "direct",
    "regular": "regular",
}

#: Plans the corpus actually publishes, so a descriptor that cannot match any page is never
#: used to filter. A bare "dividend" cannot be honoured as "Direct IDCW" — the user did not
#: say direct, and guessing would be the same class of error as citing the wrong plan's TER.
_KNOWN_PLANS: tuple[str, ...] = ("direct growth", "growth", "direct idcw")

#: Plan-ish words stripped before scoring a scheme-name match, so "direct growth" in the
#: question does not read as a scheme-name token.
_STOPWORDS = frozenset({
    "hdfc", "mutual", "fund", "funds", "the", "a", "an", "of", "for", "in", "on", "is",
    "what", "which", "how", "much", "many", "tell", "me", "please", "and", "or", "to", "my",
    "direct", "regular", "growth", "idcw", "dividend", "plan", "option", "scheme",
    "expense", "ratio", "exit", "load", "minimum", "sip", "lump", "sum", "lock", "in",
    "period", "riskometer", "risk", "benchmark", "index", "nav", "aum", "stamp", "duty",
    "fees", "charges", "charges", "tenure", "holdings", "returns", "return", "performance",
})


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")


def _name_tokens(scheme_name: str) -> set[str]:
    """Distinguishing tokens of a scheme name, ignoring any parenthetical qualifier.

    Groww publishes one of the five in-scope schemes as **"HDFC Equity (Flexi Cap) Fund"**,
    which nobody types. A user asks about "HDFC Equity Fund", so the question tokenises to
    ``{equity}`` while the name tokenises to ``{cap, equity, flexi}`` — a 33% match, below
    the full-coverage bar, so the entity filter declined and retrieval was left to pick the
    citation by cosine similarity alone. It picked the Large Cap page, because every scheme
    page leads with the same "SIP available / lump sum available" bullets. The answer was
    the right number and the wrong link, which is the failure the filter exists to prevent.

    Dropping the parenthetical makes the name match how it is actually spoken. The safety
    property is unchanged: stripping qualifiers can only make two *different* names collide,
    and a collision is caught by the ``distinct_funds`` guard below, which refuses to filter
    rather than guessing. There is no scheme called "Flexi Cap" in scope, so no collision
    arises here.
    """
    return _tokens(_PARENTHETICAL_RE.sub("", scheme_name))


def resolve_scheme(question: str) -> dict:
    """Work out which scheme (and plan) the question is about.

    Returns ``{"slugs": [...], "plan": str, "reason": str}``. ``slugs`` is empty when the
    question names no known scheme, which is a legitimate and common case for concept
    questions ("what is an ELSS?") — the caller then must not filter.

    **Why this is a filter and not a bonus (architecture §17.2).** Measured against the real
    index, "expense ratio of HDFC Large Cap Fund" scores the *Regular* Growth page at 0.9009
    and the *Direct* Growth page at 0.8985. A 0.0024 gap is not a margin — it is noise, and
    HNSW will happily return the wrong one first. The two pages differ by one token in a
    ~250-character chunk, so no embedding model can separate them; only an exact identity
    check can. That is the same reasoning as ADR-3 (deterministic before generative) applied
    to retrieval instead of to intent.

    Match requires every distinguishing token of the scheme's short name to be present, so
    "large cap" cannot match "HDFC Focused Large Cap Fund"'s tokens by accident. Ambiguity
    between two schemes (Balanced Advantage vs Equity) is resolved by requiring the question
    to name one of them.
    """
    from .sources import SOURCES

    q_lower = question.lower()
    q_tokens = _tokens(question)

    # Which plan did the user ask about, if any? Longest phrase wins, so "direct growth" beats
    # the bare "growth" also present in the same question.
    plan = ""
    for phrase, descriptor in _PLAN_WORDS.items():
        if phrase in q_lower and len(phrase) > len(plan):
            plan = descriptor
    if plan not in _KNOWN_PLANS:
        # "direct" alone, or "dividend" alone, cannot be mapped to a page without guessing.
        plan = ""

    best: list = []
    for spec in SOURCES:
        if not spec.scheme:
            continue
        name_tokens = _name_tokens(spec.scheme)
        if not name_tokens:
            continue
        overlap = name_tokens & q_tokens
        if not overlap:
            continue
        # Fraction of the scheme's own distinguishing tokens the question actually named.
        # A full match is required to filter; a partial match is noted but not acted on,
        # because filtering on a partial match would *hide* the right answer.
        coverage = len(overlap) / len(name_tokens)
        best.append((coverage, spec))

    if not best:
        return {"slugs": [], "plan": plan, "reason": "no scheme named"}

    best.sort(key=lambda pair: pair[0], reverse=True)
    top_coverage = best[0][0]

    if top_coverage < 0.999:
        return {
            "slugs": [],
            "plan": plan,
            "reason": f"weak scheme match ({top_coverage:.0%} of tokens) — not filtering",
        }

    # Several SOURCES rows can be the *same fund* in different plans (HDFC Large Cap Fund
    # appears as Direct Growth, Regular Growth and Direct IDCW). That is not ambiguity — it is
    # one fund with several pages, and the plan discriminator decides between them. Genuine
    # ambiguity is two *different funds* at full coverage ("large cap or flexi cap"), where
    # the guard upstream has already refused a comparative question anyway.
    tied_slugs = sorted({spec.slug for coverage, spec in best if coverage >= 0.999})
    distinct_funds = {spec.scheme for coverage, spec in best if coverage >= 0.999}
    if len(distinct_funds) > 1:
        return {
            "slugs": [],
            "plan": plan,
            "reason": f"{len(distinct_funds)} different funds named equally — not filtering",
        }

    plan_reason = f" · plan {plan}" if plan else " · plan unspecified (preferring primary)"
    return {
        "slugs": tied_slugs,
        "plan": plan,
        "reason": f"scheme matched exactly: {tied_slugs[0]}{plan_reason}",
    }


def _matches_filter(chunk, resolution: dict, scored_siblings: list) -> tuple[bool, str]:
    """Decide whether a retrieved chunk survives the entity filter.

    ``scored_siblings`` is the full retrieved pool, used only to answer "does another plan of
    this same fund exist among the candidates?" — which is what decides whether an unspecified
    plan needs tie-breaking at all.

    Returns ``(keep, reason)``. A chunk belonging to a *different* fund is dropped; chunks from
    non-scheme pages are always kept, because a concept question legitimately needs the
    education page and rule 6 of the system prompt covers that boundary (P3 / ADR-10).
    """
    slugs = resolution["slugs"]
    if not slugs:
        # Nothing in the question identifies a fund, so similarity alone decides the citation —
        # and that is how a bare "what is the benchmark" came back citing the Direct IDCW page,
        # winning by 0.007 over three in-scope pages. Plan-variant pages are in the corpus only
        # so the entity filter has something to reject (ADR-14); when nothing disambiguates,
        # prefer a page the bot is actually scoped to. This is the same deterministic preference
        # applied to the plan-unspecified case below, applied one step earlier.
        #
        # Context pages are exempt: a concept question ("what is the difference between ELSS and
        # SIP") legitimately needs one, and it will be the only kind of page in that pool.
        if chunk.page_role == "variant" and any(
            c.page_role != "variant" for c in scored_siblings
        ):
            return False, "no scheme named; a primary or education page is available"
        return True, "no scheme filter"

    slug = chunk.chunk_id.split("#")[0]
    if slug not in slugs:
        # Keep context pages: they are the explanation, not a competing scheme number.
        if not chunk.scheme:
            return True, "context page (page_role context)"
        return False, f"different scheme: {chunk.scheme}"

    plan = resolution["plan"]
    if plan and chunk.plan:
        # Exact descriptor match only, or the query descriptor contained in a longer page
        # descriptor. Substring matching alone is not enough: "growth" is a substring of
        # "direct growth", so an unguarded `plan in chunk.plan` would let a question about
        # Regular Growth be answered from the Direct Growth page.
        if chunk.plan.lower() == plan or chunk.plan.lower().endswith(f" {plan}"):
            return True, f"scheme + plan matched ({chunk.plan})"
        return False, f"plan mismatch: question {plan!r} vs page {chunk.plan!r}"

    # Plan unspecified but the same fund has several plan pages. Prefer the in-scope one
    # rather than letting a coin-flip decide which TER gets cited. This is deterministic
    # preference, not a score tweak, and it is only applied when a tie actually exists.
    siblings = [c for c in scored_siblings if c.scheme == chunk.scheme and c.plan != chunk.plan]
    if siblings:
        if chunk.page_role == "primary":
            return True, f"scheme matched; primary page preferred ({chunk.plan})"
        return False, f"variant page {chunk.plan!r}; primary page exists for this fund"

    return True, "scheme matched (plan unspecified)"


def mmr_rerank(candidates: list, lambda_: float, top_k: int) -> list:
    """Maximal Marginal Relevance: trade relevance against diversity.

    ``MMR = argmax_{c ∉ S}  λ·sim(c,q) − (1−λ)·max_{s ∈ S} sim(c,s)``

    Why this exists: HDFC publishes its fee block as several separate chunks (TER, exit load,
    stamp duty, turnover) whose texts share almost every token. Pure cosine returns three of
    them and crowds out the minimum-investment chunk the user actually asked about. MMR keeps
    the best one and then prefers chunks that say something *new*.

    Similarity between candidates is computed on demand from their embeddings, so this does
    not need the vectors handed to it — the score on each candidate is already the query
    similarity, and candidate-vs-candidate similarity is re-embedded only for the small set
    still in play. At ``mmr_k=4`` over 6 candidates this is a handful of extra encodings, and
    the model is already resident in memory.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return list(candidates)

    # Encode the candidate texts once; the pairwise matrix below reuses these.
    vectors = embed_batch([c.text for c in candidates])

    # sim(c_i, c_j) on unit vectors is the dot product.
    def sim_matrix() -> list[list[float]]:
        out = []
        for i in range(len(vectors)):
            row = []
            for j in range(len(vectors)):
                row.append(sum(a * b for a, b in zip(vectors[i], vectors[j])))
            out.append(row)
        return out

    similarities = sim_matrix()
    # Relevance term: the cosine score from the store, which is sim(candidate, question).
    relevance = [c.score for c in candidates]

    selected: list[int] = [max(range(len(candidates)), key=lambda i: relevance[i])]
    remaining = [i for i in range(len(candidates)) if i not in selected]

    while remaining and len(selected) < top_k:
        best_i, best_value = None, float("-inf")
        for i in remaining:
            redundancy = max(similarities[i][j] for j in selected)
            value = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_i, best_value = i, value
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch wrapper so MMR does not encode one text at a time.

    Thin alias on the single shared embedder, not a second embedding path (P6).
    """
    from .embedder import embed

    return embed(texts)


def build_context(chunks: list) -> str:
    """Render retrieved chunks as numbered prompt blocks (architecture §11.2).

    ``page_role`` is in the **prompt text**, not just metadata: rule 6 of the system prompt
    tells the model that education blocks explain concepts and must never be the source of a
    scheme number (P3 / ADR-10), and an instruction cannot refer to metadata the model never
    sees. The block number is what lets post-check 1→2 map a model's citation back to a URL
    that we actually fetched, which is what makes "exactly one real citation" mechanically
    enforceable rather than hoped for.
    """
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] source_url: {chunk.source_url} | page_role: {chunk.page_role} | "
            f"scheme: {chunk.scheme} | plan: {chunk.plan} | section: {chunk.section}\n"
            f"    {chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def retrieve(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Question → ranked, filtered, thresholded chunks. ``[]`` means "not in my sources" (P4).

    Order: entity filter → cosine top-k → MMR → threshold. The filter runs *after* fetching a
    wide candidate pool (not before) so that filtering can only ever remove wrong-scheme
    chunks, never remove the right one that a too-narrow pool failed to return.
    """
    top_k = top_k or settings.top_k
    min_score = settings.min_score if min_score is None else min_score

    # A stale index silently returns nonsense (ADR-16). Fail loudly, here, before answering.
    verify_index_fingerprint()

    query_vec = embed_one(question)
    pool_size = max(top_k * 3, top_k)
    # A missing index raises FileNotFoundError from store.search rather than returning [],
    # so "index not built" can never be mistaken for "nothing in the corpus".
    scored = vector_search(query_vec, top_k=pool_size)
    if not scored:
        logger.info("STAGE 5 · q=%r · index empty", question)
        return []

    resolution = resolve_scheme(question)

    # 1 · entity filter
    kept = []
    for chunk in scored:
        ok, reason = _matches_filter(chunk, resolution, scored)
        if ok:
            chunk.filter_reason = reason
            kept.append(chunk)
        else:
            logger.debug("STAGE 5 · dropped %s — %s", chunk.chunk_id, reason)

    if not kept:
        # The filter removed everything, which means the question named a scheme and the
        # store had nothing for it. Treat as "not in my sources" rather than falling back to
        # unfiltered results — falling back would re-introduce exactly the wrong-scheme
        # citation the filter exists to prevent.
        logger.info("STAGE 5 · q=%r · filter removed all %d candidates (%s)",
                    question, len(scored), resolution["reason"])
        return []

    # 2 · MMR for diversity
    #
    # Top-k is applied BEFORE MMR, not after. MMR greedily fills its budget in relevance
    # order, so giving it 4 slots to pick from 6 means it chooses 4 of those 6 — whereas
    # re-ranking a 6-deep list down to 4 would simply truncate it. Feeding it a wider pool
    # (top_k * 3, filtered above) is what lets diversity actually change the selection.
    #
    # MMR is asked for ``top_k`` results, not ``mmr_k``. ``mmr_k`` is the *shortlist* size —
    # how many candidates MMR chooses among — and using it as the output size silently
    # overrode the caller: retrieve(top_k=12) returned 4, which is why the extractive
    # fallback could never see past the first four groups however deep the pool was set.
    # ``top_k`` is the caller's contract; nothing downstream should quietly reduce it.
    shortlist = kept[:max(settings.mmr_k, top_k)]
    reranked = mmr_rerank(shortlist, settings.mmr_lambda, top_k)

    # 3 · threshold
    final = [c for c in reranked if c.score >= min_score]

    # MMR returns in selection order, not relevance order: the first pick is the most
    # relevant, but later picks are chosen for novelty and can score lower. Sort back by
    # score so the model sees its strongest evidence first, which also makes the post-check's
    # "map the citation to the top chunk" step predictable.
    final.sort(key=lambda c: c.score, reverse=True)

    best = max((c.score for c in final), default=0.0)
    logger.info(
        "STAGE 5 · q=%r · top_k=%d · mmr=%d · kept %d · best_score=%.3f · filter=%s",
        question, top_k, settings.mmr_k, len(final), best, resolution["reason"],
    )

    return [
        RetrievedChunk(
            chunk_id=c.chunk_id, text=c.text, source_url=c.source_url, title=c.title,
            scheme=c.scheme, category=c.category, page_role=c.page_role, plan=c.plan,
            section=c.section, char_len=c.char_len, fetched_at=c.fetched_at, score=c.score,
            filter_reason=c.filter_reason,
        )
        for c in final
    ]


if __name__ == "__main__":
    probes = [
        "expense ratio of HDFC Large Cap Fund",
        "expense ratio of HDFC Large Cap Fund Direct Growth",
        "exit load of HDFC Equity Flexi Cap fund",
        "what is the weather in Delhi?",
    ]
    for probe in probes:
        print(f"\n=== {probe!r} ===")
        resolution = resolve_scheme(probe)
        print(f"  resolve: {resolution['reason']} slugs={resolution['slugs']} plan={resolution['plan']!r}")
        chunks = retrieve(probe)
        for chunk in chunks:
            print(f"  {chunk.score:.3f}  {chunk.scheme:<32} {chunk.section[:32]:<32} [{chunk.filter_reason}]")
        if not chunks:
            print("  (no chunks)")
