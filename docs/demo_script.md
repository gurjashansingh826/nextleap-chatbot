# Demo script — 3 minutes

A timed run-of-show. Every number and quoted answer below was produced by a real run of this
build; regenerate with `tools/make_sample_qa.py` if anything here looks stale.

**Before you start** (30 s, do it the night before, not in the room):

```powershell
.\.venv\Scripts\python.exe -m mf_rag.cli all      # ~80 s, then cached
.\.venv\Scripts\python.exe -m mf_rag.cli app
```

Leave the browser open on the app. Have `cli query` in a second terminal for the fallback
demo. Confirm `tools\check_key.ps1` reports the key SET — if it does not, the run still
works, on the extractive path, and that is a *feature* you can show rather than a failure to
hide.

---

## 0:00 — What this is (20 s)

> "A facts-only chatbot over 15 public HDFC Mutual Fund pages on Groww. It answers scheme
> data with a citation and a date. It will not advise, will not compare returns, and will not
> guess. The corpus is fixed and published — every number it says is on the page it links."

Do not open with the architecture. Open with the constraint — it is the interesting part.

---

## 0:20 — A fact, with the pipeline visible (45 s)

Type:

```
What is the expense ratio of HDFC Large Cap Fund Direct Growth?
```

Expand **Retrieved context** in the answer. Point at three things:

1. the **scheme/plan prefix** on the chunk — it is in the text the model sees, not just metadata;
2. the **score** on each chunk;
3. the **filter log** — which chunk was dropped and *why* ("different scheme: HDFC Small Cap
   Fund").

> "The filter is a filter, not a bonus. On the real index, the Regular Growth page scores
> 0.9009 and the Direct Growth page 0.8985 for this same question. That is a 0.002 gap —
> noise. No embedding model separates two pages that differ by one token, so identity is
> resolved by exact match before anything generative happens."

**Ask the room:** what happens if you ask the same question about the Regular Growth page?
Answer: the plan descriptor is in the chunk prefix *and* the Chroma metadata, because Groww
sets `plan_type = "Direct"` on the IDCW page too — so `plan_type` alone cannot tell Direct
Growth from Direct IDCW (ADR-14).

---

## 1:05 — The refusal (20 s)

Type:

```
Should I buy HDFC Small Cap Fund?
```

> "One fixed sentence. It happens before retrieval and before the model — a refusal cannot be
> produced after a language model has already read the question. There is no trailing link
> either: pointing you somewhere else is still guidance."

Then, fast, for the guard matrix:

| Type this | It refuses because |
| --- | --- |
| `Which HDFC fund gave the best returns?` | returns |
| `Compare HDFC Large Cap Fund and HDFC Small Cap Fund` | comparative |
| `What if the market crashes next year?` | speculative |
| `my PAN is ABCDE1234F` | PII — and it is never logged |

**The subtle one worth calling out:** ask `What is the exit load of HDFC ELSS Tax Saver Fund?`
— it answers. The corpus publishes "exit load applies if redeemed within 1 year", a
forward-looking *fact*. The speculative guard is anchored only on hypothetical frames
("what if", "suppose") so it catches speculation without making published conditionals
unanswerable. A guard broad enough to catch "what if the market crashes" would also catch
"what is the exit load."

---

## 1:25 — It declines, and that is the feature (30 s)

Type:

```
What is the weather in Delhi?
```

> "Not in the corpus, so it says so and lists what it does cover. It does not answer from
> memory and it does not answer from the model's weights."

Then the harder version — a question about a field the page does not publish:

```
What is the Sharpe ratio of HDFC Large Cap Fund?
```

> "Same answer, and this is the case that took the most work. An earlier build answered
> `SIP available: Yes` — a real field, quoted verbatim, under a real citation, with nothing
> about it looking wrong. A wrong fact with a legitimate-looking citation is the worst thing
> this system can do, because the user has no way to tell. Now the extractive path looks for
> the chunk that actually *mentions the subject*, and declines when there isn't one."

Note which fund you ask about: `Who manages HDFC Large Cap Fund?` answers (Prashant Jain),
`Who manages HDFC Focused Large Cap Direct Plan?` declines, because that page publishes no
manager. Two sibling pages, one with the field and one without — that is the pair to show.

---

## 1:55 — Follow-ups, and the memory window (35 s)

Ask, in order, **without naming the fund again** after the first:

```
expense ratio of HDFC Large Cap Fund Direct Growth?
what about its exit load?
and the minimum SIP
what is the benchmark
who manages it
```

Every one cites the same page, and `who manages it` answers **Prashant Jain**. Do not
interleave a refusal in the middle of that chain — a refusal is not recorded, so the
subject falls back to the last *answerable* turn. Ask the refusals from the previous
section first, then start this chain fresh; otherwise `what about its exit load?`
answers about whatever fund you last got a real answer for, which looks like a bug and
is actually the rule working.

That last one is worth showing deliberately. Get the Large Cap chain going, then ask:

```
Should I buy HDFC Small Cap Fund?
and the minimum SIP
```

The refusal does not change the subject — the follow-up still resolves to Large Cap. A
refusal has no subject to lend, so lending one would mean quietly answering about a fund
the bot just declined to discuss.

> "The follow-up window holds the last 10 exchanges, in `st.session_state` only. It is
> dropped when the tab closes, and it stores `(question, slug, mode)` — never answer text.
> It is strictly a *query* rewrite: the guards always see the question exactly as typed, so
> memory can never talk its way past a rule."

---

## 2:30 — No key, still answers (20 s)

In the second terminal:

```powershell
.\.venv\Scripts\python.exe -m mf_rag.cli query "expense ratio of HDFC Large Cap Fund Direct Growth?"
```

Then rename `.env` temporarily (or blank line 7) and run it again.

> "Identical pipeline, deterministic extractive answer: `- Expense ratio (TER): 1.03%`, same
> citation, same date. It cannot hallucinate because it never generates prose — it quotes
> lines from a chunk we fetched. That is why the demo has a floor: there is no configuration
> in which this system answers without a citation."

---

## 2:50 — Close (10 s)

> "Facts only, from a published corpus, every answer cited and dated, and every refusal
> identical. 268 tests, 36 labelled eval questions, 30 of 30 in-scope questions answered
> with the right fact on the right page after filtering."

### If asked

- **"What's your accuracy?"** — 86.7% recall@5 and 96.7% citation accuracy on the labelled
  set with the entity filter; 5 known misses, itemised in `docs/eval_report.md`. Not rounded up.
- **"Why not use a bigger model?"** — the guard and filter stages do the safety work; the
  model only writes prose. A bigger model would not fix a wrong citation.
- **"How do you stop it hallucinating?"** — it cannot be stopped by prompting. Seven
  post-checks, then a path that quotes rather than generates. Post-check 2 rejects any URL
  that is not a page we fetched, which is a mechanical guarantee rather than a hope.
- **"What's the weakest part?"** — MiniLM ranks "Scheme identity" (where fund-manager facts
  live) around 13th for a manager question, so retrieval recall understates what the system
  actually answers; the deeper pool plus on-topic group selection recovers it. That ordering
  is arbitrary for short labelled bullets, and it is the weakest part.
