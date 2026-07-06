# dsv4_pilot — generate 50 divergent examples with DeepSeek V4 Pro

Isolated pilot. DeepSeek V4 Pro generates candidate examples across four sources
(Reverend Insanity · Jin Ping Mei · The Prince · Mahabharata), the **DAV** criterion
gates them, and the passers become an **unlabeled** training-shaped set.

**Architecture (refract, not one-shot):**
```
source brief → DSV4 REFRACTOR  → {problem, facets, 4 distinct angle-seeds}
            → DSV4 WORKER ×4   → one thread each, ISOLATED (blind to the others)
            → DAV gate         → passers (≤50) / near-miss (hard negatives) / rejects
```
Isolation between workers is what guarantees the threads can't rhyme.

**Safety:**
- Isolated: reads only the OpenRouter + OpenAI keys; writes only into `out/`.
- Hard money cap (`HARD_USD_CAP`) with a live ledger; refuses new calls past the cap.
- Save-as-you-go: every generated candidate is flushed before scoring.
- Expected spend well under $1; cap set far above it.

**Unlabeled:** `out/passers.jsonl` carries `{problem, facets, threads}` — **no source tag.**
The emergent source mix is reported only as a QA diagnostic; the training file never sees it.

Run: `python3 generate.py`  →  `out/passers.jsonl`, `out/passers.md`, `out/near_miss.jsonl`,
`out/rejects.jsonl`, `out/summary.json`.
