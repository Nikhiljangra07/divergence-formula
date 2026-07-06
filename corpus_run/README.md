# corpus_run — three-corpus generator (theme-conditioned, strict DAV gate)

Builds the first training set as **three separately-testable corpora** so the consequence ablation
is observable and the negative side is captured for later contrastive training.

```
config.py    sources + THEME LISTS (52 distinct dilemma-types) + two worker prompts + thresholds/knobs
dav.py       DAV scoring + gate; consequence floor is a SWITCH (DAV_full vs DAV_lite) + completeness check
generate.py  runner: set_a / set_b / aggregate
```

## The three settings

| Corpus | Gate | Worker writes | Purpose |
|---|---|---|---|
| **Set A — with consequence** | `DAV_full` (consequence ON) | move + reasoning **+ concrete cost/gain** | the primary positive set; trained/"drained" first |
| **Set B — without consequence** | `DAV_lite` (consequence OFF) | move + reasoning, **stops before any outcome** | ablation twin — isolates the consequence variable in *both* generation and gating |
| **Set C — hard negatives** | — | the gate-failures (full text + which thread failed) | second side of the coin: shown after the first set to teach the boundary (DPO) |

The single difference between A and B is the consequence dimension — so a report can compare
"with consequences, how the model behaves" vs "without, how much reasoning load is actually carried."

## Theme conditioning (the spread fix)

The pilot collapsed onto one scene per source (PR 15/16 "hold a conquered republic"). Now the refractor
is handed **one specific dilemma-type per example** from a 52-type landscape and told not to drift, with
a balanced source×theme round-robin. A 300-set therefore spans the whole source, ~5–6 examples per theme,
instead of repeating its single most famous case.

## Strictness preserved

The gate is unchanged in spirit — binary, structural, unforgiving of one flat or near-duplicate thread.
Added only: a **completeness check** (rejects mid-sentence truncation) and explicit **length floor**, both
applied to A and B. No "rescue the weak thread" logic — losers are kept on purpose as negatives.

## Outputs (`corpus/`)

```
set_a_with_consequence/     passers.jsonl (UNLABELED) · passers.md · rejects.jsonl (FULL TEXT) · near_miss.jsonl · summary.json
set_b_without_consequence/  (same shape)
hard_negatives/             negatives.jsonl · near_miss_premium.jsonl · summary.json
MANIFEST.json               counts · config · thresholds · spend
```
`passers.jsonl` stays **unlabeled** `{problem, facets, threads}`. Source/theme/metrics live only in the
`.md` (QA) and `summary`/`MANIFEST` (diagnostics). Negatives are labeled with their failure reason — that
labeling is the point of the contrastive set.

## Run (DO NOT run without sign-off)

```
python generate.py set_a --target 300
python generate.py set_b --target 300
python generate.py aggregate
```

Safety: reads only the OpenRouter + OpenAI keys; writes only under `corpus/`. Hard money cap via a live
ledger (`CORPUS_USD_CAP`, default $8); save-as-you-go; adaptive top-up loop hits the target with minimal
over-generation.
