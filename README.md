# divergence-formula — distilling divergent reasoning into small open models

Frontier chat models converge: ask a hard question, get one blended answer. This project trains small
open models to do the opposite — **refract** a hard decision into four genuinely distinct, viable
strategic threads, each a different *kind* of move, and refuse to blend them.

Eight rounds, two trained models, every result (including the negative ones) recorded in
**[`WORKING_PAPER.md`](WORKING_PAPER.md)** — the project's centerpiece document.

## The two models

| | Base | Method | Role |
|---|---|---|---|
| **v5 SFT** | IBM granite-4.0-micro (3.4B dense) | LoRA SFT on 2,285 synthetic judge-gated rows | proof the skill is teachable at all |
| **BLEND2** (shipped) | IBM granite-4.0-h-small (32B total / ~9B active hybrid Mamba+MoE) | LoRA SFT on 6,745 rows (balanced sources + a targeted "viable-cunning" round) | the real ship |

## Final scoreboard

Single-session judging (Gemini 2.5 Pro, temp 0), 48 out-of-distribution modern decision problems,
Claude Haiku 4.5 as the frontier competitor generated fresh in the same session:

| model | overall | viability | **distinctness** | decisiveness | foresight |
|---|---|---|---|---|---|
| 3.4B dense SFT | 4.24 | 2.81 | ~4.8 | — | 3.81 |
| **H-Small BLEND2 (ship)** | **4.55** | 3.56 | **4.90** | **4.83** | 4.21 |
| Claude Haiku 4.5 | 4.72 | 4.25 | 4.75 | 4.83 | 4.52 |

**The shipped specialist beats Haiku on the objective it was built for (distinctness 4.90 vs 4.75) and
ties it on decisiveness** — with ~9B active parameters against a frontier model. It loses overall
(−0.17); the residue is viability and foresight, and the paper says so plainly.

## The five findings the journey produced (details + evidence in the paper)

1. **Refraction is teachable** to small open models from purely synthetic corpora — but plateaus
   (~2.75/5 distinctness on 3.4B, triangulated three ways).
2. **Corpus quality beats corpus size, every time it was tested.** The plateau broke via source
   rebalancing, not scale; later, 850 laser-targeted examples bought only +0.10 viability.
3. **On a 3.4B, distinctness and viability are capacity-entangled** — preference optimization (DPO)
   with clean hard negatives buys viability only by selling distinctness. A crisp negative result.
4. **The entanglement breaks at ~9B active params** — identical data, both dimensions rise together.
   Foresight, immovable on the 3.4B under every data intervention, jumped +0.73 with capacity alone.
5. **A cheap strong generator saturates a tightly-prompted rubric** — DeepSeek V4 Pro tied Claude
   Sonnet 5 on corpus quality at ~1/7 the price (and beat it on distinctness) in a paired A/B.

Total cost of the entire program — 8 rounds, ~1,400 corpus examples, 6 GPU training runs, 3 benchmark
campaigns: **≈ $175**.

## Repo map

```
WORKING_PAPER.md      ← the full record: every round, every number, every mistake (start here)
MASTER.md             ← condensed project memory (thesis, models, corpora, formula)
DAV_SPEC.md           ← the mathematical criterion (divergence-and-viability) in full
ISOLATION.md          ← data-provenance audit vs the author's other projects
corpus_run/           ← corpus generators (refractor→workers→judge pipelines), benchmark harness,
                        judge scripts; configs for every corpus round (v2…v5, essence)
round2_kit/           ← training + eval kit: train_lora.py, dav_eval_v5.py, prep_* builders,
                        the exact training data (data_v5, data_v5blend, data_blend2)
out*/ dsv4_pilot/ …   ← round-0 artifacts (the original R1 criterion derivation)
```

## Models & reproduction

- **Adapters are not in this repo** (LoRA safetensors, ~GBs) — archived locally; available on request.
  Base models are IBM Granite (Apache-2.0).
- Corpus generation: `corpus_run/gen_v5.py`, `gen_essence_scale.py` (DeepSeek generator + LLM judge
  gate; API keys via env, never committed).
- Training: `round2_kit/train_lora.py` (r64/α64/5ep; H-Small needs the Mamba kernel + optimizer flags
  documented in the paper §13 — the engineering traps are all written down).
- Evaluation: `round2_kit/dav_eval_v5.py` + `corpus_run/judge_blend2_gemini.py` /
  `rejudge_all_gemini.py` (neutral-judge benchmark, same 48 problems across all rounds).

## Honest scope

This is a research program with a specialist's scoreboard, not a general benchmark win. The eval
rubric measures the one operation the models were trained for. The paper's conclusion (§15) names
what was demonstrated, what failed, and which levers remain unrun.
