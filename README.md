# divergence-formula — distilling divergent reasoning into small open models

This project trains small open models to propose four distinct strategic options for a hard
decision. It evaluates the tradeoff between distinctness and viability using synthetic training
data and a model-judged decision-task rubric.

Eight rounds, two trained models, every result (including the negative ones) recorded in
**[`WORKING_PAPER.md`](WORKING_PAPER.md)** — the project's centerpiece document.

## The two models

| | Base | Method | Role |
|---|---|---|---|
| **v5 SFT** | IBM granite-4.0-micro (3.4B dense) | LoRA SFT on 2,285 synthetic judge-gated rows | small-model adaptation experiment |
| **BLEND2** (released adapter) | IBM granite-4.0-h-small (32B total / ~9B active hybrid Mamba+MoE) | LoRA SFT on 6,745 rows (balanced sources + a targeted "viable-cunning" round) | larger hybrid-model experiment |

## Final scoreboard

Single-session judging (Gemini 2.5 Pro, temp 0), 48 out-of-distribution modern decision problems,
Claude Haiku 4.5 as the frontier competitor generated fresh in the same session:

| model | overall | viability | **distinctness** | decisiveness | foresight |
|---|---|---|---|---|---|
| 3.4B dense SFT | 4.24 | 2.81 | ~4.8 | — | 3.81 |
| **H-Small BLEND2 (ship)** | **4.55** | 3.56 | **4.90** | **4.83** | 4.21 |
| Claude Haiku 4.5 | 4.72 | 4.25 | 4.75 | 4.83 | 4.52 |

**BLEND2 received a higher distinctness point score (4.90 vs 4.75) and the same decisiveness
score in this judge session.** It scored lower overall (4.55 vs 4.72), including lower viability
and foresight. These single-judge, 48-problem comparisons do not establish general superiority,
and the repeatedly reused problems are development diagnostics rather than a fresh final test.

## The five findings the journey produced (details + evidence in the paper)

1. **Synthetic-data adaptation changed the target behavior.** Early 3.4B runs plateaued near
   2.75/5 distinctness under the then-current evaluation; later interventions changed the outcome.
2. **Source balancing mattered in the tested recipes.** Rebalancing helped, while a later
   850-example targeted addition produced only +0.10 viability in the reported comparison.
3. **The tested 3.4B DPO recipe traded distinctness for viability.** This is evidence about that
   optimization setup, not proof of an inherent capacity limit.
4. **The larger hybrid base improved both dimensions in the reported comparison.** Architecture
   and base weights changed with parameter count, so the effect of capacity alone was not isolated.
5. **The generator comparison favored the cheaper option under its rubric.** The recorded paired
   A/B found similar corpus-quality scores at lower API cost; this is scoped to those prompts,
   model versions, prices and judge.

Recorded program cost across 8 rounds, 6 GPU training runs and 3 benchmark campaigns: **≈ $175**.
Training-row counts differ by model and release, as shown above.

## Repo map

```
WORKING_PAPER.md      ← the full record: every round, every number, every mistake (start here)
ARTICLE.md            ← essay: "The $175 Specialist" — the DPO capacity-entanglement finding, for a general ML audience
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

- **Adapters are hosted on Hugging Face:** [3.4B v5 SFT](https://huggingface.co/Nikhil0097/refract-granite-3.4b-v5)
  and [H-Small BLEND2](https://huggingface.co/Nikhil0097/refract-hsmall-blend2). They require their
  respective IBM Granite base models; see the model cards for loading instructions and limits.
- Corpus generation: `corpus_run/gen_v5.py`, `gen_essence_scale.py` (DeepSeek generator + LLM judge
  gate; API keys via env, never committed).
- Training: `round2_kit/train_lora.py` (r64/α64/5ep; H-Small needs the Mamba kernel + optimizer flags
  documented in the paper §13 — the engineering traps are all written down).
- Evaluation: `round2_kit/dav_eval_v5.py` + `corpus_run/judge_blend2_gemini.py` /
  `rejudge_all_gemini.py` (neutral-judge benchmark, same 48 problems across all rounds).

## Running it yourself

```bash
git clone https://github.com/Nikhiljangra07/divergence-formula.git
cd divergence-formula
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # corpus + judging harness
# pip install -r requirements-train.txt  # training/eval kit — needs a CUDA GPU
```

API keys: copy `.env.example` and export the variables in your shell (the scripts
read `os.environ`). Corpus generation needs `OPENROUTER_API_KEY` (DeepSeek) and
`GEMINI_API_KEY` (judge gate); the benchmark harness additionally uses
`ANTHROPIC_API_KEY` (Haiku competitor). Without keys, generation/judging scripts
will fail at the first API call — nothing runs against a paid API silently.

- Regenerate a corpus round: `python corpus_run/gen_v5.py` (config at top of file)
- Re-run the neutral-judge benchmark on stored threads: `python corpus_run/rejudge_all_gemini.py`
- Train adapters (GPU pod): `python round2_kit/train_lora.py` — read WORKING_PAPER.md §13
  first; the Mamba-kernel and optimizer traps for the H-Small hybrid are all written down there.
- Exact training data ships in-repo: `round2_kit/data_v5*`, `round2_kit/data_blend2`.

## Honest scope

This is a research program with a specialist's scoreboard, not a general benchmark win. The eval
rubric measures the one operation the models were trained for. The paper's conclusion (§15) names
what was demonstrated, what failed, and which levers remain unrun.
