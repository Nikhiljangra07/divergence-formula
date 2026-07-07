# The $175 Specialist: What Breaks When You DPO a 3.4B Model

*Preference optimization did exactly what I asked. That was the problem — a small model paid for the improvement by selling the one skill it was trained for. Same data on a 9B-active model: both skills rose together. This is the record of that experiment.*

---

Over eight rounds and a total budget of about **$175**, I trained small open models to do one thing frontier chat models refuse to do: **refract** a hard decision into four genuinely distinct, viable strategic threads — four different *kinds* of move — instead of blending everything into one hedged answer. The full program is documented in this repo's [working paper](WORKING_PAPER.md), negative results included. This article is about the most instructive of those negatives: a hard-negative DPO run on a 3.4B model that *worked at its target* and still made the model worse.

If you fine-tune small models, the finding generalizes past my task: **below some capacity, skills you thought were independent are entangled — preference optimization doesn't add ability, it reallocates it.** And you won't see it happen unless your evaluation is set up to catch it.

## The setup, briefly

- **Task:** two-stage refraction. A *decomposer* adapter turns a messy real-world problem into four strategic angles from distinct move families (confront, evade, co-opt, transform, delegate, endure); a *worker* adapter expands each angle into a concrete thread with projected consequences.
- **Model:** IBM `granite-4.0-micro`, 3.4B dense, Apache-2.0. LoRA (r=64, α=64, all-linear), SFT on **2,285 synthetic, judge-gated rows**.
- **Evaluation:** 48 out-of-distribution modern decision problems (career, finance, relationships, ethics), scored per-dimension by **Gemini 2.5 Pro at temperature 0** — a judge from neither my pipeline's model family nor the competitor's — with **Claude Haiku 4.5** run on the same problems as the frontier baseline.

The SFT result that set the stage:

| Gemini 2.5 Pro, 48 OOD problems | 3.4B base | **3.4B SFT** | Haiku 4.5 |
|---|---|---|---|
| distinctness | 2.87 | **4.79** | 4.71 |
| viability | 3.57 | 2.75 | 4.23 |
| overall | 3.74 | 4.21 | **4.73** |
| threads with distinctness ≥ 4 | 38% | **100%** | 94% |

Two things are true at once here, and both matter. The 3.4B specialist **beats a frontier model on the objective it was trained for** — distinctness 4.79 vs 4.71, with 100% of threads clearing the bar versus Haiku's 94%. And its **viability fell to 2.75 — below the untrained base model's 3.57**. Training didn't just fail to improve viability; it *spent* viability to buy distinctness. That asymmetry was the first clue, though I didn't read it correctly until later.

## The experiment: hard-negative DPO

Viability was the obvious repair target, and DPO the obvious tool: keep the distinct structure the SFT had learned, and use preference pairs to push probability mass away from over-reaching threads (unrealistic timelines, budgets, assumed cooperation) toward grounded ones.

Getting the negatives right took two honest failures:

1. **Easy negatives taught nothing.** The first bank used obviously-bad rejected samples — baroque, illegal, fantasy threads, a ~2.1-point quality gap from the chosen sample. Training regressed at both β=0.1 and β=0.3. A negative the model would never have produced provides no useful gradient. This failure mode is well known; I reproduced it anyway, on purpose, before trusting the fix.
2. **The first "hard" batch was too timid — and only an integrity check caught it.** Hard negatives were defined as *minimally-perturbed rewrites of the chosen thread carrying exactly one substantive viability over-reach*, move family held fixed. The pilot validated that the preference gap pointed the right way — but not that it was big enough. Result: 22% of pairs had chosen == rejected verbatim, and 37% differed by less than 5% of characters. A pre-training integrity gate caught it; the generation prompt was rebuilt with a forced-over-reach requirement and an inline trivial-reject guard.

The final bank: **1,816 clean pairs** — median 51% character difference, zero identical pairs, zero leakage into held-out — generated for **$2.93**. DPO from the SFT worker checkpoint, β=0.1, one epoch.

## The result: it worked, and it made the model worse

| Gemini 2.5 Pro, 48 OOD | 3.4B SFT | 3.4B DPO-hard | Δ |
|---|---|---|---|
| viability (the target) | 2.75 | 3.08 | **+0.33** ✔ |
| distinctness | 4.79 | 4.25 | **−0.54** |
| decisiveness | 4.81 | 4.29 | −0.52 |
| foresight | 3.67 | 3.25 | −0.42 |
| overall | 4.21 | 3.98 | −0.23 |
| threads with distinctness ≥ 4 | 100% | 77% | −23 pts |

The hard negatives did their job: viability rose by exactly the kind of margin the mechanism promised. But the model paid for it out of every other dimension — most expensively **distinctness, which fell below Haiku (4.25 vs 4.71)**. The entire competitive edge of the specialist, gone. Net effect: a worse model that was better at the one thing I'd measured the pairs against.

One methodological detail deserves its own sentence: my cheap in-loop judge (Haiku) had suggested distinctness *rose* after DPO. The neutral judge corrected it. **If a preference-training run looks like a free win on your in-loop grader, re-score it with a judge that has no stake in your pipeline before believing anything.**

## Why this is a capacity result, not a data result

The lazy reading is "the DPO data was bad." It wasn't — the bank was clean, the target moved, the mechanism was validated end-to-end. The evidence says something more specific: **on this 3.4B, distinctness and viability draw on the same limited representational budget. The model can be distinct, or more viable, but not both — DPO doesn't grow the budget, it moves probability mass across it.**

That's a falsifiable claim, and the program tested it the only way that's conclusive: change *nothing but capacity*. The **identical 2,285 SFT rows** were trained onto `granite-4.0-h-small` — a 32B-total hybrid Mamba+MoE with **~9B active parameters**:

| Gemini 2.5 Pro, 48 OOD | 3.4B SFT | 9B-active SFT (same data) |
|---|---|---|
| viability | 2.75 | 3.47 (**+0.72**) |
| distinctness | 4.79 | 4.87 (**+0.08**) |
| foresight | 3.67 | 4.40 (**+0.73**) |
| overall | 4.21 | **4.56** |

Same corpus. Both entangled dimensions **rose together**. Foresight — which had refused to move on the 3.4B under every data intervention I tried — jumped +0.73 from capacity alone. And the viability trade didn't vanish at 9B active; it got roughly **3× cheaper** (trained viability sat −0.26 below the bigger base, versus −0.82 below base on the 3.4B). The wall is real at both scales; it's just much farther away at 9B.

Two control observations close the loop. The bigger base model was *no better at refraction untrained* (H-Small base ≈ 3.4B base on this task) — so the skill lives in the corpus, and capacity determines how much of it the model can hold without cannibalizing something else. And the shipped 9B model went on to score distinctness **4.90 vs Haiku's 4.75** in a later same-session benchmark — the specialist thesis survived; only the small-model version of it died.

## What I'd tell you before your own DPO run

1. **Define the trade before you train.** Decide which dimensions you are *not* willing to spend, and measure them in the same eval as your target. A preference run scored only on its target metric will always look like it worked.
2. **Validate negative pairs on magnitude, not just direction.** My pilot checked that chosen > rejected; it didn't check *by how much*, and 22% of pairs were literally identical. Gate on character-level difference and duplication before you spend GPU time.
3. **Easy negatives are a null experiment.** If the model would never have produced the rejected sample, DPO has nothing to learn from the pair.
4. **Don't trust the in-loop judge for verdicts.** Use the cheap grader as a gate during generation; use a neutral, uninvolved model at temperature 0 for anything you'll act on.
5. **When DPO reallocates instead of adds, suspect capacity — then test capacity.** The cheapest decisive experiment is the same data on a bigger base. If both dimensions rise together, you've found an entanglement, not a data problem. On my task, that boundary sat somewhere between 3.4B dense and 9B active.

## Scope, honestly

This is one task, one rubric, one judge, 48 problems, and two points on a capacity curve — a documented case study, not a law. The rubric measures the operation the models were trained for, so the scoreboard is a specialist's, not a general benchmark. What I can defend: every number above is reproducible from this repo, the negative-result adapter is published rather than buried, and the entanglement claim survived the strongest test I could afford to throw at it.

## Artifacts

- **Working paper** (all 8 rounds, every number, every mistake): [WORKING_PAPER.md](WORKING_PAPER.md)
- **The 3.4B adapters — including the DPO negative result, shipped on purpose as evidence:** [Nikhil0097/refract-granite-3.4b-v5](https://huggingface.co/Nikhil0097/refract-granite-3.4b-v5)
- **The shipped 9B-active model:** [Nikhil0097/refract-hsmall-blend2](https://huggingface.co/Nikhil0097/refract-hsmall-blend2)
- Corpus generators, DPO pair builder, and the neutral-judge benchmark harness: this repo (`corpus_run/`, `round2_kit/`).

---

*I design and build AI systems end-to-end, working solo, with AI coding tools as the hands and the design, evaluation methodology, and judgment as my job. More of the record: [github.com/Nikhiljangra07](https://github.com/Nikhiljangra07).*
