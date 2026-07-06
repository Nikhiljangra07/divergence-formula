# MASTER — Divergent Reasoning Model (project memory)

> **The single source of truth for this project.** Everything — what it is, the two
> models, the thesis, the architecture, the two corpora, the formula — lives here in
> summary. The complete mathematical spec (every symbol, both formula variants, all
> guards) is the companion document **[`DAV_SPEC.md`](./DAV_SPEC.md)** — read it for the
> precise criterion; read this for the whole picture.
>
> _Folder: `~/Desktop/divergence-formula/` · isolated, touches no production code._

---

## 1. What this is

The **third standalone portfolio project**, alongside (1) Constellax — the multi-agent
autonomous reasoning pipeline — and (2) the Waveform Engine. This one is a **small language
model that reasons divergently**: given a problem, it refracts the problem into multiple
distinct angles, returns multiple **non-converging threads**, and lets the **user pick**
which thread to pull. It never collapses to a single answer.

It is a **distillation**: the conserved operation of the Constellax pipeline (multi-angle
divergence, deliberately un-converged) is compressed down to model scale.

**Status:** thesis demonstration / portfolio piece — **not** a production or beta system.
The claim is the **structural property (divergence)**, not deep wisdom. (A small model can
be made to *diverge well*; it cannot be claimed to *reason wisely* at this size — that
honesty is part of the thesis.)

---

## 2. The two models

Both are **IBM Granite 4.0**. Trained on **RunPod** (cloud only; ~$200 credit). This is also
a first-time model-training skills exercise — two fine-tunes, two architectures.

| Order | Model | Size / type | Why |
|---|---|---|---|
| **1st** | `granite-4.0-micro` | **3B, pure dense transformer** | the clean, dependency-free baseline — learn the full train→eval loop on the model least likely to fight you |
| **2nd** | `granite-4.0-h-tiny` | **7B, hybrid (Mamba-2 + MoE)** | adds the rare skills (Mamba kernels, MoE routing); a second architecture to compare divergence-distillation against |

The base model **stays in place** — we only change the **weights** via fine-tuning. The
criterion (below) is **not run at inference**; it is the training methodology, baked into
the weights. (A thin harness may recompute it cheaply as a runtime gauge — that is code on
the side, not the model doing math.)

---

## 3. The thesis & the honest frame

- **Claim:** a multi-agent pipeline's divergent-reasoning behavior can be transferred into a
  single small model and **measured** (does the structural property survive distillation?).
- **Not claimed:** that the small model out-reasons frontier models. It won't, and we don't
  pretend it does.
- **What makes it credible:** it is a **controlled ablation** (§5), not a single fine-tune —
  product (the model) + the research that validates it.

---

## 4. The architecture — REFRACT, don't dismantle

A thin code **harness**, one model in every seat (the way Claude is called many times by its
harness; the way Constellax orchestrates in code). The model is the worker/decomposer; the
orchestration is code.

```
user → problem P
  ├─ [refractor]  P → m facets + k distinct ANGLES         (k decided per-problem)
  ├─ GATE-1: DAV(angles)   → re-refract if it collapses
  ├─ for each angle (ISOLATED, blind to the others) + web search → one thread (+ consequence)
  ├─ GATE-2: DAV(threads)  → regenerate weak thread(s) if it collapses
  └─ present k threads to the user  (NO convergence)  →  user picks
```

**Two non-negotiables for real divergence:**
1. **Distinct seeds** — each worker gets a *different angle* (same model + same input ⇒ k
   copies of one answer).
2. **Isolation** — each worker is blind to the others (kills autoregressive
   convergence-creep, where later threads rhyme with earlier ones).

**REFRACT, not dismantle** (the load-bearing principle): each worker gets the **whole problem
through one angle/lens**, NOT a disjoint *piece* of it. Pieces are divide-and-conquer →
they reconverge into one answer (a checklist, not angles). Angles compete and coexist → the
user recognizes every thread as *their* problem and chooses. Diagnostic: *"does each thread
still feel like MY problem?"* — only angle-decomposition passes.

**No blender.** Constellax's convergent synthesis stage is **removed** — it would defeat
divergence. The **user is the synthesizer.** The pipeline organs we keep are the *quality*
organs (they serve divergence): drift-checker → grounding, HALO → wholeness, formalizer →
consequence.

---

## 5. The two corpora (the ablation)

We build **two training corpora** that differ by **exactly one factor** — the consequence —
fine-tune both from the same base, and report the tradeoff.

| Corpus | Threads carry a consequence? | Formula | Trains |
|---|---|---|---|
| **A — full** | yes — *where each path leads*, revealed when the user picks the thread | `DAV_full` | divergence **+** decidable path-outcomes |
| **B — lite** | no — bare angles | `DAV_lite` | divergence only |

**Why two:** for a small model kept sharp and focused, a consequence on *every* thread may
be redundant and may even hurt (a 3B can project consequences shallowly). "Consequence" here
means the **path-outcome** ("if you pull thread B, here is where it leads") — the
*decision-information* that lets the user choose — **not** a generic caveat/flag. It is core
at scale but plausibly *nice-to-have to initiate*. The ablation tells us, with a number,
what the consequence gate is worth: **divergence + sharpness gained by dropping it vs
decidability lost.** Deliverable = that tradeoff ratio.

**Discipline:** one variable only — same problems, same angles, same threads; consequence
appended (A) or not (B). Any other difference confounds the result.

---

## 6. The criterion — DAV (summary; full spec in `DAV_SPEC.md`)

Chosen as **#1 of ten** candidate formulas, each derived by an independent web-enabled
DeepSeek-R1 agent through a different branch of mathematics (info-theory, geometry,
game-theory, dynamical-systems, optimal-transport, stat-mech, category-theory, topology,
evolutionary-ecology, quantum). DAV (the determinantal / geometric one) won on
trainability + ungameability + clarity.

**Both formula variants (canonical precision in [`DAV_SPEC.md`](./DAV_SPEC.md) §11):**

```
DAV_full(X) = log det(K + λI)
   s.t.  min_i ⟨xᵢ,p⟩ ≥ ε_g        (grounding   ← drift-checker)
         min_i (1/m)Σ_j ⟨xᵢ,φⱼ⟩ ≥ ε_w  (wholeness  ← HALO)
         min_i val(Cᵢ) ≥ ε_c        (consequence ← formalizer)

DAV_lite(X) = log det(K + λI)
   s.t.  min_i ⟨xᵢ,p⟩ ≥ ε_g        (grounding)
         min_i (1/m)Σ_j ⟨xᵢ,φⱼ⟩ ≥ ε_w  (wholeness)
```

**In one line:** *spread the threads as wide as possible (`log det` volume = divergence), as
long as the worst thread is still on the problem (grounding), still covers the whole problem
(wholeness), and — in A only — still carries a real consequence.* The volume can't be faked
(two collapsed threads drop it toward `−∞`); the gates are hard floors on the *worst*
thread, so nonsense can't hide in an average.

**Roles of DAV:** (a) **corpus filter** at data-generation time (keep only divergence-clean
runs as training data), (b) **runtime gauge** in the harness (regenerate on low score), (c)
**eval / benchmark** number for the write-up. It is **not** a differentiable loss.

---

## 7. Provenance — formula ⟵ pipeline

The criterion is **not hand-designed** — it is the conserved quantity of a pipeline we
actually run (Constellax). Each term traces to a real stage:

| DAV term | Constellax organ |
|---|---|
| `log det(K)` volume (divergence) | Wander + Governor CLOSE + Blender distinct-thesis preservation |
| `min ⟨xᵢ,p⟩` grounding | Laundering + Wander structural-match + Coverage Dₜ |
| `min coverage` wholeness | refract-not-dismantle + HALO auditor + Coverage Dₜ |
| `min val(Cᵢ)` consequence | Formalizer (predict/confirm/falsify) |
| the `min` floors | the chaos-law discipline (every seat clears the bar independently) |

---

## 8. Artifacts in this folder

| File | What |
|---|---|
| **`MASTER.md`** | this document — the whole picture |
| **`DAV_SPEC.md`** | the complete criterion: every symbol, both formulas, all guards, harness, training-example shape (§11 is canonical) |
| `derive_formula_swarm.py` | the 10 web-enabled R1 agents that generated the candidate formulas (generate-only; hard $8 money cap; ~$0.36/run) |
| `derive_formula.py` | earlier single-thread R1 derivation + the vetted pipeline ground-truth |
| `out/swarm_formula_*.md` | the 10 candidate formulas + proofs (one per math lens) |
| `out/swarm_formulas.md` | all 10 concatenated |
| `out_v1/` | first single-thread run (partial; superseded) |
| `README.md` | how the R1 swarm works + safety/isolation notes |

OpenRouter wallet (R1 swarm) ≠ RunPod credit (training) — separate.

---

## 9. Locked next steps

1. Implement `dav(X, p, F)` (~30 lines: embed → centered Gram → `slogdet` → the min-gates).
2. Stand up the minimal harness (§4) around the **untrained base `granite-4.0-micro`** + web search.
3. Run ~10 real problems; print DAV(angles) and DAV(threads); **read the threads yourself**
   and tune `ε_g, ε_w, ε_c, λ` until DAV agrees with your eye.
4. Build **Corpus A and Corpus B** from the divergence-clean runs (one variable apart).
5. Fine-tune `granite-4.0-micro` on each — **decomposer first** (divergence is decided there).
6. Test, compare, report the tradeoff ratio (§5 / `DAV_SPEC.md` §11.8).
7. Then repeat on `granite-4.0-h-tiny` (hybrid).

*Validate the cheap thing (harness on the base model) before paying for the expensive thing
(the fine-tune).*

---

## 10. Boundaries

- **Separate from Constellax and LoRa** at brand / code / data level. No LoRa-prefixed
  identifiers anywhere in this work.
- Constellax is the *source* (the teacher whose conserved operation we distil); this project
  is its own thing.

---

## 11. Citation audit (2026-06-24, verified against live web)

Every citation in `out/` was checked against the live web.

- **No hallucinated links.** Every cited arXiv/ACL/PMLR ID resolves to a real paper —
  including `2605.28465` (*Beyond One Path… Divergent Thinking in Interactive LLM Agents*,
  May 2026, which introduces **MUTATE/ReDNA**), DiMo `2510.16645`, RPD `2510.26122`, DoT
  (EMNLP 2024 `.992`), SBERT `1908.10084`, DeBERTa `2006.03654`, Kulesza & Taskar DPP
  `1207.6083`. A prior "hallucinated 2026 arXiv links" flag was a **false positive** — it
  assumed any 2026 ID must be fake; the swarm had live web search and cited real recent work.
- **Misattributions corrected** (real papers attached to claims they don't support):
  - `topology_persistence`: "stability" / "birth-death intervals" were cited to `2605.28465`
    and `yu25k` (a GFlowNet paper) → corrected to **Cohen-Steiner, Edelsbrunner & Harer 2007**
    (the real persistence-stability theorem, `10.1007/s00454-006-1276-5`).
  - `statistical_mechanics`: "Landau theory" cited to `2605.28465`, "Free Energy Principle"
    cited to `2106.02583` (a control-theory paper) → links removed (claims stand on textbook
    statistical mechanics, already covered by Feynman). The orphaned "MUTATE Benchmark"
    (`2605.28465`) entry left in the Summary-Card Sources list was also removed.
  - `category_order_theory`: "lattice/order theory" cited to `2605.28465` → corrected to
    **Dilworth's theorem** (antichains in a poset).
  - `quantum_superposition`: hyperparameter tuning cited to PSIS `1507.02646` → reworded to
    "tuned on a held-out validation set."
- **The winner (DAV / `determinantal_geometry`) is not citation-free.** It cites three
  sources: Kulesza & Taskar DPP (`1207.6083`, clean), Johnson–Lindenstrauss (clean), and
  `2605.28465` — used for an "entropy-based nonsensity filter" to detect gamed consequences.
  That last cite is **loose**: `2605.28465` is a divergent-thinking benchmark (MUTATE) + an
  idea-generate-then-filter method (ReDNA), so it only weakly supports a "nonsense filter"
  and supplies no entropy-based mechanism. It is a supporting pointer, not a load-bearing
  proof, and DAV's two core results (log-det volume, JL near-orthogonality) do not rely on
  it — but it is a citation in the winner and is recorded here honestly, not as "clean."

---

*Companion: [`DAV_SPEC.md`](./DAV_SPEC.md) — the additional, fully-precise document of this
same project.*
