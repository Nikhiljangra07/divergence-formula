# DAV — the Divergence Criterion (locked spec)

> The single trainable criterion for the divergent-reasoning model, chosen as #1 of
> ten R1-derived candidates. This version folds in three things the architecture
> discussion surfaced: the **wholeness term** (refract, don't dismantle), the
> **two-level gauge** (score the angles AND the threads), and the **runtime-gauge
> role** (regenerate on low score). Plain language first, math boxed, every symbol
> grounded.

---

## 0. The one operation we are measuring

Given a user's problem, produce **k threads that are (1) genuinely different from each
other, (2) each still about the user's actual problem, (3) each refracting the *whole*
problem rather than a fragment of it, and (4) each carrying its own projected
consequence — and never converge them.** The user reads the threads and picks.

DAV is the number that says, honestly, *"did that actually happen, or did the threads
secretly collapse / drift into noise / break into disjoint parts?"*

---

## 1. The criterion in human words

DAV separates into **gates** (hard pass/fail floors — ungameable) and a **score** (graded
quality you rank/threshold by). A set of items only gets scored if it passes every gate.

**Gates — the *worst* item must clear each bar (min, never average):**
- **Grounding:** every thread is close to the problem. *(no thread drifts off into noise)*
- **Wholeness:** every thread engages the *whole* problem, not one slice. *(refract, not dismantle — this is the new term)*
- **Consequence:** every thread carries a real, non-vacuous projected outcome. *(no "further research needed")*

**Score — divergence quality, among sets that pass the gates:**
- **Volume:** how much *space* the threads fill in meaning-space. Big volume = they truly
  point different directions. If any two collapse to the same thing, the volume drops to
  zero by itself. *(this is the part that can't be faked)*

That's the whole idea: **gates stop the three failure modes; volume measures how good the
divergence is.**

---

## 2. The math (boxed)

**Objects.** Let `E(·)` be a fixed (frozen) sentence-embedder mapping text → a unit vector
in ℝ^d. For a problem `P`:
- `p = E(P)` — the problem vector (the anchor).
- `F = {f₁,…,f_m}` — the problem's **facets** (its key sub-aspects), extracted once from `P`; `fⱼ = E(facet_j)`.
- A **set** `X = {x₁,…,x_k}` — either the **angles** (level 1) or the **threads** (level 2).
  `xᵢ = E(itemᵢ)`. For threads, also a consequence `cᵢ` with `validity(cᵢ) ∈ [0,1]`.

**Centered embeddings (for the volume term):** `x̄ = (1/k)∑ xᵢ`, `x̃ᵢ = (xᵢ − x̄)/‖xᵢ − x̄‖`.
Centering measures spread *among the items*; grounding (below) handles absolute position.

**Gram matrix:** `Kᵢⱼ = ⟨x̃ᵢ, x̃ⱼ⟩` (a k×k cosine matrix).

```
GATES  (reject the set unless ALL hold):
  grounding:    min_i  cos(xᵢ, p)            ≥ ε_g
  wholeness:    min_i  coverage(xᵢ, F)       ≥ ε_w        ← refract-not-dismantle
  consequence:  min_i  validity(cᵢ)          ≥ ε_c        ← threads only

SCORE  (rank / threshold the sets that pass):
  DAV(X) = log det( K + λI )                              ← the volume / divergence

where  coverage(xᵢ, F) = (1/m) ∑_j cos(xᵢ, fⱼ)           ← how broadly thread i
                                                            engages the whole problem
  λ  = ridge (e.g. 1e-3); REQUIRED — centering forces rank(K) ≤ k−1, so λI makes det>0
  ε_g, ε_w, ε_c, λ  = dials, set once on a few hand-checked examples

  (full precision, both formula variants, and all guards: see §11 — canonical)
```

**Why each piece does its job:**
- `log det(K+λI)` → the squared volume of the parallelepiped the threads span. Two
  identical threads ⇒ K rank-deficient ⇒ `log det → −∞` (score craters). **Collapse is
  punished automatically.**
- `min_i cos(xᵢ, p)` → the *worst* thread must still point at the problem. One nonsense
  thread can't hide in an average. **Drift is blocked.**
- `min_i coverage(xᵢ,F)` → the *worst* thread must still touch *all* the problem's facets.
  A **fragment** (a "part") touches ~one facet ⇒ low coverage ⇒ fails. An **angle** (a
  whole-problem refraction) touches them all from one stance ⇒ passes. **Dismantling is
  blocked.** Crucially, wholeness and volume are *orthogonal*: all threads can cover all
  facets (wholeness) while reaching different conclusions (volume) — same problem, different
  angles.
- `min_i validity(cᵢ)` → every thread carries a real consequence. **Vacuity is blocked.**

---

## 3. Two-level gauge (where DAV runs)

DAV is the **same formula applied twice** — once to the angles, once to the threads.

| | Input set `X` | Gates run | What it guards |
|---|---|---|---|
| **Level 1 — angles** | the k angle-framings from the refractor | grounding + **wholeness** | refraction quality: are the angles distinct, on-problem, and *whole* (not parts)? |
| **Level 2 — threads** | the k worker outputs | grounding + wholeness + **consequence** | final output: did the threads stay divergent + grounded + consequential? |

Consequence is a thread-only gate (angles don't have consequences yet).

---

## 4. The harness loop (where each gauge plugs in)

```
user → problem P
  │
  ├─ [refractor call]  P → facets F  +  k angles A = {a₁…a_k}     (the model, one call)
  │
  ├─ GATE-1: DAV(A)   grounding + wholeness + volume
  │        fail → re-refract (≤ R retries; nudge "make angles more distinct / more whole")
  │
  ├─ for each angle aᵢ  (ISOLATED — blind to the other angles & threads):
  │     [worker call + web search]  (P, aᵢ, web) → thread tᵢ + consequence cᵢ   (×k, parallel)
  │
  ├─ GATE-2: DAV(T)   grounding + wholeness + consequence + volume
  │        fail → regenerate the weak thread(s), or re-refract if the set collapsed
  │
  └─ present k threads to the user  (NO convergence)  →  user picks thread(s)
```

Two ingredients keep it genuinely divergent, both required:
1. **Distinct seeds** — the refractor gives each worker a *different angle* (without this,
   same model + same input ⇒ k copies of one answer).
2. **Isolation** — each worker is blind to the others (without this, autoregressive
   convergence-creep: later threads rhyme with earlier ones).

---

## 5. Provenance — every term traces to a real pipeline stage

| DAV piece | Comes from |
|---|---|
| `log det(K)` volume | **Governor CLOSE** (halts when the structure/volume plateaus) + **Blender** distinct-thesis preservation (never merges → high-rank set) |
| `min cos(xᵢ,p)` grounding | **Laundering** (leads derived from the anchor) + **Wander** structural-match (cards must match the anchor) + **Coverage D_t** |
| `min coverage(xᵢ,F)` wholeness | the **refract-not-dismantle** principle + **HALO auditor** (full-problem facets / blind spots) + **Coverage D_t** (angles span the question) |
| `min validity(cᵢ)` consequence | **Formalizer** (predict/confirm/falsify test) + **Blender** consequence projection |
| the `min` floors | the **chaos-law** discipline — every seat independently clears the bar |

This is the line you put in the paper: *the objective is not hand-designed; it is the
conserved quantity of a pipeline we actually run.*

---

## 6. How to compute it (cheap, small k ≤ ~6)

- **Embedder `E`:** a frozen small sentence-transformer (bge-small / e5-small / MiniLM).
  CPU-cheap, no training.
- **`K` and `log det`:** `numpy.linalg.slogdet(K + λI)` — one line.
- **`coverage`:** extract `m` facets from `P` once (a cheap LLM call, or KeyBERT /
  noun-phrase extraction), embed them, take mean cosine. Or an LLM judge (1–5 → [0,1]).
- **`validity(cᵢ)`:** small LLM judge or classifier; gates vacuous consequences.

No mutual-information estimation, no optimal-transport solving, no persistent homology.
Every term is an embedding op on ≤6 short texts.

---

## 7. What one training example looks like (end to end)

Run the harness on a problem `P`. Keep the run **only if its final DAV(T) passes + scores
high** — so the model only ever learns from divergence-clean runs.

- **Decomposer example (the crown jewel):**
  `input:  P`  →  `target:  F (facets) + A (k distinct, whole, on-problem angles)`
  *Teaches: given a problem, surface its facets and refract it into k whole-problem angles.*
- **Worker example:**
  `input:  (P, aᵢ, web_context)`  →  `target:  (tᵢ, cᵢ)`  for a thread that passed the gates.
  *Teaches: given one angle + material, write a grounded whole-problem thread with a real
  consequence.*

Train the **decomposer first / most** — divergence is decided there. The worker is the
easier role.

---

## 8. Failure modes → which gate catches them

| Failure | Caught by |
|---|---|
| Refractor **dismantles into parts** | wholeness gate (low coverage) at GATE-1 |
| Workers **converge** despite isolation (shared-weight mode) | low `log det` volume at GATE-2 |
| **Surface divergence** (same idea, reworded) | low volume (embeddings collapse) |
| **Divergent nonsense** (ungrounded) | grounding gate |
| **Vacuous consequence** ("do more research") | consequence gate |

---

## 9. Honest limits (say these out loud)

- DAV scores a **set**, so it is a **gate / reward / eval / corpus-filter — not** a
  differentiable token loss you backprop through generation.
- It is **embedding-based**: surface-vs-semantic divergence detection is only as good as
  the embedder. Use a real semantic small embedder, not a bag-of-words.
- **Wholeness** adds one facet-extraction step; its quality depends on that extraction.
- Keep **k small (≤6)** for a stable `log det`; for larger k raise `λ`.
- It measures the **structural property (divergence)**, not wisdom. On a 3B/7B model that
  is the honest claim — do not oversell thread quality.

---

## 10. The locked next step

1. Implement `dav(X, p, F)` (≈30 lines: embed → centered Gram → `slogdet` → three min-gates).
2. Stand up the **minimal harness** (§4) around the **untrained base Granite** model + web search.
3. Run it on ~10 real problems, print DAV(angles) and DAV(threads) per run.
4. **Read the threads yourself** and check DAV agrees with your eye (tune ε_g, ε_w, ε_c, λ).
5. Only then fine-tune — decomposer first — on the runs that passed.

*Validate the cheap thing (harness on base model) before paying for the expensive thing
(the fine-tune).*

---

## 11. Ablation & final formulas — CANONICAL (train + present from this section)

This section is the authoritative reference. It supersedes the shorthand in §2 with full
precision: every symbol typed, both corpus variants, the single difference between them,
and every guard. Nothing here is approximate.

### 11.1 The experiment

Two training corpora differing by **exactly one factor** — the consequence gate:

| Corpus | Formula | Consequence per thread? | Trains |
|---|---|---|---|
| **A — full** | `DAV_full` | yes (`Cᵢ` appended, gate active) | divergence **+** decidable path-outcomes |
| **B — lite** | `DAV_lite` | no (`Cᵢ` absent, gate removed) | divergence only (bare angles) |

Both are fine-tuned **from the same base model**, on the **same problems / angles / threads**;
only the consequence is added-or-not. We then test both, and report the tradeoff
(divergence + sharpness gained by dropping consequence) vs (decidability lost). This is a
controlled **ablation of one gate**, not two products. It is a thesis demonstration, not a
production system.

### 11.2 Definitions (every symbol, once)

| Symbol | Space | Meaning |
|---|---|---|
| `P` | text | the user's problem |
| `T₁,…,T_k` | text | the `k` threads the model produces (`k ≥ 2`, decided per-problem) |
| `C₁,…,C_k` | text | the consequence of each thread — *where the path leads* (**Corpus A only**) |
| `Φ₁,…,Φ_m` | text | the `m` facets of `P` (its key sub-aspects), extracted once (`m ≥ 1`) |
| `E(·)` | text → ℝ^d | frozen sentence-embedder, **L2-normalized output** (‖E(·)‖ = 1) |
| `p = E(P)` | ℝ^d | unit embedding of the problem |
| `xᵢ = E(Tᵢ)` | ℝ^d | unit embedding of thread `i` |
| `φⱼ = E(Φⱼ)` | ℝ^d | unit embedding of facet `j` |
| `x̄ = (1/k) Σᵢ xᵢ` | ℝ^d | mean thread embedding (centroid) |
| `x̃ᵢ = (xᵢ − x̄)/‖xᵢ − x̄‖` | ℝ^d | thread `i` centered (centroid removed), then renormalized |
| `K ∈ ℝ^{k×k}`, `Kᵢⱼ = ⟨x̃ᵢ, x̃ⱼ⟩` | — | cosine Gram of centered threads (`Kᵢᵢ = 1`) |
| `λ > 0` | scalar | ridge (e.g. 1e-3) — REQUIRED (see guard 2) |
| `I` | ℝ^{k×k} | identity |
| `val(Cᵢ) ∈ [0,1]` | text → [0,1] | consequence-validity scorer (LLM-judge or classifier); 1 = real, 0 = vacuous |
| `ε_g, ε_w, ε_c ∈ [−1,1]` | scalars | gate thresholds (dials) |

All cosines are inner products because every embedding is unit-norm: `cos(a,b) = ⟨a,b⟩`.

### 11.3 Formula A — full (Corpus A)

```
DAV_full(X) = log det(K + λI)

accepted  ⟺   min_{1≤i≤k} ⟨xᵢ, p⟩                    ≥ ε_g      (grounding)
          ∧   min_{1≤i≤k} (1/m) Σ_{j=1}^m ⟨xᵢ, φⱼ⟩   ≥ ε_w      (wholeness)
          ∧   min_{1≤i≤k} val(Cᵢ)                     ≥ ε_c      (consequence)

If any gate fails:  DAV_full(X) := −∞   (reject)
```

### 11.4 Formula B — lite (Corpus B)

```
DAV_lite(X) = log det(K + λI)

accepted  ⟺   min_{1≤i≤k} ⟨xᵢ, p⟩                    ≥ ε_g      (grounding)
          ∧   min_{1≤i≤k} (1/m) Σ_{j=1}^m ⟨xᵢ, φⱼ⟩   ≥ ε_w      (wholeness)

If any gate fails:  DAV_lite(X) := −∞   (reject)
```

### 11.5 The single difference (the independent variable)

```
DAV_full  =  DAV_lite  +  the one gate   min_i val(Cᵢ) ≥ ε_c
```

Identical between A and B: the volume term, `K`, the centering, `λ`, the embedder `E`, the
facets `F`, the grounding gate, the wholeness gate. Corpus B is Corpus A with the
consequence gate deleted **and** the `Cᵢ` text not appended. Nothing else moves.

### 11.6 Guards & edge cases (so not one variable can misbehave)

1. **`k ≥ 2`.** Divergence needs ≥ 2 threads (`k = 1` has no volume).
2. **The ridge `λI` is structurally required, not just numerical.** Centering forces
   `Σᵢ(xᵢ − x̄) = 0`, so the centered vectors lie in a `(k−1)`-dimensional subspace ⇒
   `rank(K) ≤ k−1` ⇒ `det K = 0`. Adding `λI` (`λ > 0`) makes `K + λI` positive-definite, so
   `det > 0` and `log det` is finite. **Never set `λ = 0`.**
3. **Total-collapse guard.** If `‖xᵢ − x̄‖ < η` for any `i` (e.g. `η = 1e-6`) — threads
   collapsed onto the centroid — `x̃ᵢ` is undefined; define `DAV := −∞` (reject). Intended:
   collapse scores worst.
4. **Embeddings must be unit-norm** (so `cos = ⟨·,·⟩`). If `E` isn't normalized, divide by
   the norm first.
5. **`m ≥ 1` facets**, extracted once from `P`, shared by the wholeness check in both formulas.
6. **Thresholds live in `[−1,1]`** (compared against cosines). Tune `ε_g, ε_w` (and `ε_c` for
   A) on a handful of hand-labeled good/bad sets before any training.

### 11.7 Why rank `k−1` is correct (not a bug)

`k` threads define a `(k−1)`-dimensional simplex (3 points → a 2-D triangle; 4 points → a
3-D tetrahedron). DAV measures **that simplex's volume.** Removing the centroid (centering)
is exactly right — it strips the shared "all-near-the-problem" component so the volume
reflects *only how the threads differ from each other.* Grounding (closeness to `P`) is
handled separately by its own gate. Divergence and grounding never contaminate each other.

### 11.8 Measurement protocol (what to report)

Train A and B from the same base; on a held-out set of problems, report per-corpus:
- **Divergence:** mean `log det(K + λI)` over threads (higher = more divergent).
- **Grounding / Wholeness:** mean `min_i ⟨xᵢ,p⟩`, mean `min_i coverage`.
- **Sharpness:** mean thread length + a coherence check (small model staying focused).
- **Decidability (A only):** quick human "could you pick a thread from its consequence?" rate.
- **The ratio:** `(divergence + sharpness gained by dropping consequence)` vs
  `(decidability lost)`. One sentence is the deliverable:
  *"the consequence gate buys X% decidability at the cost of Y% divergence / Z% sharpness on
  this model size."*

**Discipline:** one variable only. Same problems, same angles, same threads — consequence
appended (A) or not (B). Any other change confounds the result and voids it.

### 11.9 Implementation note — `corpus_run/` (spec ↔ code, stay honest)

The running generator is [`corpus_run/`](./corpus_run/) (`config.py` · `dav.py` · `generate.py`).
The objective and gates map 1:1 to the boxed formulas above: `log det(K + λI)` (`dav.volume`),
grounding `min_i⟨xᵢ,p⟩` (`(t@p).min()`), wholeness `min_i (1/m)Σ_j⟨xᵢ,φⱼ⟩` (`(t@f.T).mean(1).min()`).
Corpus A = `WORKER_A` (thread carries the consequence) **+** consequence gate ON; Corpus B = `WORKER_B`
(thread stops before any outcome) **+** gate OFF — the single variable is held in *both* generation and
gate. Two honest deltas between this code and the ideal formula:

1. **`val(Cᵢ)` is a regex proxy, not a learned validity score.** Code realizes it as a binary
   keyword check (`CONS_RE`) + the ≥`MIN_WORDS` floor ⇒ effectively `val ∈ {0,1}`, `ε_c = 1`.
   *Upgrade path:* swap `CONS_RE` for an LLM-judge to recover the continuous `val(·) ∈ [0,1]` the spec defines.
2. **Three operational floors not in the boxed formula, applied identically to A *and* B** (so they
   cannot confound the one-variable ablation): `pairdist ≥ PD_FLOOR` (near-collapse catch **and** the
   ranker — volume is flat among good sets, so pairdist is what orders them), `length ≥ MIN_WORDS`, and a
   **completeness** check (thread must end on terminal punctuation; rejects mid-sentence truncation).

Calibrated thresholds in code: `VOL_GATE=−8.0`, `EPS_G=0.27`, `EPS_W=0.18`, `PD_FLOOR=0.30`,
`MIN_WORDS=18`, `NEAR_MARGIN=0.06` (validated on the 50-passer pilot).
