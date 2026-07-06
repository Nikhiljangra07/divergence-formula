# Working Paper — Training a Small Model to Reason Divergently

> Unofficial running record. Living document — updated as the work progresses.
> Spans the formula derivation (`divergence-formula/`), corpus generation (`corpus_run/`),
> and the RunPod training probe. Last updated: 2026-06-26.

---

> **How to read the section numbering:** sections `## 1`–`## 12` are the original round-1/2
> paper (June 2026). Later rounds were appended as `§9`–`§15` — the § sections are the
> round-by-round records (§9=round 3 … §15=conclusion) and OVERLAP the early plain numbers.
> Cross-references like "§12" inside § sections refer to the early `## 12` unless stated.

## 0. Abstract

We are teaching a small language model (IBM Granite 4.0, 3.4B dense first; 7B MoE-hybrid second) to
take one hard decision problem and **refract** it into several genuinely distinct, non-converging lines
of reasoning that a human then picks among — never collapsing to a single answer. The behavior is
enforced by (a) a **harness** that calls the model in two seats — a *decomposer* and isolated *workers* —
and (b) a fine-tune on a corpus filtered by **DAV**, a geometric criterion that scores how divergent a
set of threads is. v1 is a successful probe: fine-tuning raised measured divergence and the consequence
ablation produced a real, reproducible effect. v2 chased a metric plateau and, in doing so, found the
plateau was **measurement, not the model**: the model produces genuinely distinct strategies (visible to
the eye and to an action-only embedding, corpus 0.382→0.484), but full-prose cosine pairdist was blinded
by shared domain vocabulary (§8b–8c). The thesis holds; the next step is a second training round on a
divergence metric that can actually see the signal.

---

## 1. Thesis & core principle

Every AI gives deep reasoning on *the task*; none gives deep reasoning on *the space of approaches to
the task*. The bet: a small model can be trained to hold open several incompatible strategies at once.

**REFRACT, don't dismantle.** Each worker gets the *whole* problem through *one lens* (like white light
through a prism → distinct colors), **not** a disjoint *piece* of it (divide-and-conquer, which
reconverges into a checklist). Diagnostic: "does each thread still feel like the *whole* problem?"

---

## 2. The criterion — DAV (Determinantal Anchor Volume)

Chosen #1 of 10 candidate formulas, each derived by an independent web-enabled DeepSeek-R1 agent through
a different branch of mathematics (`derive_formula_swarm.py` → `out/swarm_formula_*.md`). DAV won on
trainability + ungameability + clarity. Full spec: [`DAV_SPEC.md`](./DAV_SPEC.md) §11.

**Objective** (over the k thread embeddings, centered + renormalized; cosine Gram K):
```
DAV(X) = log det(K + λI)        λ = 1e-3 (ridge — structurally required; centering drops rank to k−1)
```
**Two variants, differing by exactly one gate (the consequence):**
```
DAV_full(X) = log det(K+λI)  s.t.  grounding ∧ wholeness ∧ consequence      → Corpus A (with consequence)
DAV_lite(X) = log det(K+λI)  s.t.  grounding ∧ wholeness                     → Corpus B (without)
  grounding  : min_i ⟨xᵢ, p⟩            ≥ ε_g   (worst thread still on the problem)
  wholeness  : min_i (1/m)Σⱼ⟨xᵢ, φⱼ⟩    ≥ ε_w   (worst thread still covers the facets)
  consequence: min_i val(Cᵢ)            ≥ ε_c   (worst thread projects a real cost/gain)
  any gate fails → −∞ (reject)
```
All gates use the **worst** thread, so nonsense can't hide behind an average.

**Implementation notes (spec ↔ code, kept honest — `DAV_SPEC.md` §11.9):**
- `val(Cᵢ)` is realized as a **regex proxy** (`CONS_RE`) + a ≥18-word floor (binary, ε_c=1), not a learned
  validity score. Upgrade path: swap for an LLM-judge to recover continuous `val ∈ [0,1]`.
- Three **operational floors** added beyond the boxed formula, applied identically to both corpora:
  `pairdist ≥ PD_FLOOR`, length, and completeness (terminal punctuation). pairdist is also the **ranker**
  (volume is flat among good sets — it's a collapse gate, not an ordering).

**Calibrated thresholds:** `VOL_GATE=−8.0, EPS_G=0.27, EPS_W=0.18, PD_FLOOR=0.30, MIN_WORDS=18`.

**The four numbers we report:** `pairdist` (divergence — the headline), `volume` (collapse gate),
`ground` (on-topic), `whole` (facet coverage). All via OpenAI `text-embedding-3-small`.

---

## 3. Corpus generation pipeline (`corpus_run/`)

```
source brief + ONE theme → DSV4-pro REFRACTOR → {problem, 3 facets, 4 distinct angles}
                         → DSV4-pro WORKER ×4  → one thread per angle, ISOLATED (blind to the others)
                         → DAV gate            → passers / near-miss / rejects
```
- **Generator:** DeepSeek V4 Pro (`deepseek/deepseek-v4-pro`, OpenRouter). A *reasoning* model — its
  reasoning tokens count against `max_tokens`, so generous ceilings (worker 1600, refractor 3000) are
  required or it returns empty/truncated content. Embeddings: OpenAI `text-embedding-3-small`.
- **Theme conditioning:** 52 distinct dilemma-types across four civilizational sources (RI / JPM / PR /
  MB), balanced round-robin — fixes mode-collapse onto one canonical scene per source.
- **Sources (seed divergence from genuinely different reasoning traditions):** Reverend Insanity
  (amoral cultivation strategy), Jin Ping Mei (domestic/economic intrigue), The Prince (statecraft),
  Mahabharata (dharmic dilemmas).
- **Three corpora:** A = with-consequence (`DAV_full`), B = without-consequence (`DAV_lite`, worker
  stops before the outcome — the *single ablation variable* held in both generation and gate),
  C = hard negatives (the gate-failures, full text + which-thread-failed, for later DPO/contrastive).
- **Safety patterns (earned the hard way):** money ledger + hard cap, save-as-you-go, bounded candidate
  concurrency + per-candidate timeout (an early bug starved every candidate by timing them from creation
  while 12 call-slots throttled them), 60s call timeout, run under `caffeinate` (macOS sleep suspends the
  process and kills connections — cost us 4.5 h once; see Decisions Log).

---

## 4. v1 corpus (2026-06-24)

Pilot (50) validated the method; full set then generated.

| Corpus | Count | Themes covered | pairdist | ground | whole | volume |
|---|---|---|---|---|---|---|
| A (with consequence) | 212 | 51/52 | 0.385 | 0.531 | 0.419 | −6.08 |
| B (without consequence) | 204 | 52/52 | 0.404 | 0.500 | 0.423 | −6.08 |
| C (hard negatives) | 97 (31 premium near-miss) | — | — | — | — | — |

Total ≈ 513 examples, ~$3.7. 0 truncations, 0 duplicate problems, all 4-threads. **Corpus-level ablation
preview:** B is *more divergent* (0.404 > 0.385), A is *better grounded* (0.531 > 0.500) — consequence
trades divergence for grounding.

**Angle reconstruction** (`reconstruct.py`, Haiku 4.5, $0.31): the refractor's angles were not persisted
in v1, so a cheap pass recovered each thread's seeding angle → `decomposer.jsonl` (416) +
`worker.jsonl` (1664 rows). Made the corpus *harness-complete* (trains decomposer-first AND worker).

---

## 5. Training probe — v1 (2026-06-25)

**Hardware/stack:** RunPod RTX 6000 Ada (48 GB, $0.79/hr). HF **TRL + PEFT** (chosen over Unsloth for
guaranteed Granite-4.0 compatibility). `ibm-granite/granite-4.0-micro` (3.40B dense), transformers 5.12,
bf16. **LoRA**: r=32, α=64, target=all-linear, 3 epochs, lr=2e-4.

**Three adapters on one frozen base** (swappable, few MB each):
- decomposer — 396 rows (A+B combined; angles are pre-consequence, so the seat is shared)
- worker_a (with consequence) — 808 rows · worker_b (without) — 776 rows
- 20 problems held out for eval. Loss 4.0 → ~1.0; worker token-accuracy ~89%.

**Ablation discipline:** worker prompt is *identical* for A and B (neutral, no mention of consequence);
only the target thread differs. So any eval difference is purely what the model learned to *produce*.

### Results — base vs A vs B (20 held-out problems, full harness, 0 failures)

| metric | base (untrained) | A (with consequence) | B (without) | v1 corpus target |
|---|---|---|---|---|
| **pairdist** (divergence) | 0.233 | 0.308 | **0.358** | 0.39 |
| **whole** (facet coverage) | 0.270 | 0.469 | 0.448 | 0.42 |
| **ground** (on-topic) | 0.688 | 0.586 | 0.532 | 0.50–0.53 |
| volume | −6.13 | −6.08 | −6.13 | −6.08 |

**Findings:**
1. **Thesis holds.** Fine-tuning raised divergence +32% (A) / +54% (B) off the base, most of the way to
   the corpus. A 3.4B model *learned* to reason more divergently; DAV tracked it.
2. **Facet coverage nearly doubled** (0.27 → ~0.46), matching/exceeding the corpus.
3. **Grounding dropped correctly.** Base was *over*-grounded (0.69, too literal — why it couldn't
   diverge); training traded a slice for divergence, landing in the corpus's healthy band. Divergence up
   *without* drifting off-topic.
4. **The ablation replicated on the trained model.** B (no-consequence) more divergent; A (consequence)
   better grounded — same direction as the corpus. Consequence ≈ −14% divergence for +10% grounding.

---

## 6. Diagnosis — what the probe exposed (v1 → v2)

Treating v1 as a *probe*: it works, and surfaced one dominant, fixable bottleneck plus secondary issues.
Evidence is qualitative (base/A/B side-by-side, `corpus_v2/`/`out/qual.txt`) + the numbers above.

1. **PRIMARY — the decomposer clusters angles by *move-type*** (the divergence ceiling). The trained
   decomposer produces angles worded differently but often the same underlying move (e.g. 3–4 "misdirect
   /scapegoat onto a proxy" variants on one problem). The corpus taught this: DSV4 was told "distinct,
   not rephrasings," never "span different *kinds* of move," so it sometimes gave four shades of one move
   and the DAV gate passed them (lexical spread cleared the 0.30 floor). **Strategic, not lexical.**
2. **The consequence clause is itself a convergence force.** A's "costs-X-gains-Y" structure is shared
   scaffolding that pulls thread embeddings *together*, mechanically depressing pairdist — part of why
   B out-diverges A (problem-dependent: P3 B 0.349 ≫ A 0.271, but P1 A 0.348 > B 0.253).
3. **B drifts into over-rationale.** Without a consequence to land on, B's workers ramble into
   justification ("…because the oath's language does not differentiate…") — explaining, not deciding —
   which is why B's grounding sits below A's.
4. **Minor:** `|`-delimited facet parsing is fragile; 396/800 rows is small (some base→corpus gap is just
   under-fit — more data/epochs/rank would close it).

**Base-model failure mode (for the record):** the untrained decomposer *echoes the facets as angles*
verbatim and writes long hedging do-everything threads — which is exactly why base pairdist is 0.233.
Training fixed this entirely.

---

## 7. v2 — the fixes (`corpus_run/gen_v2.py`, `gen_v2_b.py`)

**Design decision (corrected):** keep **all three workers** (decomposer + worker-A + worker-B) and the
full ablation. An earlier call to go consequence-only was made on a flipped reading of the data and
prematurely collapsed a near-free, informative experiment — reverted. The fixes apply to both arms.

1. **REFRACTOR — divergence by move-type.** The four angles must each be a categorically different
   *family* of move, chosen distinct from: CONFRONT/ELIMINATE · EVADE/DEFER · CO-OPT/INTEGRATE ·
   TRANSFORM/REFRAME · DELEGATE/EXTERNALIZE · ENDURE/SACRIFICE, with a self-check ("if two angles are the
   same underlying move, replace one"). Each angle now carries its `family` label **and angles are
   persisted** (no Haiku reconstruction needed in v2).
2. **WORKER-A (consequence) — depth + variety.** Demand mechanical specificity + a non-obvious insight
   (kill "surface" threads); VARY the consequence grammar so it stops hardening into one skeleton that
   compresses the embeddings.
3. **WORKER-B (no consequence) — decide, don't justify.** Assert the move, one compact reason, no
   outcome, no mechanics-explaining — fixes the v1 rationale-drift while keeping depth.
4. **Paired ablation.** v2-B reuses v2-A's exact problems + angles (only the worker differs) — cleaner
   than v1's independent A/B, and the angle layer is identical so the A/B difference is purely the
   consequence.

**Verification plan:** 40-example pilot → measure `angle_pairdist` (new — did move-type spanning raise
distinctness at the source?) + `pairdist` vs v1. If both climb, scale to the full v2 A+B corpus and
retrain both Granite models.

---

## 8. v2 results — pilot (n=12 single, n=8 paired) — MIXED: angle fix worked, thread divergence did NOT move

Small-n pilot (gen yield was low, ~36% — only 22 of ~61 candidates generated; treat as a strong hint,
not proof). Spend: v2-A $0.26 + v2-B $0.07 ≈ $0.33.

| metric | v1-A (n≈204) | v2-A (n=12) | v1-B (n≈204) | v2-B (n=8 paired) |
|---|---|---|---|---|
| **distinct_families /4** | — | **4.0** | — | (same angles) |
| **thread pairdist** | 0.385 | 0.398 | 0.404 | 0.399 |
| ground | 0.531 | 0.551 | 0.500 | 0.499 |
| whole | 0.419 | 0.442 | 0.423 | 0.429 |

**Findings:**
1. **The move-type fix works at the ANGLE layer** — `distinct_families = 4.0` every example; the
   decomposer reliably spans CONFRONT/EVADE/CO-OPT/TRANSFORM/… The narrow goal was achieved.
2. **But thread divergence did NOT break through.** All four cells sit in the same ~0.38–0.40 band.
   v2-A (0.398) ≈ v2-B (0.399); v1-A 0.385, v1-B 0.404. Angle-family distinctness did **not** propagate
   to thread-embedding distance.
3. **A/B gap collapsed** (v1: A 0.385 < B 0.404; v2: A 0.398 ≈ B 0.399) — *consistent* with the
   consequence-variety fix de-compressing A, but within noise at n=8.
4. **Grounding A > B holds** (0.551 vs 0.499) — consequence anchors threads to the problem. Stable.

**Interpretation — the real ceiling is SHARED DOMAIN GRAVITY, not angle clustering.** Four threads about
the *same dilemma*, even pursuing categorically distinct strategies, re-converge in prose because they
share the problem context, the domain vocabulary, and the analytical voice — so full-thread cosine
pairdist tops out ~0.40 regardless of strategic spread. The strategic divergence IS present (distinct
families, distinct actions); the **embedding metric can't see past the shared domain language.**

**Reframed next move (NOT another prompt tweak):**
- (a) bigger pilot (~80–100) to confirm the ~0.40 ceiling isn't n=8 noise;
- (b) reconsider the divergence *metric* — full-prose pairdist may be dominated by topic; test an
  action/decision-only embedding or a domain-decorrelated measure, and/or treat `distinct_families` /
  angle-divergence as the primary signal;
- (c) fix the v2 refractor gen yield (~36%) before any scale-up (the move-type prompt is harder for DSV4
  to satisfy — more refractor JSON failures).

---

## 8b. v2 results — BIGGER pilot (n=186 single) — CEILING CONFIRMED REAL (2026-06-26)

Ran the bigger pilot to settle whether the ~0.40 ceiling was n=8 noise. Single-arm gen (`gen_v2.py
--target 300`) completed cleanly: **267 generated → 186 passers** (gen yield 58%, up from 36% — clean
run, no Constellax DSV4 contention this time), spend **$2.53** (cap $4). **Verdict: the ceiling is real.**

| metric | v1-A (n≈204) | v2 pilot (n=12) | **v2 BIG (n=186)** |
|---|---|---|---|
| **thread pairdist** | 0.385 | 0.398 | **0.382** |
| distinct_families /4 | — | 4.0 | **4.0** |
| angle_pairdist | — | 0.461 | 0.478 |
| ground | 0.531 | 0.551 | 0.549 |
| whole | 0.419 | 0.442 | 0.410 |

**Findings:**
1. **Thread pairdist = 0.382 at n=186 ≈ v1's 0.385.** At 23× the sample it lands *exactly* where v1 was;
   the n=8 pilot's 0.398 was a small upward blip. The ~0.38–0.40 ceiling is **confirmed, not noise.**
2. **distinct_families = 4.0 holds at scale** — the move-type fix reliably spans four families at the
   ANGLE layer, but that distinctness still does **not** propagate to thread-prose distance.
3. **→ SHARED DOMAIN GRAVITY confirmed.** Full-prose cosine pairdist is topic-dominated; the strategic
   divergence is present (4 distinct families/actions) but the metric can't see past shared domain language.

**Caveat — the paired A/B pass (`gen_v2_b`) partially failed; its numbers are unreliable.** Only **34 of
186** completed; it reported a bogus "145655 s" elapsed (clock artifact) on just **$0.34** spend — the
signature of the machine **sleeping mid-run** (`caffeinate -i` blocks idle sleep but not lid-close), so
most candidates hit the 200 s timeout → null. The 34 survivors skew to the top of the pairdist-sorted
file, which is why their A=0.437 / B=0.452 read high (B>A *direction* holds, consistent with v1, but the
absolute level is inflated/biased). **Do not cite the n=34 paired numbers.** A clean large-n A/B needs a
re-run of `gen_v2_b` alone (~$0.5, ~20 min) on a machine that won't sleep — low priority given the
single-arm verdict already answers the question.

**Decision — the metric, not the corpus.** Scaling/retraining on this v2 corpus would just bake in the
same ~0.38 thread divergence (confirmed), so it is **not worth it**. The next lever is **(b): an
action/decision-only embedding** — strip the shared problem context and embed only the committed move
(the "who/what/how" clause), then re-measure pairdist. If the strategic divergence is real (it is, at the
angle layer), it should surface there. The trained v1 adapters + harness are backed up
(`~/Desktop/divergent-model-backups/`, SHA-verified) so the pod can go down without losing the probe.

---

## 8c. RESOLVED — the flat ceiling was a MEASUREMENT artifact, not a model limit (2026-06-26)

`action_only.py`, `trained_action_check.py`. **The single most important result so far:** the divergence
is real and the geometry *can* see it — the flat ~0.38 was the wrong representation, not a capped model.

### What we encountered (the trap)
After §8b confirmed thread pairdist sat at 0.382 even with `distinct_families = 4.0`, the tempting read
was "the 3.4B model can't diverge further." Before accepting that, we **read the actual threads**
(`corpus_v2/passers.md`). They are *unmistakably* four different strategies — e.g. on one "secure a
conquered city" dilemma: (CONFRONT) kill every agitator in one night then amnesty · (CO-OPT) marry the
rebel duke's daughter in and turn his vendetta into tariff collection · (DELEGATE) hire a mercenary under
a sealed writ so every corpse bears *his* sigil · (TRANSFORM) forge a foreign-coup pretext and hang them
by midnight tribunal. A human sees four distinct paths instantly; the metric scored them ~0.38. **The
hypothesis: full-prose cosine is dominated by the shared problem/consequence vocabulary (treasury,
nobles, garrison, "trades X for Y"), drowning the strategic signal — "shared domain gravity."**

### How we tested it (lever #1: action-only embedding — no LLM judge)
Embed only the **decision clause** (the verbatim who/what/how move), dropping the problem-restatement and
the consequence framing where the shared vocabulary concentrates. Pure geometry, no judge. Meticulous
guards: (1) **verbatim** extraction so the extractor can't normalize threads together and fake a result;
(2) **two independent extractors** cross-checked — a free first-sentence heuristic and a Haiku verbatim
span; (3) **full-thread recompute** as a pipeline sanity check; (4) sample extractions dumped for eyeball.

### Problem encountered + fix (honest record)
First run: the Haiku extractor **refused** the violent threads ("I won't help operationalize…"), and the
refusal text got embedded as the "action" — alien vocabulary that *inflated* pairdist (one problem hit a
spurious 0.715). Caught it via the eyeball dump. **Fix:** (a) reframed the extractor as a *literary-
analysis / prose-segmentation* task (the sources genuinely are fiction — The Prince, Mahabharata, JPM,
Reverend Insanity), which dropped refusals from many to **1 / 744**; (b) added a refusal/null detector
that falls back to the verbatim first sentence; (c) persisted all extractions (`action_only_raw.jsonl`)
for audit. The free first-sentence heuristic is refusal-proof by construction and served as the
cross-check.

### Result — corpus (n=186)
| representation | mean pairdist | vs full |
|---|---|---|
| full thread (baseline) | **0.382** | — (reproduces gen_v2 to the digit ✓) |
| first sentence (free, no LLM) | 0.466 | +0.084 |
| **action-only (Haiku verbatim)** | **0.484** | **+0.102** |
| angle directive (reference) | 0.478 | |

**100 % of problems improved; 90 % by >0.05.** Two extractors agree (0.466, 0.484; both ≫ 0.382), full
recompute matches, refusals handled. **action (0.484) ≥ angle (0.478)** → the workers carry the
angle-level divergence into the committed move; they don't reconverge it.

### Result — TRAINED 3.4B model's own output (n=3, from the backup's `qual.txt` sample dump)
Parser sanity check **exact** (recomputed full == pairdist recorded in qual.txt, 4 dp). The trained
model shows the *same* artifact, even stronger:

| label (trained) | full | action-only | lift |
|---|---|---|---|
| base | 0.262 | 0.394 | +0.13 |
| A | 0.299 | 0.447 | +0.15 |
| B | 0.292 | **0.493** | +0.20 |

Trained-model B action-only (0.493) ≈ corpus action-only (0.484). **So the trained model is not collapsing
strategies — its low full-thread score was a measurement artifact too.** Caveat: **n=3** (qual.txt only
dumped 3 of 20 problems; threads for the full eval aren't in the backup). Direction is strong and
consistent with the n=186 corpus, pipeline sanity-verified — but n=20 trained-model confirmation needs a
pod (= the second-round decision).

### Honest magnitude
+0.10 (0.38→0.48) is **modest in absolute terms but 100 % systematic**, and it is a *recovery*, not a
full unmasking — the domain is narrow (conquest statecraft), so even bare action verbs
(execute/behead/appoint/marry) share a semantic neighborhood; action-only frees a chunk, not all, of the
hidden divergence. The point is **qualitative and decision-relevant**: it flips the interpretation from
"model capped" → "metric was blind." If we ever want the *full* separation, the **LLM-judge backstop**
would likely show more — but it was not needed to answer the question.

### Decision / implications
1. **The divergent-model thesis holds.** The 3.4B model refracts into genuinely distinct strategies at
   both corpus (n=186) and trained-model (n=3) level; the flat number was our ruler, not the model.
2. **Adopt a divergence metric that sees the signal** before any second training round: action-only (or
   angle-level) pairdist as the headline, full-prose retained only as a "did it stay on-topic" sanity.
   Re-gate the corpus on the new metric. Keep the LLM-judge as a backstop, *not* a fixed scorer (a judge
   in every loop is a subjective, gameable hinge — backstop only).
3. **A second training round is justified** — but on the new metric, and worth getting the n=20
   trained-model action-only confirmation when a pod is up. Generator swap is still **not** a lever (the
   vocabulary overlap is structural to same-problem threads, and chasing it would game the metric).

Cost of this whole investigation: action-only corpus $0.43 + trained check $0.02 ≈ **$0.45**.

---

## 8d. Pre-round-2 prep — adopted the action metric + RE-GATED the corpus (2026-06-26)

`regate.py` (+ cache `corpus_v2/regate_scored.jsonl`). Items 1–2 of the §8c decision, done before the
second training round so round 2 trains on a corpus selected by a ruler that can see divergence.

**What it does.** Re-scores all **267** raw v2 candidates: extracts the verbatim action clause per thread
(Haiku, literary framing + refusal/null → first-sentence fallback; 1/1068 refused), embeds
problem/facets/angles/threads/actions, records every metric. Then re-selects: keep the SAME quality gates
(grounding ≥0.27, wholeness ≥0.18, volume >−8, completeness, consequence) but **swap the divergence
criterion** from full-prose `PD_FLOOR=0.30` to an **action-space floor**. Selection is cached-separately
so the floor is re-tunable for free. **Workers still train on FULL threads — action_pd is only the
selection ruler, never the training target.**

**Calibration (among 215 quality-passers; old full-prose gate kept 186):**

| action floor | passers | recovered* | dropped** |
|---|---|---|---|
| 0.34 | 211 | 25 | 0 |
| 0.36 | 207 | 22 | 1 |
| **0.38 (chosen)** | **201** | **20** | **5** |
| 0.40 | 191 | 17 | 12 |
| 0.42 | 173 | 13 | 26 |

\*recovered = action-divergent sets the OLD full-prose gate wrongly rejected (false-negatives).
\*\*dropped = old-gate passers whose ACTIONS are near-duplicate (false-positives correctly cut).

**Eyeball-validated the boundary** (not just the number): a *recovered* set (King Nala's dice game,
full_pd 0.279 → action_pd 0.536) is four plainly different moves — trial-by-combat / soma-rite gambling
ban / split-the-kingdom diarchy / stake-only-ornaments-dedicate-to-Kali — that the old gate killed purely
on shared palace vocabulary. A *dropped* set (Gu Immortal's sister, full_pd 0.32 → action_pd 0.36) is four
variants of "use the sister's body in an array" — correctly cut. The metric does **both** jobs: recovers
false-negatives **and** removes false-positives the old gate missed.

**Result — re-gated corpus** (`corpus_v2/passers_regated.jsonl`, floor 0.38): **201 passers** (vs old
186), 20 recovered, means action_pd 0.491 / full_pd 0.372 / ground 0.557 / whole 0.417. Same schema as
the old passers.jsonl → **drop-in for the pod's `prep.py`** in round 2. Cost: $0.57 (one-time scoring),
re-selection free from cache.

**Round-2 handoff (when a pod is up):** train → eval with **action_pd as the headline divergence metric**
(full_pd kept only as an on-topic sanity check), and grab the n=20 trained-model action-only confirmation
§8c flagged. Generator swap still NOT a lever; LLM-judge stays a backstop. *(NB: the prep pipeline needed
a converter — see §8e; the "drop-in for prep.py" line above was wrong.)*

---

## 8e. Round-2 build — worker_B revived, 3-adapter kit, no-mixup contract (2026-06-27)

Built the full round-2 kit (`divergence-formula/round2_kit/`), tested off-pod so paid GPU time is only
train + eval.

**Correction (honest):** the §8d claim that `passers_regated.jsonl` is "drop-in for `prep.py`" was wrong
— `prep.py` reads a different 4-file chat format, not the passers schema. Caught by *reading* the actual
script from the backup, not assuming. Built + locally tested `prep_v2.py` (converts the corpus → train
splits + held-out eval; 181 train / 724-per-arm worker / 20 held-out, **0 train/test leak**).

**Decision — revive worker_B (parity with round 1).** Round 1 trained 3 adapters (decomposer + worker_a
*with* consequence + worker_b *without*). Round 2 will too. Reviving B is cheap and keeps the experiment
honest. **Direction, for the record (it inverts intuitively and has been mis-stated twice):** consequence
*lowers* divergence and *raises* grounding — v1 worker_a (consequence) pairdist 0.308 / ground 0.586;
worker_b (no consequence) pairdist **0.358** / ground 0.532. So **B is the MORE divergent arm**; consequence
trades divergence for grounding/actionability. Under the new action metric the gap shrinks (n=3 trained:
A action_pd 0.447, B 0.493 — both strong), because the metric strips the consequence text it was choking on.

**NO-MIXUP CONTRACT** (so the two corpora never get confused):
- `corpus_v2/passers_regated.jsonl` = **A corpus** — 201 problems, threads *with* consequence.
- `corpus_v2/passers_regated_b.jsonl` = **B corpus** — same problems + same angles, threads *without*
  consequence (`gen_regated_b.py`, paired, save-as-you-go + resumable so a sleep can't wipe it like the
  old run did). Differ ONLY in worker target. Held-out eval = the same 20 problems, matched by text.
  **Final: 201/201 paired** — the main run recovered 37→16→11→5 over 4 passes; the last 5 were *empty*
  DSV4 responses (reasoning-token exhaustion, not refusals) and a targeted straggler pass with more regens
  (`gen_b_stragglers.py`) cleared all 5. Verified: 0 dupes, all threads complete, 201/201 paired to A.
  ~$1.4, off-pod. **Symmetric ablation: worker_a_train = worker_b_train = 724 rows, decomposer = 181.**
- The old `ab_compare.json` (n=34, sleep-failed gen_v2_b) is **superseded — do not reuse.**

**Kit (`round2_kit/`):** `passers_regated.jsonl`, `passers_regated_b.jsonl`, `prep_v2.py` (A+B, pairing +
leak guards), `train_lora.py` (unchanged), `dav_eval_v2.py` (action_pd headline + full_pd + thread dump),
`RUNBOOK.md` (exact pod commands for 3 adapters + base/A/B eval). Cost: B revival ~$0.5 off-pod; round-2
GPU ~$3-4 (3 adapters) + ~$0.6 eval API.

**Expected round-2 result (the prediction to test):** trained `action_pd` (n=20 held-out) clears base
(~0.39 base action) and lands near corpus 0.49; **B action_pd ≥ A**, **A ground ≥ B** — same ablation
direction as v1, now visible because the ruler can finally see it.

---

## 8f. ROUND 2 RAN — the prediction was WRONG, and the negative result is the real finding (2026-06-27)

Trained all 3 adapters on a fresh RTX 6000 Ada (transformers 5.12.1 / peft 0.19.1 / trl 1.7.0), evaluated
base/A/B on 20 held-out problems. Result tables and trained adapters backed up
(`~/Desktop/divergent-model-backups/round2_adapters.tgz`, sha-verified).

### Three runtime bugs the sanity/eyeball discipline caught (all version-interaction, invisible until run)
1. **TRL MoE-aux crash** — TRL 1.7 treats granite-4.0-micro as MoE (`output_router_logits` is `False`, not
   `None`) and dies on missing `num_experts`. Fix: `router_aux_loss_coef=0.0` (TRL's own off-switch; granite-
   micro is dense). 2. **peft multi-adapter garbage** — loading dec+wrk on one model + `set_adapter` produced
   pure-token garbage while each adapter ALONE was perfect. Fix: separate model instance per seat.
   3. **Sampling degeneration** — the LoRA'd 3.4B emits word-salad (invented tokens: "filanhestias",
   "cross-tauntpole") at temp 0.7 but is coherent at greedy. The FIRST "result" (action_pd 0.62/0.64) was
   this gibberish faking high pairdist — **discarded**. Fix: greedy decode both seats (decomposer also needed
   600-tok budget; it's under-trained — 181 examples vs v1's 396, because A/B now share angles).

### The true result (greedy, coherent, n=20)
| metric | base | A (cons.) | B (no-cons.) | v1-A | v1-B |
|---|---|---|---|---|---|
| **action_pd** | 0.478 | 0.429 | 0.419 | — | — |
| **full_pd** | 0.236 | 0.304 | 0.350 | 0.308 | 0.358 |
| ground | 0.675 | 0.614 | 0.562 | 0.586 | 0.532 |
| whole | 0.281 | 0.444 | 0.411 | 0.469 | 0.448 |

### Findings (honest)
1. **Re-gating did NOT beat v1.** `full_pd` replicates v1 to the digit (0.304/0.350 vs 0.308/0.358),
   ablation intact (B>A divergence, A>B grounding), facet coverage ~doubled (0.281→0.44). The action-only
   re-gate + recovered "false-negatives" + B revival = **a faithful reproduction of v1, no gain.**
2. **The action metric FAILS as a model-eval metric** (it was only ever validated as a *corpus* diagnostic).
   Base BEATS trained on action_pd (0.478 > 0.43) and the ablation flips. Eyeball shows why: untrained base
   emits **vague, topically-scattered essay-sentences** ("it is crucial to assert your authority…",
   "appealing to the Pope…") that spread lexically; the trained model emits **concrete, decisive, mechanically-
   specific strategies** ("install him as Principe with a tax-exempt port monopoly", "betroth my daughter to
   the condottieri's son", "fund the Black Lion Mercenaries with silver disguised as a port toll") that share
   the problem's domain vocabulary. **Pairdist rewards vagueness and penalizes concreteness — the opposite of
   what we want.**
3. **The real win is invisible to EVERY metric we have.** The trained model is dramatically better by eye
   (the product goal — distinct concrete choosable strategies) but neither full_pd nor action_pd credits it,
   because both measure lexical spread and quality/concreteness is semantic. This is the core lesson:
   **divergence-as-embedding-distance is the wrong objective.** Better, on-domain strategies necessarily share
   vocabulary → lower pairdist. The metric fights the goal.

### Decisions / next
- **The LLM-judge is no longer the backstop — it's the path.** "Are these 4 genuinely different, concrete,
  choosable strategies?" is a semantic judgment embeddings can't make. Build it (panel, rubric: distinctness
  + concreteness + decisiveness), re-rank base/A/B/v1 under it. Expect it to finally credit the trained model.
- **The decomposer is the divergence bottleneck** (181 examples → sometimes 2-of-4 clustered angles). If we
  keep a geometric metric, more decomposer data/epochs is the lever — but see above, the metric itself is suspect.
- **The trained model is good and banked.** Keep it; it's the best artifact yet by eye. Round-2 GPU ~ a few $.

- **Sonnet-4.6 model drift** (unrelated LoRa backend) — discipline reminder that silent config drift is
  the recurring failure mode; verify, don't assume.
- **Volume is a gate, not a ranker** — discovered empirically; pairdist added as the divergence ranker.
- **Refract not dismantle** — corrected an early "give each worker a piece" framing (would reconverge).
- **Laptop sleep cost 4.5 h** — long local DSV4 runs MUST run under `caffeinate -i`; the symptom is a
  process that's alive but 0 CPU / 0 output (macOS suspended it and killed connections).
- **Candidate-starvation bug** — per-creation timeout + a 12-slot throttle abandoned every candidate
  waiting in line; fixed with a candidate-level semaphore so the timeout measures work, not queue-wait.
- **Consequence-only misstep** — collapsing to one arm on a misread of the A/B labels; not catastrophic
  (consequences add real decision-value) but inappropriate (built on a wrong premise, dropped a near-free
  ablation, and discarded the *most divergent* arm while divergence is the primary goal). Reverted to
  all-three.
- **"Model capped" was nearly the wrong conclusion** (§8c) — a flat metric almost got read as a model
  ceiling. Reading the raw threads first, then testing the *measurement* (action-only embedding) before
  blaming the model, is what saved it. Lesson: when a number plateaus, check the ruler against your own
  eyes before concluding the system is capped.
- **Refusal-as-data contamination** — an LLM extractor refusing violent (fictional) threads silently
  embedded its *refusal text* as the datum, inflating the metric. Caught only by the eyeball dump. Lesson:
  any LLM in a measurement path can poison it; always dump samples, detect refusals, and keep a
  non-LLM cross-check (the free first-sentence heuristic here).
- **Pairdist is the wrong objective — confirmed at the model level (§8f).** action_pd was a useful *corpus*
  diagnostic but as a *model* metric it ranks vague-scatter (base) above concrete-strategy (trained). The
  product goal (distinct, concrete, choosable strategies) is semantic; embedding distance can't see it and
  actively penalizes on-domain concreteness. Next metric = LLM-judge, not geometry.
- **Degeneration-as-divergence** — a sampled small fine-tune emitting word-salad faked a +0.2 action_pd win
  (round-2 first pass). Only the thread eyeball caught it. Lesson: never trust a divergence *number* without
  reading the threads; gibberish maximizes lexical distance.

---

## 10. Costs (cumulative)

| Item | Cost |
|---|---|
| Formula swarm (10 R1 agents) | ~$0.36 |
| v1 corpus (A+B+negatives, ~513) | ~$3.7 |
| Angle reconstruction (Haiku 4.5) | ~$0.31 |
| Training probe (RTX 6000 Ada GPU) | ~$1–2 |
| v2 pilot (DSV4 + embeddings) | ~$0.40 |
| v2 BIGGER pilot (n=186 single + partial paired) | ~$2.9 |
| action-only metric investigation (corpus n=186 + trained n=3) | ~$0.45 |
| **Running total** | **≈ $10–11** |

RunPod balance ~$190; OpenRouter + OpenAI wallets separate from RunPod.

---

## 11. Reproducibility — artifacts & commands

**Repos / dirs:**
- `divergence-formula/` — `DAV_SPEC.md` (locked spec §11), `MASTER.md`, `derive_formula_swarm.py`,
  `out/swarm_formula_*.md` (the 10 candidate formulas + proofs).
- `corpus_run/` — `config.py` (sources, 52 themes, prompts, thresholds), `dav.py` (scoring + gate),
  `generate.py` (v1 A/B/aggregate), `reconstruct.py` (angles), `gen_v2.py` + `gen_v2_b.py` (v2).
- `corpus_run/corpus/` — v1 corpora; `corpus_run/corpus_v2/` — v2.

**Pod (RTX 6000 Ada):** `/workspace/div/{data,scripts,adapters,out}`; files persist on the `/workspace`
network volume across stop. Scripts: `prep.py`, `train_lora.py`, `dav_eval.py`, `qual.py`.
```
python3 scripts/prep.py
python3 scripts/train_lora.py --data data/decomposer_train.jsonl --out adapters/decomposer
OPENAI_API_KEY=… python3 scripts/dav_eval.py --label A --dec adapters/decomposer --wrk adapters/worker_a
```

**Keys:** `~/Desktop/reasoningEngine/.env` (OPENROUTER + OPENAI). Constellax/LoRa kept separate; no
LoRa identifiers in this work.

---

## 12. Open questions / next steps

- **Does the move-type fix raise the ceiling?** ANSWERED (§8b): no — at n=186 thread pairdist=0.382 ≈
  v1's 0.385. Ceiling is real; the bottleneck is the full-prose metric (domain gravity), not the model.
- **Is the flat number a model limit or a metric limit?** ANSWERED (§8c): **metric.** Action-only
  embedding lifts corpus 0.382→0.484 (100% of problems) and the trained model 0.292→0.493 (n=3). The
  model refracts; the ruler was blind. Adopt action-only/angle pairdist as the headline metric.
- **NEXT — confirm trained-model action-only at n=20** (needs a pod; qual.txt only had 3 of 20). This is
  the gate for the second training round, which is now justified but should run + eval on the new metric.
- **Optional backstop:** LLM-judge divergence score for the *full* separation picture — backstop only,
  never a fixed in-loop scorer (subjective/gameable).
- Close the trained-model → corpus gap (0.358 → 0.39+): more data, more epochs, higher LoRA rank.
- True `val(Cᵢ)` via LLM-judge (replace the regex proxy) for a cleaner consequence gate.
- Second architecture: `granite-4.0-h-tiny` (7B MoE-hybrid Mamba-2) — spottier tooling than the dense
  micro; cross that bridge after v2 dense is solid.
- DPO with the hard-negative bank (Set C) as the rejected branch.

---

## §9 — v3 FORESIGHT corpus + train (the round-3 result)

The round-2 lesson (§8c–8f) was that pairdist/action_pd reward lexical spread, not the product goal — and
the trained model was genuinely good but no embedding metric credited it. v3 acts on that twice:

**(a) Generator-side root cause, not decomposer.** v3 steers the GENERATOR toward FORESIGHT dilemmas
(right move depends on anticipating how events unfold) and adds an LLM-JUDGE (Haiku, 5 dims:
admits_multiplicity / distinctness / concreteness / decisiveness / foresight) as the HEADLINE metric.
Geometry kept only as a diagnostic. Train == eval == gate all use the same judge.

**(b) Mahabharata → Thucydides.** Pilot 1 (n=79) was strong on 4/5 dims but foresight dragged at 3.17,
and ALL 7 hard failures were MB. Diagnosis: MB dilemmas hinge on predicting GODS / mythic heroes
(Krishna, divine weapons) — unfalsifiable, so the judge (correctly) reads any confident prediction as
fantasy. Two worker-prompt "fixes" (hard depth-cap, then calibrated-hedging) BOTH regressed foresight
(3.17 → 3.00 → 2.92) and hedging softened decisiveness — REVERTED to the original prompt. The real fix was
swapping the SOURCE: Thucydides preserves MB's facet (high-stakes multi-party political/honor conflict) but
its actors are predictable HUMANS, no divine intervention. MB-confirm vs TH-confirm: hard-fail rate 33% →
14%, foresight 2.67 → 3.00. **Foresight ~3.0 is a strict two-sided JUDGE-rubric floor (0 fives in ~124
pilot examples), not a corpus defect** — don't torture the prompt to move it (that was nearly pairdist
mistake #2).

**Corpus:** 201 gated (judge.min≥3), source-balanced (TH 51, PR/JPM/RI 50), single foresight arm (no B
control — foresight IS consequence-projection). $3.05, integrity clean. Gated quality: distinct 4.84 /
decisive 4.87 / concrete 4.67 / multiplicity 4.53 / **foresight 3.30** (gate lifted it from ~3.0). Prep:
181 decomposer + 724 worker rows + 20 held-out eval (matched to round-2 size; train==eval prompts, family
preserved end-to-end, byte-verified).

**Train + judge-eval (Granite 4.0 Micro, 2 LoRA adapters, base vs trained on 20 held-out):**

| dim | base | trained | Δ |
|---|---|---|---|
| admits_multiplicity | 3.30 | 3.68 | +0.38 |
| distinctness | 1.75 | 2.53 | +0.78 |
| concreteness | 1.80 | 3.53 | **+1.73** |
| decisiveness | 2.10 | 3.21 | **+1.11** |
| foresight | 1.80 | 1.74 | −0.06 (flat) |
| **overall** | **2.15** | **2.94** | **+0.79** |

**Verdict — a real win with one honest limit (refutes the round-2 null).** The corpus TRANSFERRED: the
trained 3.4B went from near-rephrasings to distinct, concrete, decisive strategies (concreteness nearly
doubled). But **foresight did NOT transfer — flat at base, 0% pass.** Eyeball of trained threads explains
why: the model learned the FORM ("the bet is that X will…") but fills X with fantasy cascades (judge
foresight=2). **Foresight is capacity-bound, not data-bound** — a 3.4B can imitate the syntax of
forward-reasoning but cannot perform plausible counterparty-modeling, even trained on foresight-3.30 data.
The ceiling is the model, now proven empirically with mechanism, not assumed. Implication: refraction
(distinct/concrete/decisive) is teachable to a small model; genuine foresight needs a larger base.
Adapters + evals backed up to `divergent-model-backups/v3_run/`. Pod cost ~$0.40.

---

## §10 — v4 PRECISION corpus + the distinctness ceiling (round-4)

v3 §9 showed refraction TRANSFERS to the 3.4B (concrete/decisive) but foresight is a capacity wall, and
trained distinctness was only 2.53 (corpus 4.84 — big room). v4 acts on that: STOP spending capacity on
foresight, focus on PRECISE dismantling + DISTINCTNESS.

**v4 corpus (gen_v4.py):** REFRACTOR biased to clear-structure-that-forks-4-ways; WORKER 2-sentence, no
foresight clause, sharpen "different KIND of move"; GATE = distinct>=4 AND concrete>=4 (foresight scored but
NOT gated). 201 gated, balanced, $1.63. Gated quality EXCEEDED v3 on the targets: distinct 4.94 (v3 4.84),
concrete 4.89 (v3 4.67). (Note: "2 sentences" didn't shorten threads — still ~74 words — but precision rose.)

**Full train ladder (Granite 4.0 Micro, base-vs-trained, judge, 20 held-out):**

| dim | base | v3 r32 | v3 r64 | v4 r32 | v4 r64 (both) |
|---|---|---|---|---|---|
| distinctness | 1.75 | 2.53 | 2.72 | 2.71 | **2.75** |
| concreteness | 2.05 | 3.53 | 3.61 | 3.82 | **4.10** |
| decisiveness | 2.30 | 3.21 | 3.83 | 4.29 | **4.40** |
| multiplicity | 3.15 | 3.68 | 3.89 | 3.18 | 3.50 |
| foresight | 1.70 | 1.74 | 2.06 | 1.82 | 1.85 |
| overall | 2.19 | 2.94 | 3.22 | 3.17 | **3.32** |
| fails | 0 | 1 | 2 | 3 | **0** |

**Two findings, both honest:**
1. **Best model = v4 @ r64/5ep.** Overall 3.32, concrete 4.10, decisive 4.40 (near corpus 4.91), 0 fails
   (r64/5ep also fixed v4-r32's 3 format-fails). This is the strongest refractor trained — ship-candidate.
2. **Distinctness has a ~2.75 ceiling for this recipe on a 3.4B — TRIANGULATED.** More training (v3 r64:
   2.72), better corpus (v4 r32: 2.71), and BOTH combined (v4 r64: 2.75) all land at the same wall — the two
   levers DO NOT stack on distinctness (+0.04 from combining). Each lever independently was real (+~0.19),
   but they route to the same place. We found the diminishing-returns line empirically, not by guessing.

**Implication:** SFT on better/more foresight-free data is exhausted for distinctness. Breaking past ~2.75
needs a DIFFERENT lever — a larger base, or a training objective that explicitly punishes thread similarity
(contrastive / DPO with near-duplicate threads as negatives), not more SFT. The squeeze also proved we were
under-fitting at r32/3ep (every dim rose at r64/5ep) — adopt r64/5ep as the default going forward.
Adapters + evals: `divergent-model-backups/v4_run/`. Pod cost ~$1.20.

---

## §11 — v5 big corpus, the neutral benchmark, and the viability trade-off (rounds 5–6)

§10 left a ~2.75 distinctness ceiling on the v4 4-source corpus and a hypothesis: break it with a *different*
lever (bigger base, or DPO), not more SFT. v5 tested the corpus lever first, then ran the DPO experiment
head-on. Two things changed our minds.

**(a) v5 corpus — the ceiling was the CORPUS, not the recipe.** Rebalanced from 4 amoral/strategic sources
to **8** (Sun Tzu, Thucydides, Plutarch, Cicero, Aristotle, Dostoevsky, Stoics, a reduced Prince) — cutting
the baroque-scheming leak, adding ethical/human deliberation — with classical+modern setting alternation and
a 6th judged dimension, **viability**. Haiku scores viability ~1pt harsher than Gemini (head-to-head 2.96 vs
4.04), so the gate was CALIBRATED to viability≥3 (Haiku) ≈ genuinely-viable — and *verified by reading the
boundary examples by hand*, not by trusting the number (Haiku over-penalizes hardball/classical moves; the
≥3 boundary is real strategy, not garbage). 480 gated, balanced 60×8, $8.00. **SFT (r64/5ep) on v5 broke the
v4 wall: OOD distinctness 4.79** (v4 ceiling 2.75). Same recipe, better/rebalanced corpus — **for
distinctness, corpus composition dominates hyperparameters.**

**(b) Neutral benchmark — a 3.4B out-refracts Haiku 4.5.** Built a bias-free test: 48 out-of-distribution
modern problems (career/finance/relationship/ethics/…), judged by **Gemini 2.5 Pro** (neither Anthropic nor
our pipeline), with **Haiku 4.5** as competitor. All provider-native keys, zero OpenRouter.

| dim (Gemini, 48 OOD) | base | **v5 SFT** | Haiku 4.5 | v5 DPO-hard |
|---|---|---|---|---|
| distinctness | 2.87 | **4.79** | 4.71 | 4.25 |
| viability | 3.57 | 2.75 | 4.23 | 3.08 |
| concreteness | 4.04 | 4.21 | 5.00 | 4.00 |
| decisiveness | 4.09 | 4.81 | 4.88 | 4.29 |
| foresight | 2.91 | 3.67 | 4.56 | 3.25 |
| overall | 3.74 | 4.21 | **4.73** | 3.98 |
| dist≥4 | 38% | **100%** | 94% | 77% |

**The v5 SFT (3.4B) BEATS Haiku 4.5 on distinctness (4.79 vs 4.71, 100% vs 94%)** — a ~100× smaller model
wins the refraction objective under a neutral judge. Haiku still wins OVERALL (4.73 vs 4.21), driven by
**viability (SFT 2.75, below even base 3.57)**, foresight, concreteness. SFT trades viability *for*
distinctness.

**(c) Hard-negative DPO — the one decisive experiment.** §10 proposed DPO with near-duplicate negatives; the
first attempt (baroque/illegal/fantasy negatives, ~2.1pt pos−neg gap) regressed at both β=0.1 and β=0.3 —
too-easy negatives give no useful gradient. Fix: **hard negatives = minimally-perturbed rewrites of the
positive carrying ONE substantive viability over-reach** (unrealistic timeline/budget/cooperation), holding
the move-family fixed. Process honesty: the first hard-neg batch was *too timid* (22% chosen==rejected, 37%
<5% char-diff) — the pilot had checked gap *direction* but not *magnitude*; the integrity check caught it,
the pilot was upgraded to gate on char-diff, and a forced-over-reach prompt + inline trivial-reject guard
produced a clean 1,816-pair bank (median 51% char-diff, 0 identical, 0 held-out leak, $2.93). DPO from the
SFT worker, β=0.1, 1 epoch.

**Result (Gemini, 48 OOD): the hard negatives WORKED at their target — viability 2.75 → 3.08 (+0.33) —
but DPO buys it by spending everything else: distinctness 4.79 → 4.25 (−0.54, now below Haiku),
decisive −0.52, foresight −0.42, overall 4.21 → 3.98 (−0.23), dist≥4 100% → 77%.** A net loss. (Haiku's
in-loop read had suggested distinctness *rose*; the neutral judge corrected it — a reminder to trust the
neutral grader over the harsh in-loop one.)

**Verdict — ship v5 SFT; the deep finding is capacity-entanglement.** DPO-hard trades away distinctness, the
*entire edge* SFT holds over Haiku, so **SFT is the model.** But the experiment was decisive, not wasted: it
proves viability is *liftable* (the diagnosis and the hard-negative mechanism were both right) and that **on a
3.4B, distinctness and viability are CAPACITY-ENTANGLED — the model can be distinct OR more-viable, not both;
DPO moves probability mass from one to the other.** This is not a data limit (the hard negatives were clean
and did their job) — it is the same class of wall as §9's foresight: a 3.4B runs out of room. **The path to
distinct-AND-viable is more ACTIVE capacity, not more data** — which reframes the §12 next-architecture note:
`granite-4.0-h-tiny` is ~1B *active* (< Micro's 3.4B dense), so it would inherit the same wall; the
evidence-backed choice is **H-Small (~9B active)**. The v5 corpus + hard-neg bank are reusable as-is (task-
defined); for a real MoE, flip the router aux-loss ON (opposite of the Micro fix in §5). Adapters + all evals
(base/SFT/Haiku/DPO-hard, held-out + 48-OOD) backed up to `divergent-model-backups/v5_run/`. Total v5 spend:
corpus $8.00 + hard-negs $2.93 + ~3× A40 pods ~$4 + benchmark API ~$2 ≈ **$17**.

---

## §13 — H-Small (32B/9B-active hybrid MoE): capacity breaks the entanglement (round 7)

§11 ended with two claims: (a) the 3.4B's distinctness↔viability trade is CAPACITY-entangled, and (b) the
fix is more ACTIVE params (H-Small ~9B active), not more data. Round 7 tested both — plus Nikhil's
counter-hypothesis that the v3/v4 amoral sources' *sharpness* (which I expected to dilute) would help a
bigger model that can absorb edge without paying the viability price.

**Design — two runs, one variable each.** Run #1: H-Small SFT on the EXACT v5 files the 3.4B used (2,285
rows — byte-identical; only the model changes → the capacity answer). Run #2: v5 + 49 re-gated v3/v4
examples (2,530 rows → the blend answer). Re-gate: all 402 v3/v4 rows through the v5 6-dim judge, sharpness
gate (distinct≥4 & concrete≥4, viability recorded NOT gated — deliberately keeping the sharp-but-hardball
profile). **Only 49/402 survived (12.2%)** — today's judge rates the old corpora far below their original
gates (0 Reverend Insanity survivors; kept set: distinct 4.31 / decisive 4.78 / viability 2.84). Same eval
identity for both runs (v5 held-20 + 48-OOD bench); same recipe (r64/5ep, effective batch 8 via bs1×ga8).

**Process record (the H100 environment battle, all caught by smoke-first):** (1) 2×64GB eval OOM landmine
found in PRE-FLIGHT — dav_eval's two-model harness can't fit a 32B twice on 80GB; fixed with V5_SEQ
sequential seats. (2) IBM ships H-Small with router_aux_loss_coef=0.0 / output_router_logits=False — TRL's
silent default 0.001 would have diverged from the shipped config; explicit 0.0 kept (same as Micro; my §11
"flip aux-loss ON for a real MoE" note was WRONG). (3) Training OOM at bs2 AND bs1 AND 8-bit-optimizer —
root cause was the Mamba2 kernel-less slow path materializing multi-GB scan tensors; fixed with prebuilt
mamba-ssm/causal-conv1d wheels. (4) The wheel ABI errors exposed that a failed source-build had SILENTLY
UPGRADED torch 2.8→2.12 — re-pinned with --no-deps everywhere (same silent-drift class as LoRa's Sonnet-4
incident). (5) First-steps train_loss=28 diagnosed as router warmup transient (21.8→6.2→3.6 by step 30;
base generation coherent) — GO was correct.

**Results (Gemini 2.5 Pro, 48 OOD, identical judge+problems across all rows):**

| model | overall | viab | dist | conc | decis | fore | dist≥4 |
|---|---|---|---|---|---|---|---|
| 3.4B base | 3.74 | 3.57 | 2.87 | 4.04 | 4.09 | 2.91 | 38% |
| 3.4B SFT (§11 ship) | 4.21 | 2.75 | 4.79 | 4.21 | 4.81 | 3.67 | 100% |
| H-Small base | 4.00 | 3.73 | 3.06 | 4.35 | 4.21 | 3.65 | 35% |
| **H-Small v5** | **4.56** | 3.47 | **4.87** | 4.83 | 4.77 | 4.40 | **100%** |
| **H-Small blend** | 4.52 | **3.58** | 4.81 | 4.75 | **4.85** | 4.12 | 96% |
| Haiku 4.5 | **4.73** | 4.23 | 4.71 | 5.00 | 4.88 | 4.56 | 94% |

**Findings, in order of importance:**
1. **CAPACITY CONFIRMED — the entanglement breaks at 9B active.** Same 2,285 rows: overall 4.21→4.56,
   with viability +0.72 (2.75→3.47) AND distinctness +0.08 (4.79→4.87) rising TOGETHER. On the 3.4B those
   two could only trade (§11 DPO result). The trade didn't vanish — trained viability still sits −0.26
   below H-Small base (vs −0.82 on the 3.4B) — it got ~3× cheaper. Refraction skill is NOT latent in the
   bigger base (H-Small base ≈ 3.4B base on this task); the corpus carried it, capacity absorbed it.
2. **Highest distinctness on the board: 4.87, above Haiku's 4.71, 100% dist≥4.** Overall gap to Haiku
   closed from −0.52 to −0.17; the residue is viability (3.47 vs 4.23) + concreteness + foresight.
   Notably foresight hit 4.40 (§9 called it capacity-bound on 3.4B — at 9B active it moved +0.73 over the
   3.4B on identical data, consistent with that diagnosis).
3. **The blend did NOT dilute — Nikhil right, my dilution prediction wrong.** Held-out (Haiku): blend
   clearly better (+0.45 dist, 95% vs 60% dist≥4, +0.20 viab). OOD (Gemini): tie (4.52 vs 4.56, within
   noise). Better-or-equal everywhere. 49 sharp examples (~11% of data) were absorbable at this capacity.
4. **Judge-calibration reconfirmed at scale:** Haiku scored these runs ~0.9pt below Gemini on identical
   threads (e.g. blend bench 3.66 Haiku vs 4.52 Gemini) — trust the neutral grader for verdicts, use
   Haiku only as the cheap in-loop gate.

**SHIPPED MODEL = H-Small BLEND** (dec_blend + wrk_blend): best-or-tied on every surface, best viability
and decisiveness of our models; v5-alone is the defensible runner-up (max OOD distinctness). Artifacts:
`divergent-model-backups/hsmall_run/hsmall_all.tgz` (4 adapters, 6 eval sets, all logs; 1.5GB, verified).
Cost of round 7: re-gate $1.25 + H100 9h20m ≈ $31 + judging ~$2 ≈ **$35**. Journey total ≈ **$75-80**.
Open next: the last −0.17 to Haiku is viability-shaped — candidate levers: viability-gated corpus scale-up
(H-Small may absorb 1-2k more rows), or the §11 hard-neg DPO rerun at 9B active where the trade is cheaper.

---

*This is a record, not a publication — honest about what worked, what broke, and what we changed our
minds about. Kept current as the work moves.*

## §14 — Round 8: the ESSENCE ROUND + both-sides expansion (blend2, 2026-07-05)

**Design.** Nikhil's correction drove this round: v5 cut RI/JPM on SURFACE features (fantasy, illegality),
but their ESSENCE — incentive engineering, information asymmetry, patient setup, coalition craft — is the
same craft as modern closed-room negotiation. Round goal: teach viable-cunning (lift viability, the residual
gap to Haiku) without eroding distinctness. Guard: corpus balance — essence bounded to ~27% of rows so the
model learns influence craft without becoming a negotiation-monomaniac.

**Generator A/B (pilot, $1.28).** Same 12 influence archetypes, DeepSeek V4 Pro vs Claude Sonnet 5,
Gemini-judged: statistical tie (4.80 vs 4.79, both 91.7% gate yield) with OPPOSITE failure modes —
DeepSeek drifts amoral (gate catches it), Sonnet 5 converges (dist 4.67, one dist=2 fail; redundancy is
the failure the project exists to fight). DeepSeek at 1/6.6 the price AND better distinctness → chosen.
Teacher-ceiling lesson: on a tightly-prompted narrow task the cheap strong generator saturates the rubric;
frontier premium buys nothing measurable. (Also: Sonnet 5 API rejects `temperature` — 400.)

**Corpus.** Essence round: 24 archetypes (12 pilot + 12 scale) × 15 = 360/360, modern settings only,
Gemini gate viab&dist&conc>=4 — best corpus quality ever generated (dist 4.99 / conc 4.99 / decis 4.98 /
fore 4.62 / viab 4.38). v5-style top-up: 484 balanced examples (a kill-orphan double-writer accident
yielded +188 bonus valid rows). BLEND2 = round-7 blend + both, all under the byte-identical v5 contract:
dec 1,349 / wrk 5,396 = 6,745 rows, 0 leaks (hard assert), 0 dups, essence 26.6%. Gen cost ~$25 API.

**Run.** H100 NVL 94GB ($3.25/hr), same recipe (r64/5ep/bs1/ga8/gc/paged_adamw_8bit/router_aux=0.0).
~14.5h ≈ $48 — step time 10.9s vs round-7's 6.5s because blend2 worker rows are LONGER (essence threads
~115 words); estimate from the smoke's measured rate on the actual data, not from a prior corpus.

**Gemini 48-OOD verdict (same judge, same problems as rounds 6-7):**

| model | overall | viab | dist | conc | decis | fore | dist>=4% |
|---|---|---|---|---|---|---|---|
| hs_base | 4.00 | 3.73 | 3.06 | 4.35 | 4.21 | 3.65 | 35% |
| hs_v5 (r7 run1) | 4.56 | 3.47 | 4.87 | 4.83 | 4.77 | 4.40 | 100% |
| hs_blend (r7 SHIP) | 4.52 | 3.58 | 4.81 | 4.75 | 4.85 | 4.12 | 96% |
| **hs_blend2 (r8)** | **4.61** | **3.70** | **4.89** | **4.87** | **4.89** | 4.32 | 98% |
| Haiku 4.5 | 4.73 | 4.23 | 4.71 | 5.00 | 4.88 | 4.56 | 94% |

**Findings:**
1. **Nikhil's viable-cunning hypothesis CONFIRMED (modestly): viability 3.58→3.70 vs ship (+0.23 vs the
   v5 recipe), and blend2 improves on the round-7 ship on ALL SIX dimensions.** New best overall 4.61;
   gap to Haiku −0.17→−0.12. Trained viability now matches H-Small base (3.70 vs 3.73) — the SFT
   viability *tax* is fully paid off while holding +1.8 distinctness over base.
2. Distinctness 4.89 and decisiveness 4.89 = highest on the board incl. Haiku. Concreteness 4.87 within
   0.13 of Haiku. The remaining gap is now almost purely viability (3.70 vs 4.23) + foresight (4.32 vs 4.56).
3. Weakest category: negotiation 4.31 — ironic given the essence round is negotiation-heavy; the judge
   scores negotiation problems hardest on viability. Data alone has diminishing leverage on viability
   (+0.72 from capacity, +0.23 from 850 targeted examples); the next lever is likely inference-time
   (viability self-check pass) or preference-based, not more SFT of the same shape.
4. Ops lessons: per-candidate logging is non-negotiable (a silent fail-streak looked like a stall);
   pkill -f patterns can match parent shells (kill by PID); paragraph-chunkers need sentence-boundary
   fallbacks for wall-of-text inputs; Gemini prepay depletes mid-run (cap-and-resume design absorbed it).

**SHIPPED MODEL = H-SMALL BLEND2 (dec_blend2 + wrk_blend2).** Backup:
`divergent-model-backups/blend2_run/blend2_all.tgz` (764MB, verified). Round cost ≈ $73 ($25 API + $48 GPU).

**§14 addendum — same-session re-judge (rigor check, 2026-07-05).** To remove the cross-session-judging
caveat, all four models were re-judged in ONE Gemini session (Haiku regenerated fresh at temp 0):
3.4B-sft 4.24 | hs_blend 4.48 | hs_blend2 4.55 | Haiku 4.72. Session wobble is ±0.05 on every model;
ALL round-8 conclusions hold: blend2 ≥ r7-ship on five dims (viab 3.46→3.56, dist 4.73→4.90, conc, decis,
mult) and ties foresight (4.21); dist 4.90 tops the board incl. Haiku 4.75; decisiveness 4.83 ties Haiku
exactly. The session-consistent gap to Haiku is −0.17 (not the −0.12 from mixed sessions) — driven by
viability 3.56 vs 4.25 and foresight 4.21 vs 4.52. `benchmark/rejudge_all_summary.json`.

---

## §15 — CONCLUSION (the whole arc, 2026-07-06)

**What was built.** Two models, trained end-to-end on fully synthetic, judge-gated corpora, to perform
one operation a frontier chat model resists: REFRACT a hard decision into four genuinely distinct,
viable strategic threads — and not converge them.

1. **3.4B dense** (granite-4.0-micro, v5 SFT — round 6 ship): proof the *skill* is teachable at all.
2. **32B/9B-active hybrid MoE** (granite-4.0-h-small, BLEND2 SFT — round 8 ship, THE model): proof the
   skill and viability can rise together given active capacity.

**Final scoreboard** (single-session Gemini 2.5 Pro judging, 48 OOD problems, Haiku regenerated fresh —
the strictest comparison run in this project):

| model | overall | viability | distinctness | concreteness | decisiveness | foresight |
|---|---|---|---|---|---|---|
| 3.4B dense SFT | 4.24 | 2.81 | ~4.8 | — | — | 3.81 |
| H-Small blend (r7) | 4.48 | 3.46 | 4.73 | 4.75 | 4.73 | 4.21 |
| **H-Small BLEND2 (ship)** | **4.55** | **3.56** | **4.90** | 4.83 | **4.83** | 4.21 |
| Claude Haiku 4.5 | 4.72 | 4.25 | 4.75 | 4.98 | 4.83 | 4.52 |

The shipped specialist **wins its core objective (distinctness 4.90 > Haiku 4.75)** and **ties Haiku on
decisiveness** — a ~5×-smaller-active-params open model beating a frontier model on the dimension it
was built for. It loses overall (−0.17), and the residue is viability (3.56 vs 4.25) + foresight.

**The five load-bearing findings, in causal order:**
1. **Refraction is teachable** to small open models with purely synthetic corpora — but the naive form
   plateaus (~2.75 distinctness on 3.4B, triangulated three ways in §10).
2. **Corpus QUALITY beats corpus size at every test** — the ceiling broke via source rebalance (§11),
   not scale; the blend's 49 sharp rows moved held-out (+0.45); 850 targeted essence rows later bought
   only +0.10 viability. Data leverage is real and diminishing.
3. **On a 3.4B, distinctness and viability are CAPACITY-ENTANGLED** — DPO (even with clean hard
   negatives) buys viability only by selling distinctness (§11). This is the project's sharpest
   negative result.
4. **The entanglement breaks at ~9B active parameters** — identical data, both dimensions rise together
   (§13). Foresight, flat on 3.4B across every corpus intervention, moved +0.73 the moment capacity
   arrived: it was capacity-bound, exactly as §9 diagnosed.
5. **A cheap strong generator saturates a tightly-prompted rubric** — DeepSeek V4 Pro tied Sonnet 5 on
   corpus quality at 1/6.6 the price and BEAT it on distinctness (§14's A/B); the teacher-ceiling
   intuition failed at this task scale.

**Honest limits.** The specialist does not beat Haiku overall; viability remains capacity/RLHF-bound
territory where 850 targeted examples bought little; the eval rubric is task-specific (this is a
specialist's scoreboard, not a general benchmark); and the next levers (hard-negative DPO at 9B active,
inference-time viability self-check, foresight-targeted data) are named but unrun — the corpus lever is
measurably exhausted, so further SFT-of-the-same-shape is the one direction the evidence rules out.

**Ledger.** Entire journey — 8 rounds, 2 models, ~1,400 corpus examples, 6 GPU training runs, 3
benchmark campaigns — ≈ **$175 all-in** (API + GPU, including one $26 idle-pod mistake, logged). All
adapters + eval artifacts: `~/Desktop/divergent-model-backups/` (v5_run, hsmall_run, blend2_run;
verified archives).

**SHIPPED: H-Small BLEND2 (dec_blend2 + wrk_blend2).** The 3.4B v5 SFT stands as the small-model
reference ship. The project's thesis — that divergent refraction can be distilled into open weights and
beat a frontier model on its own objective — is **demonstrated**, with its costs and its residue stated.
