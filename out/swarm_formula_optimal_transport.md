### Reading of the Conserved Operation
The pipeline's core operation is **divergent expansion with grounded consequences**: generating multiple independent reasoning threads (each projecting a distinct consequence) while maintaining structural grounding in the problem, deliberately avoiding convergence until human selection. This is formalized as a measure over option-space where dispersion (divergence) and support constraints (grounding) are jointly optimized.

---

### (1) FORMAL SPEC  
**Objects and Spaces**:  
- $\mathcal{P}$: Problem space (cushion after laundering).  
- $\mathcal{T}$: Thread space (each thread $t_i$ is a sequence of digs/cards).  
- $\mathcal{O}$: Consequence space (formalized projections).  
- $\mu_T := \frac{1}{n}\sum_{i=1}^n \delta_{t_i}$: Uniform measure over $n$ threads.  

**Criterion**:  
$$\max_{\mu_T} \underbrace{\mathbb{E}_{t,t' \sim \mu_T \otimes \mu_T} \left[ d(t,t')^2 \right]}_{\text{divergence}} + \lambda \cdot \underbrace{\inf_{p \in \mathcal{P}} \mathbb{E}_{t \sim \mu_T} \left[ c(t, p) \right]}_{\text{grounding}} \cdot \underbrace{\mathbb{E}_{t \sim \mu_T} \left[ \| F(t) \|_{\mathcal{O}} \right]}_{\text{consequence}}$$  
**Terms**:  
- $d: \mathcal{T} \times \mathcal{T} \to \mathbb{R}^+$: Wasserstein-2 metric on thread embeddings (e.g., via BERT).  
- $c: \mathcal{T} \times \mathcal{P} \to [0,1]$: Grounding score (1 if $t$ uses valid structure from Segment 1).  
- $F: \mathcal{T} \to \mathcal{O}$: Consequence map (formalizer output quality, Segment 4).  
- $\lambda > 0$: Trade-off hyperparameter.  

**Tension**: The product $\inf_p \mathbb{E}[c] \cdot \mathbb{E}[\|F\|]$ forces grounding and consequence to be *simultaneously* satisfied; divergence is unbounded but penalized if grounding/consequence collapse. Pure divergence ($\lambda=0$) allows ungrounded nonsense.

---

### (2) GENESIS  
**Line-by-line derivation from pipeline**:  
- **Divergence term**:  
  $\mathbb{E}[d(t,t')^2]$ encodes *structural dispersion* of threads. Provenance:  
  - Segment 0 (laundering): Independent leads $L_i$ force initial separation.  
  - Segment 1 (wandering): Governor's `CLOSE` halts only when skeleton gaps exist (non-redundancy).  
  - Segment 3 (blender): Distinct theses preserved → high $d(t,t')$.  
- **Grounding term**:  
  $\inf_p \mathbb{E}[c(t,p)]$ ensures threads anchor to $\mathcal{P}$. Provenance:  
  - Segment 1 (digs): Cards require structural match to anchor (via LLM judge).  
  - Segment 2 (halo auditor): Blind spots penalize threads drifting from $\mathcal{P}$.  
- **Consequence term**:  
  $\mathbb{E}[\|F(t)\|]$ maps to formalized outputs. Provenance:  
  - Segment 4 (R1 formalizer): $F(t)$ explicit as "predict/confirm/falsify test".  
- **Product form**:  
  Tension reflects Segment 2 coverage checkpoint: $D_t$ (coverage) fails if grounding *or* consequence weakens.  

---

### (3) PROOFS  
**Theorem 1** (Degeneracy of pure divergence):  
If $\lambda = 0$, $\max \mathbb{E}[d^2]$ is achieved by maximizing pairwise distance, *ignoring* $\mathcal{P}$.  
*Proof*: Follows from [Monge-Kantorovich duality](https://en.wikipedia.org/wiki/Transportation_theory) – Wasserstein dispersion is unconstrained without support restrictions. $\blacksquare$  

**Theorem 2** (Grounded divergence):  
$\mu_T$ maximizes criterion iff:  
(i) $\text{supp}(\mu_T) \subseteq G(\mathcal{P})$ (grounding set),  
(ii) Threads are $\epsilon$-separated in $\mathcal{T}$,  
(iii) Each $F(t_i)$ is non-trivial.  
*Proof sketch*: Lagrangian-like product $\inf_p \mathbb{E}[c] \cdot \mathbb{E}[\|F\|]$ acts as a barrier – vanishing if grounding *or* consequence fails. $\mathbb{E}[d^2]$ then reduces to dispersion over $G(\mathcal{P}) \cap F^{-1}(\mathcal{O}\setminus\{0\})$. $\square$ (Heuristic: No known OT theorem covers product constraints; empirically validated in [arxiv.org](https://arxiv.org/html/2605.28465v1))  

**Non-convergence invariant**:  
No term rewards thread similarity; blender's distinct-thesis preservation (Segment 3) ensures $\mu_T$ avoids collapse.  

**Monotonicity**:  
Divergence $\mathbb{E}[d^2]$ increases with cycle count in Segment 1 (wandering expands) but is capped by grounding (Segment 2 shepherds drifting threads).  

---

### (4) FIT-TEST  
**Alignment**:  
- ✅ **Wander expands**: Divergence term grows with agent independence.  
- ✅ **Blender preserves distinctness**: No convergence force in criterion.  
- ✅ **Coverage $D_t$**: Grounding term correlates with angle coverage.  

**Gaps**:  
- ❗ **Governor's skeleton gaps**: Not explicitly modeled (loss of structural topology).  
- ❗ **Shepherd's trajectory sensing**: "Circling/drifting" not captured in $d(\cdot,\cdot)$.  
- ❗ **Consequence threshold**: $\|F(t)\|$ lacks Segment 4's "partial/yes/no" granularity.  

---

### (5) TRAINABILITY  
**Operationalization**:  
- **SFT target**: For input $p$, generate $\{t_i\}$; optimize $\mathcal{J}$ via contrastive loss.  
- **Reward**: $\mathcal{J}$ as RL reward; use REINFORCE with baseline.  
- **Corpus filter**: Accept $(p, \{t_i\})$ iff $\mathbb{E}[c] \geq \tau_g$ and $\mathbb{E}[\|F\|] \geq \tau_c$.  

**Computability**:  
- $d(t,t')$: Embed threads with frozen BERT → Euclidean distance.  
- $c(t,p)$: Binary classifier (trained on Segment 1 dig acceptance).  
- $F(t)$: Scalar output from R1 formalizer (or distilled 3B model).  
- **Feasible for 3B/7B models**: All terms O($n^2$) but $n \leq 4$ (cycle cap).  

**Failure modes**:  
- **Collapse to "safe" threads**: If $\lambda$ too large, model minimizes risk → low divergence. *Detect*: Monitor $\mathbb{E}[d^2]$.  
- **Consequence gaming**: Optimize $\|F\|$ via verbose but vacuous outputs. *Detect*: Validate test cases from Segment 4.  

---

### SUMMARY CARD  
| **Aspect**          | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| **Name**             | OT-Div (Optimal Transport Divergence Criterion)                                |
| **Form**             | $\max \mathbb{E}[d^2] + \lambda \cdot (\inf_p \mathbb{E}[c]) \cdot \mathbb{E}[\|F\|]$ |
| **Strongest Provenance** | Segment 1 (governor's `CLOSE` on skeleton gaps → dispersion) + Segment 4 (consequence map) |
| **Strongest Proof**  | Theorem 1 (pure divergence degeneracy) via OT duality                          |
| **Biggest Fit-Gap**  | Governor's structural skeleton not encoded (loss of topology)                  |
| **Trainability**     | Feasible: All terms computable via embeddings/classifiers; watch for $\lambda$-induced collapse |
| **Key Sources**      | [aclanthology.org](https://aclanthology.org/anthology-files/pdf/emnlp/2024.emnlp-main.992.pdf) (DoT problem); [arxiv.org](https://arxiv.org/html/2605.28465v1) (MUTATE divergence levels) |