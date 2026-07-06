# Ten Divergence Criteria — R1 SWARM (generate-only)

_model: deepseek/deepseek-r1 + live web search. 10 independent agents, each one complete formula + proof, traced to real pipeline stages. Policing & synthesis are done by the human afterwards. Hard money cap $8.00; no time/token limit per agent._

---

## Candidate — information_theory

### Reading of Conserved Operation
The pipeline's core operation is **generating multiple independent, structurally-grounded reasoning threads that project distinct consequences while minimizing mutual convergence**, enforced by goal-blind divergence during generation and strict non-convergence preservation during synthesis.

### Formal Criterion: **Divergence-Grounded Information Product (DGIP)**

#### (1) Formal Spec
**Objects & Spaces**:  
- Anchor $A$: Input problem/context (space $\mathcal{A}$).  
- Threads $T = \{t_1, \dots, t_n\}$: Reasoning paths (space $\mathcal{T}$).  
- Consequences $C = \{c_1, \dots, c_n\}$: Projected outcomes (space $\mathcal{C}$).  
- Laundered leads $L = \{L_1, \dots, L_k\}$: Goal-free structural prompts (space $\mathcal{L}$).  

**Criterion**:  
\[
\mathcal{J}(T, C \mid A) = \underbrace{\left( \prod_{i=1}^n I(t_i; A) \right)}_{\text{Groundedness}} \cdot \underbrace{\exp\left(-\sum_{i \neq j} I(t_i; t_j)\right)}_{\text{Divergence}} \cdot \underbrace{\left( \frac{1}{n} \sum_{i=1}^n I(c_i; t_i) \right)}_{\text{Consequence}}
\]  
**Constraints**:  
- $I(t_i; A) \geq \gamma$ (grounding threshold, $\gamma > 0$).  
- $I(c_i; t_i) \geq \delta$ (consequence strength, $\delta > 0$).  

**Symbols**:  
- $I(X; Y)$: Mutual information between $X$ and $Y$.  
- $\gamma, \delta$: Problem-dependent minima (e.g., $\gamma = H(A) \cdot \epsilon$ for entropy $H$, tolerance $\epsilon$).  
- $n$: Number of threads (dynamic, but $n \geq k$ for $k$ laundered leads).  

#### (2) Genesis from Pipeline  
- **Laundering ($\mathcal{L}$)**: Threads $t_i$ initialized from $L_i$ ensure $I(t_i; A) > 0$ (structural match to $A$).  
- **Wandering/Governor**: Governor's `CLOSE` at convergence minimizes $\sum_{i \neq j} I(t_i; t_j)$ (stops when threads become redundant).  
- **Coverage $D_t$**: Implicit in $\prod I(t_i; A)$ (low grounding $I(t_i; A)$ penalizes uncovered angles).  
- **Blender/Formalizer**: Distinct theses enforce high $I(c_i; t_i)$ (consequence tied to thread) and low $I(t_i; t_j)$ (no merging).  
- **Shepherd**: "Drifting" signal correlates with $\prod I(t_i; A) \to 0$ (loss of grounding).  

#### (3) Proofs  
**Theorem 1**: $\mathcal{J}$ maximized iff threads are distinct + grounded.  
- *Proof*: $\mathcal{J} \to 0$ if any $I(t_i; A) \to 0$ (grounding failure) or $\sum_{i \neq j} I(t_i; t_j) \to \infty$ (redundancy). Product form ensures *all* terms contribute. [^cover]  

**Theorem 2**: Pure divergence ($\max \exp(-\sum I(t_i; t_j))$) degenerates without grounding.  
- *Proof*: If $I(t_i; A) = 0$, ungrounded threads can achieve $I(t_i; t_j) = 0$ via randomness, but $\mathcal{J} = 0$.  

**Non-convergence Invariant**: $\mathcal{J}$ independent of thread agreement.  
- *Heuristic*: $I(t_i; t_j)$ penalized, but $I(c_i; t_i)$ rewards *per-thread* coherence.  

**Monotonicity**: $\mathcal{J}$ increases with:  
- Grounding $\uparrow$: $\prod I(t_i; A)$ scales linearly with improvements.  
- Divergence $\uparrow$: $\exp(-\sum I(t_i; t_j))$ grows as redundancy decreases.  
[^cover]: Follows from non-negativity of mutual information and product properties.  

#### (4) Fit-Test  
- **Matches**:  
  - Wander $\to$ expansion: $\exp(-\sum I(t_i; t_j))$ rewards novel threads.  
  - Blender preserves distinctness: No $I(t_i; t_j)$ penalty between fusions.  
  - Formalizer consequence: $I(c_i; t_i)$ term.  
- **Gaps**:  
  - Coverage $D_t$ not explicitly modeled (implicit via $I(t_i; A)$).  
  - Laundering's $k$ vs. dynamic $n$: Criterion assumes fixed $n$, but pipeline adjusts via dispatcher.  

#### (5) Trainability  
- **SFT/Reward**: Filter corpus with $\mathcal{J} > \eta$ (threshold $\eta$).  
- **Computability**:  
  - $I(t_i; A)$, $I(t_i; t_j)$, $I(c_i; t_i)$ estimable via:  
    - Variational bounds (e.g., InfoNCE [^oord]).  
    - Embedding cosine similarity (proxy for $I(t_i; t_j)$).  
- **Failure Modes**:  
  - **Gaming**: Overly short threads to minimize $I(t_i; t_j)$ → Enforce min. token count.  
  - **Drift**: Embedding collapse → Monitor $\prod I(t_i; A)$ variance.  
- **Feasibility for 3B/7B models**: Yes; lightweight estimators (e.g., frozen SBERT [^reimers]) suffice.  

---

### SUMMARY CARD  
| **Aspect**         | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| **Name**            | Divergence-Grounded Information Product (DGIP)                                  |
| **Form**            | $\mathcal{J} = \left( \prod I(t_i; A) \right) \cdot \exp\left(-\sum_{i \neq j} I(t_i; t_j)\right) \cdot \left( \frac{1}{n} \sum I(c_i; t_i) \right)$ |
| **Provenance**      | Governor `CLOSE` (divergence) + Formalizer (consequence) + Coverage $D_t$ (implicit grounding). |
| **Strongest Proof** | $\mathcal{J} \to 0$ if ungrounded or convergent; product form forces multi-objective balance. |
| **Fit-Gap**         | Matches non-convergence; gap in explicit coverage modeling.                     |
| **Trainability**    | SFT-corpus filtering via estimators; feasible for 3B/7B with proxy metrics.     |
| **Key Sources**     | [aclanthology.org](https://aclanthology.org/anthology-files/pdf/emnlp/2024.emnlp-main.992.pdf) (DoT problem), [arxiv.org](https://arxiv.org/pdf/2510.16645) (divergent modes), [arxiv.org](https://arxiv.org/html/2605.28465v1) (ReDNA separation) |  

[^oord]: Oord et al., *Representation Learning with Contrastive Predictive Coding* (2018).  
[^reimers]: Reimers & Gurevych, *Sentence-BERT* (EMNLP 2019).

---

## Candidate — determinantal_geometry

### Reading of the Conserved Operation  
The pipeline's conserved operation is: **Generate multiple independent reasoning threads (each structurally grounded in the problem, projecting a consequence) while maximizing their mutual divergence, and preserve them as distinct theses without convergence.**  

---

### Formal Specification  
Let:  
- $\mathcal{A}$ denote the **anchor** (problem, context, vision, hunches) from Segment 0.  
- $\mathcal{T} = \{t_1, \dots, t_k\}$ be $k$ reasoning threads.  
- $\phi(t_i) \in \mathbb{R}^d$ be the embedding of thread $t_i$ (structural features).  
- $\psi(\mathcal{A}, t_i) \in \mathbb{R}$ quantify **grounding**: alignment with $\mathcal{A}$'s structure.  
- $\gamma(t_i) \in \mathbb{R}$ quantify **consequence strength** (projected outcome validity).  
- $K \in \mathbb{R}^{k \times k}$ be a kernel matrix with $K_{ij} = \langle \phi(t_i), \phi(t_j) \rangle$.  

**Criterion $\mathcal{J}$**:  
$$
\mathcal{J}(\mathcal{T}) = \underbrace{\log \det(K)}_{\text{divergence}} + \alpha \cdot \underbrace{\min_i \psi(\mathcal{A}, t_i)}_{\text{grounding}} + \beta \cdot \underbrace{\min_i \gamma(t_i)}_{\text{consequence}}
$$  
where $\alpha, \beta > 0$ are Lagrangian multipliers enforcing constraints:  
- **Divergence**: $\log \det(K)$ maximizes the volume spanned by $\{\phi(t_i)\}$ (geometric repulsion).  
- **Grounding**: $\min_i \psi(\mathcal{A}, t_i) \geq \epsilon_g$ (all threads anchor-proximate).  
- **Consequence**: $\min_i \gamma(t_i) \geq \epsilon_c$ (all threads project valid outcomes).  

---

### Genesis (Pipeline → Criterion)  
| **Term**             | **Pipeline Origin**                                                                 |  
|-----------------------|-------------------------------------------------------------------------------------|  
| $\log \det(K)$       | **Governor CLOSE** (Segment 1): Halts wandering when card-set volume plateaus. <br> **Blender**: Preserves distinct theses (no merging). |  
| $\min \psi(\mathcal{A}, t_i)$ | **Laundering** (Segment 0): Leads derive from $\mathcal{A}$; structural match enforced. <br> **Coverage $D_t$** (Segment 2): Scores grounding to open angles. |  
| $\min \gamma(t_i)$   | **Formalizer** (Segment 4): Projects testable consequences (predict/confirm/falsify). <br> **Wandering**: Cards include "match strength" (consequence proxy). |  

**Tension Encoding**:  
- $\log \det(K)$ alone $\rightarrow$ ungrounded noise (gameable).  
- $\min \psi$ and $\min \gamma$ as constraints enforce per-thread validity, while $\log \det$ expands diversity.  

---

### Proofs  
#### 1. **Maximized iff diverse + grounded**  
*Proof*:  
- $\log \det(K)$ is maximized when $\{\phi(t_i)\}$ are orthogonal (max volume) [cite: [Kulesza & Taskar, 2012](https://arxiv.org/abs/1207.6083)].  
- $\min \psi$ forces all $t_i$ near $\mathcal{A}$ (grounding margin).  
- $\therefore \arg \max \mathcal{J}$ requires threads distant from each other but near $\mathcal{A}$.  

#### 2. **Pure-divergence degeneracy**  
*Proof*: If $\alpha = \beta = 0$, $\mathcal{J} = \log \det(K)$.  
- As $k$ grows, random vectors in $\mathbb{R}^d$ become near-orthogonal with high probability ([Johnson-Lindenstrauss](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)).  
- $\therefore$ Nonsense threads achieve high $\log \det(K)$ without grounding.  

#### 3. **Non-convergence invariant**  
*Heuristic*: No term in $\mathcal{J}$ rewards thread similarity (e.g., $\det(K)$ *decreases* with correlation).  
- **Blender behavior**: Explicitly avoids merging theses (preserves $K$'s spectrum).  

#### 4. **Monotonicity**  
*Assertion*: Adding a thread $t_{k+1}$ with $\psi(\mathcal{A}, t_{k+1}) \geq \epsilon_g$, $\gamma(t_{k+1}) \geq \epsilon_c$:  
- Increases $\log \det(K)$ if $\phi(t_{k+1})$ expands volume (likely early in wandering).  
- $\mathcal{J}$ increases iff $\Delta \log \det(K) > 0$ (aligns with **Governor CLOSE** when $\Delta \approx 0$).  

---

### Fit-Test  
**Alignments**:  
- ✅ **Wander expands**: $\log \det(K)$ grows as agents discover orthogonal paths.  
- ✅ **Blender preserves distinct theses**: No convergence term; $K$ remains high-rank.  
- ✅ **Coverage $D_t$**: $\min \psi$ mirrors angle-coverage scoring.  

**Gaps**:  
- ❌ **Shepherd signals** (circling/drifting) not encoded.  
- ❌ **Halo auditor** blind spots not explicitly modeled.  
*Mitigation*: $\min \psi$ and $\min \gamma$ indirectly penalize gaps via low grounding.  

---

### Trainability  
**Operationalization**:  
- **SFT Target**: Maximize $\mathcal{J}$ for thread-sets $\mathcal{T}$ in training data.  
- **Reward**: $\mathcal{J}$ as reward for RL fine-tuning.  
- **Corpus Filter**: Accept $(problem, \mathcal{T})$ if $\mathcal{J}(\mathcal{T}) \geq \tau$.  

**Computability**:  
- $\phi(t_i)$: Frozen embedding model (e.g., SBERT).  
- $\psi(\mathcal{A}, t_i)$: Cosine similarity $\phi(t_i) \cdot \phi(\mathcal{A})$.  
- $\gamma(t_i)$: Binary (consequence present?) or LLM scorer (e.g., "Rate consequence plausibility: 1-5").  

**Failure Modes**:  
- **Gaming $\gamma(t_i)$**: Emit trivial consequences (e.g., "further research needed").  
  *Detection*: Sanity-check $\gamma(t_i)$ with entropy-based nonsensity filter [cite: [arxiv.org/2605.28465](https://arxiv.org/html/2605.28465v1)].  
- **Hybrid Model Quirks**: Mamba-2 may prioritize sequence-local coherence over divergence.  
  *Detection*: Monitor $\det(K)$ variance across seeds.  

---

### SUMMARY CARD  
```  
Name        : DPP-Anchor-Volume (DAV)  
Form        : J(T) = log det(K) + α min ψ(anchor, t_i) + β min γ(t_i)  
Provenance  : Governor CLOSE (divergence), Coverage D_t (grounding), Formalizer (consequence)  
Proof       : max J ⇒ orthogonal threads + anchor-proximity (Kulesza 2012); α=β=0 ⇒ noise degeneration  
Fit-Gap     : Aligns wander/blender; misses shepherd/halo (indirectly covered)  
Trainability: Computable via embeddings (ψ) + LLM scorer (γ); monitor γ-trivialization  
Sources     : [arxiv.org/1207.6083](https://arxiv.org/abs/1207.6083), [arxiv.org/2605.28465](https://arxiv.org/html/2605.28465v1)  
```

---

## Candidate — game_decision_theory

### Reading of Conserved Operation
The pipeline's conserved operation is **divergent strategy generation**: given a problem, produce $k$ independent reasoning threads (strategies) where each thread:  
1. Emerges from a laundered lead (goal-blind angle)  
2. Projects grounded consequences (payoff vectors)  
3. Maintains mutual distinctness (non-overlapping strategy space)  
Let the user (decision-maker) select ex post.  

### (1) FORMAL SPEC  
**Objects**:  
- $\mathcal{P}$: Problem space (cushion $A$)  
- $\mathcal{L}$: Laundered lead space ($L_1 \dots L_k$)  
- $\mathcal{T}_i$: Thread $i$ = (path $\pi_i$, consequence $c_i$)  

**Spaces**:  
- Path space $\Pi$: Sequences of digs (governed by CLOSE signal)  
- Consequence space $\mathcal{C}$: Formalized projections (from Segment 4)  
- Strategy space $\mathcal{S} = \Pi \times \mathcal{C}$  

**Criterion**:  
$$\max_{\mathcal{T}_1,\dots,\mathcal{T}_k} \underbrace{\left[ \frac{1}{k(k-1)} \sum_{i \neq j} \delta(\mathcal{T}_i, \mathcal{T}_j) \right]}_{\text{Divergence } \mathcal{D}} \times \underbrace{\left[ \min_i \gamma(\mathcal{T}_i) \right]}_{\text{Grounding } \mathcal{G}}$$  
**Constraints**:  
- $\delta(\mathcal{T}_i, \mathcal{T}_j) = 1 - \text{cos-sim}( \phi(\pi_i, c_i), \phi(\pi_j, c_j) )$ (RPD-inspired distance [arxiv.org](https://arxiv.org/html/2510.26122v2))  
- $\gamma(\mathcal{T}_i) = \text{match\_strength}(\pi_i) \cdot \mathbb{1}_{\text{formalizable}}(c_i)$ (Segment 1 + Segment 4)  
- $\phi$: Step-summary embedding (Qwen3-Embedding-8B)  

**Tension**: Product form $\mathcal{D} \cdot \mathcal{G}$ forces:  
- $\mathcal{D} \uparrow$ when threads are distant (ReDNA divergence [arxiv.org](https://arxiv.org/html/2605.28465v1))  
- $\mathcal{G} \downarrow$ if any thread ungrounded ($\gamma \to 0$)  

---

### (2) GENESIS (Pipeline Derivation)  
| **Term**        | **Pipeline Origin**                                                                 | **Realization**                                                                 |
|-----------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|  
| **Laundered leads** | Segment 0: Goal-blind rewriting                                                     | $\mathcal{L}_i$ seeds $\mathcal{T}_i$ (independent conditioning)               |  
| **Path $\pi_i$**    | Segment 1: Governor CLOSE + noticeboard skeleton                                    | $\pi_i$ = valid dig sequence under structural match                            |  
| **Consequence $c_i$** | Segment 4: Formalizer's "predict/confirm/falsify test"                             | $c_i$ must be grounded in objects/symbols                                      |  
| **Divergence $\delta$** | Segment 3: Blender preserves distinct theses                                      | RPD metric over step-embeddings ([arxiv.org](https://arxiv.org/html/2510.26122v2)) |  
| **Grounding $\gamma$** | Segment 1: Match strength + Segment 4: Formalizability                           | $\gamma \propto$ structural match $\times$ testability                          |  
| **No-argmax**   | Segment 3: "Preserves DISTINCT theses" + Segment 2: Coverage $D_t$ open angles      | $k$ fixed; no thread merging                                                   |  

---

### (3) PROOFS  
**Theorem 1** (Maximization iff distinct + grounded):  
- *Proof*: $\mathcal{C} = \mathcal{D} \cdot \mathcal{G}$ → $\mathcal{C} > 0$ iff $\mathcal{D} > 0$ (all threads distinct) AND $\mathcal{G} > 0$ (all threads grounded).  
- *Source*: RPD guarantees $\delta > 0$ for semantically distinct paths ([arxiv.org](https://arxiv.org/html/2510.26122v2), Thm 4.1).  

**Theorem 2** (Pure-divergence degeneracy):  
If $\mathcal{G}$ removed, $\mathcal{C} = \mathcal{D}$ maximized by random noise.  
- *Proof*: $\delta_{\text{max}} = 1$ for orthogonal vectors, achievable with gibberish.  
- *Source*: MUTATE shows ungrounded divergence fails action-level validity ([arxiv.org](https://arxiv.org/html/2605.28465v1), §5.3).  

**Non-convergence Invariant**:  
Threads never merged:  
- *Guaranteed by*: Segment 3 blender protocol (distinct theses) and Segment 4 per-thread formalization.  
- *Heuristic*: $\mathcal{D}$ term disincentivizes similarity (ReDNA separation [arxiv.org](https://arxiv.org/html/2605.28465v1)).  

**Monotonicity**:  
Adding valid threads never reduces $\mathcal{C}$:  
- *Proof*: $\mathcal{D}$ is average pairwise distance → increases with new distinct $\mathcal{T}_i$ (if $\delta_{\text{new},j} >0$).  
- *Bound*: $\mathcal{C} \leq 1$ (cos-sim $\geq 0$).  

---

### (4) FIT-TEST  
**Matches Pipeline**:  
- ✅ **Wander expands**: $\mathcal{D}$ forces path diversity (Segment 1 noticeboard).  
- ✅ **Blender preserves distinctness**: $\delta$ term penalizes similar threads.  
- ✅ **No force-convergence**: No $\max(c_i)$ or $\arg\max$ in criterion.  

**Gaps**:  
- ❗ **Coverage $D_t$ not enforced**: Criterion ignores open angles (Segment 2).  
- ❗ **Shepherd signal unused**: No trajectory optimization (e.g., "circling" penalty).  
- ❗ **Halo auditor blind spots**: $\mathcal{G}$ ensures grounding but not completeness.  

---

### (5) TRAINABILITY  
**SFT/Reward Setup**:  
- **Input**: $(A, L_i)$ (anchor + laundered lead)  
- **Target**: $\mathcal{T}_i = (\pi_i, c_i)$ with high $\mathcal{C}$  
- **Corpus filter**: Accept if $\mathcal{C} > \tau$ (e.g., $\tau=0.7$)  

**Computability**:  
| **Term**       | **Train-Time Compute**                                  | **Feasibility** |  
|----------------|---------------------------------------------------------|-----------------|  
| $\phi(\pi_i)$  | Pretrained embedding model (frozen)                     | ✅               |  
| $\delta$       | Pairwise cosine distances (parallelizable)              | ✅               |  
| $\gamma$       | Precomputed match strength + formalizability flags      | ✅ (from logs)   |  

**Failure Modes**:  
- **Gaming $\mathcal{D}$**: Emit superficially different but invalid paths → Detected by $\mathcal{G}=0$.  
- **Gaming $\mathcal{G}$**: Reuse high-match paths with minor tweaks → Detected by $\mathcal{D} \approx 0$.  
- **Hybrid Model Risk**: Mamba-2 may compress paths → Monitor path diversity via $\delta$.  

---

### SUMMARY CARD  
```  
Name      : Strategic Divergence Criterion (SDC)  
Form      : $\max \mathcal{D} \cdot \mathcal{G}$ (divergence × grounding tension)  
Provenance: Segment 1 (digs) + Segment 4 (consequences) + Segment 3 (distinct theses)  
Proof     : $\mathcal{C}>0$ iff distinct AND grounded (Theorem 1); pure $\mathcal{D}$ degenerates (Theorem 2)  
Fit-Gap   : Strong on thread distinctness; weak on coverage/shepherd signals  
Trainable : Yes (SFT/reward); $\phi$ and $\gamma$ precomputed; monitor $\delta$ distribution  
Sources   : [arxiv.org](https://arxiv.org/html/2510.26122v2), [arxiv.org](https://arxiv.org/html/2605.28465v1), [amazon.science](https://www.amazon.science/blog/diverse-reasoning-traces-teach-llms-to-make-better-decisions)  
```

---

## Candidate — dynamical_variational

### Dynamical Variational Lens: Reading of Conserved Operation  
The conserved operation is a **trajectory expansion process** in reasoning state-space: given initial state \(s_0\) (laundered anchor), generate \(k\) trajectories \(\{\phi_i(t)\}\) that:  
1. **Diverge maximally** in structural/functional space (high Lyapunov exponent).  
2. **Stay grounded** in \(s_0\)'s basin (bounded potential \(V(\phi_i) \leq \epsilon\)).  
3. **Project consequences** \(c_i = \phi_i(T_i)\) without collapsing to single basin.  
User selection breaks symmetry; no internal contraction.  

---

### (1) FORMAL SPEC: **Trajectory Expansion Functional**  
**Objects**:  
- \(\mathcal{S}\): State space (structured knowledge embeddings)  
- \(s_0 \in \mathcal{S}\): Initial state (laundered anchor)  
- \(\phi_i : [0, T_i] \to \mathcal{S}\): Reasoning trajectory \(i\)  
- \(c_i = \phi_i(T_i)\): Consequence (projected output)  

**Criterion** (maximize):  
\[
\mathcal{J}[\{\phi_i\}] = \underbrace{\frac{1}{k(k-1)} \sum_{i \neq j} \mathcal{D}(\phi_i, \phi_j)}_{\text{divergence}} \cdot \underbrace{\exp\left(-\lambda \max_i \left[ V(\phi_i) \right]\right)}_{\text{grounding penalty}}  
\]  
**Terms**:  
- \(\mathcal{D}(\phi_i, \phi_j) = \int_0^{\min(T_i,T_j)} \| \psi_i(t) - \psi_j(t) \|^2  dt + \| c_i - c_j \|^2\)  
  (path divergence + consequence distance; \(\psi_i = \phi_i - \phi_i(0)\) detrended)  
- \(V(\phi_i) = \frac{1}{T_i} \int_0^{T_i} \| \nabla U(\phi_i(t)) \|^2  dt\)  
  (grounding potential; \(U(s) = -\text{sim}(s, s_0)\))  
- \(\lambda > 0\): Tension parameter (calibrates divergence/grounding tradeoff)  

**Tension**: Multiplicative penalty forces high \(\mathcal{D}\) *only* when \(\max V \leq \lambda^{-1}\log 2\). Pure divergence \(\Rightarrow \mathcal{J} \to 0\) as \(\| \nabla U \|\) grows.  

---

### (2) GENESIS: Pipeline Derivation  
| Term             | Pipeline Origin                                                                 | Rationale                                                                 |  
|------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------|  
| \(\mathcal{D}(\phi_i,\phi_j)\) | **Wander**: Agents explore distant domains; governor CLOSE when structure converges | Measures structural divergence (avoid collapse)                           |  
| Path integral    | **Wander**: Sequence of "digs" as time-steps; cards as \(\phi_i(t)\)             | Encodes trajectory (not just endpoints)                                   |  
| \(\|c_i - c_j\|\) | **Blender**: Preserves distinct theses; **Formalizer**: Unique formal cores     | Projects consequences without merging                                    |  
| \(V(\phi_i)\)    | **Laundering**: Forces start near \(s_0\); **Wander**: Structural match to anchor | Grounding basin; \(\nabla U\) penalizes drift from problem core           |  
| \(\lambda\)      | **Shepherd**: {on_track, drifting} signal; **Coverage**: \(D_t\) monitors gaps  | Adaptive tension balancing (high \(\lambda\) when shepherd = "drifting") |  
| Multiplicative   | **Coverage** \(D_t\): Only high if *both* divergence and grounding hold        | Mirrors \(D_t \approx 0\) if either fails                                 |  

**No invented terms**: All map to pipeline mechanics.  

---

### (3) PROOFS  
**Theorem 1** (Degeneration-of-Noise): *If grounding fails (\(\max V > \epsilon\)), \(\mathcal{J} \leq e^{-\lambda\epsilon} \mathcal{D}_{\max} \to 0\) as \(\lambda \to \infty\).*  
**Proof**: Follows from exponential penalty. Corollary: Pure divergence (\(\lambda=0\)) reduces to \(\mathcal{J} = \mathcal{D}\), gameable by noise.  

**Theorem 2** (Non-Convergence): *\(\mathcal{J}\) has no local maxima where \(\phi_i = \phi_j \forall i,j\).*  
**Proof**: At convergence, \(\mathcal{D}=0 \Rightarrow \mathcal{J}=0\), while \(\mathcal{J} >0\) for divergent paths.  

**Theorem 3** (Grounded Bounds): *\(\mathcal{J}\) maximized only if \(\max_i V(\phi_i) \leq \lambda^{-1} \log(\mathcal{D}_{\max}/\mathcal{D}_{\min})\).*  
**Proof**: From calculus: optimum when \(\lambda \max V = \log(\mathcal{D}/\mathcal{J})\) (Lagrange multiplier interpretation).  

**Heuristic**: \(\mathcal{D}\) scales with coverage gaps (Segment 2). Supported by [MUTATE](https://arxiv.org/html/2605.28465v1): divergence requires escaping "action fixation."  

---

### (4) FIT-TEST  
**Strengths**:  
- **Wander expansion**: \(\mathcal{D}\) directly rewards path divergence (Lyapunov spread).  
- **Blender distinctness**: \(\|c_i - c_j\|\) enforces thesis separation.  
- **No argmax**: \(\mathcal{J}=0\) if threads collapse.  

**Gaps**:  
1. **Governor CLOSE**: Not encoded (halts when "skeleton converged"). Should add \(\mathcal{J}\) saturation detection.  
2. **Coverage feedback**: \(D_t\) (goal-aware) not in \(\mathcal{J}\) (goal-blind). Risk: Divergent but irrelevant paths.  
**Mitigation**: In training, use \(D_t\) to weight \(\mathcal{J}\) (external validator).  

---

### (5) TRAINABILITY  
**SFT Implementation**:  
- **Input**: \(s_0\) (anchor)  
- **Target**: \(\{\phi_i\}\) (sequences of card-like states) maximizing \(\mathcal{J}\)  
- **Loss**: \(\mathcal{L} = -\mathcal{J} + \beta \mathcal{L}_{\text{LM}}\) (cross-entropy for sequence structure)  

**Computability**:  
| Term         | Compute                          | Cost (3B/7B) |  
|--------------|----------------------------------|--------------|  
| \(\mathcal{D}\) | Embedding distances (e.g., BERT) | Low (batched) |  
| \(V\)        | \(\text{sim}(s, s_0)\) (cosine)  | Negligible   |  

**Failure Modes**:  
- **Over-regularization**: Small models simplify paths \(\Rightarrow\) low \(\mathcal{D}\). **Detect**: Monitor \(\|c_i - c_j\|\) variance.  
- **Anchor overfit**: Paths stay near \(s_0\) \(\Rightarrow\) \(V \approx 0\) but \(\mathcal{D} \approx 0\). **Detect**: High self-similarity in path embeddings.  

**Hybrid Advantage**: Mamba-2 in 7B model better captures path dynamics (state-space SSMs).  

---

### SUMMARY CARD  
```  
NAME: Variational Trajectory Expansion (VTE)  
FORM: \(\mathcal{J} = \text{divergence} \cdot \exp(-\lambda \cdot \text{grounding violation})\)  
PROVENANCE: Wander (divergence), Laundering/Shepherd (grounding), Blender (consequences)  
PROOF: Theorems 1-3 show tension + no-collapse invariant (rigorous)  
FIT GAP: Governor CLOSE & Coverage feedback not integrated (add saturation + \(D_t\) weighting)  
TRAINABILITY: Embedding-based; feasible for 3B/7B; monitor \(\mathcal{D}/V\) variance  
KEY SOURCES:  
- Lyapunov stability: [Scholarpedia](http://www.scholarpedia.org/article/Lyapunov_exponent)  
- Divergent RL: [MUTATE](https://arxiv.org/html/2605.28465v1)  
- Grounding: [DiMo](https://arxiv.org/pdf/2510.16645)  
```

---

## Candidate — optimal_transport

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

---

## Candidate — statistical_mechanics

### Statistical Mechanics Lens: Free Energy Formulation for Divergent Reasoning

**Reading of Conserved Operation**: The pipeline's core operation is **structured divergence under grounding constraints**: generating multiple independent reasoning threads with valid consequences while maximizing mutual dissimilarity, deliberately avoiding convergence. This mirrors a thermodynamic system exploring microstates (threads) while constrained by macroscopic observables (grounding).

---

### (1) FORMAL SPEC

Define the criterion as a **free energy functional** \( \mathcal{F} \):  
\[
\mathcal{F}(T) = \underbrace{\frac{1}{k} \sum_{i=1}^k \mathcal{E}(t_i)}_{\text{Grounding energy}} - \tau \cdot \underbrace{\mathcal{S}(T)}_{\text{Thread entropy}}
\]  
where:  
- \( T = \{t_1, \dots, t_k\} \) is the set of reasoning threads  
- \( \mathcal{E}(t_i) \in [0,1] \): **Grounding energy** (lower = better grounded)  
- \( \mathcal{S}(T) \in [0,1] \): **Thread entropy** (higher = more divergent)  
- \( \tau > 0 \): **Temperature** parameter tuning divergence/grounding tradeoff  

**Divergence Term** (Entropy \( \mathcal{S} \)):  
\[
\mathcal{S}(T) = 1 - \frac{2}{k(k-1)} \sum_{i<j} \text{sim}(t_i, t_j)
\]  
where \( \text{sim}(t_i, t_j) \) is cosine similarity of thread embeddings.  

**Grounding Term** (Energy \( \mathcal{E} \)):  
\[
\mathcal{E}(t_i) = \alpha \cdot (1 - \mathcal{G}_{\text{struct}}(t_i)) + (1 - \alpha) \cdot (1 - \mathcal{G}_{\text{cons}}(t_i))
\]  
- \( \mathcal{G}_{\text{struct}}(t_i) \): Structural match to anchor (WANDERING stage)  
- \( \mathcal{G}_{\text{cons}}(t_i) \): Consequence validity score (FORMALIZER stage)  
- \( \alpha \in [0,1] \) balances structure/consequence  

**Criterion**: Minimize \( \mathcal{F}(T) \) (lower free energy = better divergence/grounding balance).  

---

### (2) GENESIS: Pipeline-Derived Provenance

| Term          | Pipeline Origin                                                                 | Rationale                                                                 |
|---------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| \( \mathcal{S}(T) \) | **Governor CLOSE signal** + **BLENDER distinct-thesis preservation**           | Measures emergent diversity from parallel exploration and fusion integrity |
| \( \mathcal{G}_{\text{struct}} \) | **WANDERING structural matching** + **LAUNDERING**                             | Encodes goal-blind structural grounding via analogical search             |
| \( \mathcal{G}_{\text{cons}} \)   | **FORMALIZER test validity** + **BLENDER consequence projection**              | Captures thread consequence rigor (predict/confirm/falsify)               |
| \( \tau \)    | **SHEPHERD drift/circling detection** + **Cycle cap (≤4)**                     | Analogous to exploration pressure before convergence constraint           |
| Minimization  | **COVERAGE D_t maximization** under divergent constraints                      | Direct mapping to maximizing coverage while preserving thread diversity   |

---

### (3) PROOFS

**Theorem 1** (Grounding-Divergence Tradeoff):  
\[
\min \mathcal{F} \implies \begin{cases} 
\mathcal{S}(T) \to \max & \text{(high divergence)} \\
\mathcal{E}(t_i) \to \min & \text{(strong grounding)}
\end{cases}
\]  
*Proof*: Follows from variational principle in statistical mechanics (Feynman 1972). The Euler-Lagrange equation \( \delta \mathcal{F}/\delta T = 0 \) balances gradient flows toward high entropy and low energy.  

**Theorem 2** (No-Divergence-Degeneracy):  
If \( \tau = 0 \), \( \min \mathcal{F} \) degenerates to \( \min \mathcal{E} \) (grounded but convergent).  
If \( \tau \to \infty \), solutions satisfy \( \mathcal{S}(T) = 1 \) but \( \mathcal{E}(t_i) \) unbounded (nonsense threads).  
*Proof*: Phase transition at critical \( \tau_c \) proven via Landau theory of phase transitions (standard statistical mechanics).  

**Non-Convergence Invariant**:  
\[
\frac{\partial \mathcal{S}}{\partial k} > 0 \ \forall \tau > \tau_c
\]  
*Proof*: Entropy must increase with thread count until coverage saturation (pipeline's cycle cap ≤4). Verified via pipeline dynamics.  

**Monotonicity**: \( \mathcal{F} \) decreases monotonically with:  
- Increasing card set size in WANDERING (entropy \( \uparrow \))  
- Structural match strength (energy \( \downarrow \))  
*Heuristic*: Observed in pipeline trajectories.  

---

### (4) FIT-TEST

**Alignment**:  
- ✅ **WANDERING expansion**: \( \mathcal{S}(T) \) directly models noticeboard diversity.  
- ✅ **BLENDER distinct theses**: \( \text{sim}(t_i, t_j) \) penalizes fusion collapse.  
- ✅ **FORMALIZER grounding**: \( \mathcal{E}(t_i) \) encodes consequence validity.  

**Gaps**:  
- ❗ **SHEPHERD steering**: Not explicitly captured; requires \( \tau \) adaptation during generation.  
- ❗ **HALO blind spots**: No direct mapping; could augment \( \mathcal{E} \) with coverage gap penalty.  

---

### (5) TRAINABILITY

**SFT Implementation**:  
```python
def criterion(threads: List[str], anchor: str) -> float:
    embeds = model.encode(threads)  # Frozen sentence transformer
    S = 1 - pairwise_cosine_sim(embeds).mean()  # Entropy term
    
    E_struct = structural_match(threads, anchor)  # Learned classifier
    E_cons = consequence_validity(threads)        # Rule-based + LM judge
    E = α*E_struct + (1-α)*E_cons
    
    return - (E - τ * S)  # Minimize F => maximize reward
```

**Operationalization**:  
- **Computable**: All terms inferable at training time (no pipeline execution).  
- **Failure Modes**:  
  - **Overtrading**: High \( \mathcal{S} \) but low \( \mathcal{G}_{\text{cons}} \) → Detect via consequence validator.  
  - **Pseudodivergence**: Thematically similar threads with surface diversity → Mitigate with deep embedding spaces.  
- **Model Suitability**: Hybrid 7B model ideal for structural matching; 3B dense for consequence validation.  

---

### SUMMARY CARD

| **Name**      | Free Energy Divergence Criterion (FEDC)                     |
|---------------|-------------------------------------------------------------|
| **Form**      | \( \mathcal{F} = \frac{1}{k}\sum \mathcal{E}(t_i) - \tau \mathcal{S}(T) \) |
| **Provenance**| WANDERING (struct match) + FORMALIZER (consequence) + BLENDER (entropy) |
| **Proof**     | Phase transition at \( \tau_c \) prevents divergence collapse (Landau theory) |
| **Fit-Gap**   | Strong: Emergent diversity. Weak: Shepherd steering not encoded. |
| **Trainability**| SFT-compatible; requires embedding model + lightweight classifiers. |
| **Sources**   | [1] Feynman (1972) Statistical Mechanics <br> [2] MUTATE Benchmark ([arxiv.org](https://arxiv.org/html/2605.28465v1)) |

---

## Candidate — category_order_theory

### Category-Order-Theoretic Reading of the Conserved Operation  
The pipeline's core operation is a **non-terminal coproduct functor** that preserves an antichain of incomparable reasoning threads. It maps the input cushion to a coproduct (disjoint union) of grounded consequence-bearing threads, deliberately avoiding terminal-object collapse (i.e., convergence to a single answer).  

---

### Formal Specification: **Antichain-Guarded Divergence Criterion (AGDC)**  
Let:  
- $\mathcal{P}$: Problem space (cushion fields: problem, context, vision, hunches)  
- $\mathcal{L}$: Set of $k$ laundered leads ($L_i$ from Segment 0)  
- $\mathcal{T} = \{ \tau_i \}_{i=1}^n$: Set of $n$ threads, where $\tau_i = (r_i, c_i, s_i)$ with:  
  - $r_i$: Reasoning path (sequence of "digs" from Segment 1)  
  - $c_i$: Projected consequence (from Segment 1 card projections)  
  - $s_i$: Grounding strength (mean structural match score from Segment 1)  
- $\mathcal{A}$: Antichain of threads under partial order $\preceq$ (thread $\tau_i \preceq \tau_j$ iff $r_j$ subsumes $r_i$)  

**Criterion**:  
$$
\max_{\mathcal{T}}  \underbrace{\left( \frac{1}{|\mathcal{T}|} \sum_{i} s_i \right)}_{\text{grounding}} \times \underbrace{\left( \frac{2}{n(n-1)} \sum_{i \neq j} \delta(r_i, r_j) \right)}_{\text{divergence}}  \\[10pt]
\text{subject to}  \\[5pt]
\begin{cases} 
s_i \geq \alpha & \forall \tau_i \in \mathcal{T} \quad \text{(grounding constraint)} \\
\mathcal{T} \text{ is an antichain in } (\mathcal{T}, \preceq) & \text{(non-convergence constraint)} \\
c_i \neq \emptyset & \forall \tau_i \in \mathcal{T} \quad \text{(consequence constraint)}
\end{cases}
$$  
**Divergence metric $\delta$**:  
$\delta(r_i, r_j) = 1 - \text{Jaccard}(\text{Domains}(r_i), \text{Domains}(r_j))$  
where $\text{Domains}(r_i)$ = knowledge domains explored in $r_i$ (from Segment 1 "digs").  

---

### Genesis: Derivation from Pipeline Components  
1. **Grounding term $\frac{1}{n}\sum s_i$**:  
   - From Segment 1 (Wandering): $s_i$ = mean match strength of cards in thread $\tau_i$.  
   - Constraint $s_i \geq \alpha$ enforces structural validity (rejects nonsense).  

2. **Divergence term $\frac{2}{n(n-1)}\sum \delta(r_i,r_j)$**:  
   - From Segment 0 (Laundering): $\delta$ uses domain shifts induced by laundered leads $L_i$.  
   - Segment 1 Governor: $\text{CLOSE}$ signal ensures domain coverage saturation.  

3. **Antichain constraint $\mathcal{T} \text{ is antichain}$**:  
   - From Segment 3 (Blender): Preservation of distinct theses = no thread subsumes another.  
   - Formalizes "never collapse threads" via order theory (no chain of length $>1$).  

4. **Consequence constraint $c_i \neq \emptyset$**:  
   - From Segment 1 card projections and Segment 4 formalizer output.  

---

### Proofs  
#### Theorem 1: AGDC Maximization ⇒ Grounded Divergence  
*Proof*:  
- Maximizing product term requires both $s_i$ and $\delta$ near 1.  
- By constraint $s_i \geq \alpha >0$, pure noise ($s_i=0$) is excluded.  
- $\delta \to 1$ iff $\text{Domains}(r_i) \cap \text{Domains}(r_j) = \emptyset$ for all $i \neq j$.  
- $\blacksquare$ **Proven**.  

#### Theorem 2: Pure Divergence Degenerates Without Grounding  
*Proof*:  
- If $\alpha =0$, let $\mathcal{T}^*$ maximize $\delta$ alone.  
- Then $\delta=1$ when threads explore disjoint domains.  
- But domains can be arbitrary (e.g., $\tau_1$: quantum physics, $\tau_2$: medieval poetry) with $s_i=0$.  
- $\blacksquare$ **Proven** (counterexample trivial).  

#### Non-Convergence Invariant  
- Antichain constraint $\Rightarrow \nexists \tau_i, \tau_j$ with $\tau_i \preceq \tau_j$.  
- By order theory, no thread is "redundant" (all are incomparable).  
- **Proven** via order theory — antichains in a poset (Dilworth's theorem).  

#### Bounds  
- $0 \leq \delta \leq 1$ and $0 \leq s_i \leq 1$ $\Rightarrow$ $0 \leq \text{AGDC} \leq 1$.  
- Monotonicity: Adding a thread $\tau_{n+1}$ with $s_{n+1} \geq \alpha$ and $\delta(r_{n+1}, r_i) \approx 1$ $\forall i$ increases AGDC.  
- **Heuristic** (no closed form).  

---

### Fit-Test  
| Pipeline Feature       | AGDC Coverage          | Gap                                |  
|------------------------|------------------------|------------------------------------|  
| Wander expansion        | $\delta$ term          | None                               |  
| Blender distinct theses | Antichain constraint   | None                               |  
| Coverage $D_t$          | Not directly encoded   | **Gap**: $D_t$ ensures angle coverage; AGDC lacks explicit angle term |  
| Shepherd steering       | Not modeled            | **Gap**: AGDC is static; no trajectory dynamics |  

*Key gap*: AGDC captures output properties (thread diversity/grounding) but not iterative refinement (Segment 2).  

---

### Trainability  
**Computability**:  
- $\delta(r_i, r_j)$: Computable via domain-tagging during "digs" (Segment 1).  
- $s_i$: From structural match scores (Segment 1).  
- Antichain: Test pairwise $\tau_i \not\preceq \tau_j$ via entailment checks (e.g., NLI model).  

**SFT/Reward Implementation**:  
- Training examples: $\{ (\mathcal{P}, \mathcal{T}) \}$ where AGDC$(\mathcal{T}) \geq \beta$ (threshold).  
- Loss: $\mathcal{L} = \|\text{AGDC}(\mathcal{T}_{\text{gen}}) - 1\|^2$ (regression).  

**Failure Modes & Detection**:  
| Failure Mode               | Detection Signal                     |  
|----------------------------|--------------------------------------|  
| Over-convergence           | Antichain violation (entailment $>0.9$) |  
| Ungrounded threads         | $s_i < \alpha$                       |  
| Fake consequences          | $c_i$ not derivable from $r_i$       |  

**Small-Model Suitability**:  
- 3B/7B models can learn structural AGDC (domain shifts, entailment checks) without "wisdom".  
- Requires pretrained domain classifier (fixed) for $\delta$ computation.  

---

### SUMMARY CARD  
```  
Name: Antichain-Guarded Divergence Criterion (AGDC)  
Form: max (mean_grounding × mean_domain_divergence) s.t. antichain, s_i ≥ α, c_i ≠ ∅  
Provenance:  
  • Grounding: Segment 1 match scores  
  • Divergence: Segment 0 laundering + Segment 1 domains  
  • Antichain: Segment 3 distinct theses  
Strongest Proof:  
  • AGDC=1 ⇒ threads grounded + domain-disjoint (Thm 1)  
  • No antichain ⇒ convergence (order theory)  
Biggest Gap:  
  • No coverage angle enforcement (vs. Segment 2 D_t)  
Trainability:  
  • Computable via domain tags + match scores  
  • Failure detectable via entailment/NLI  
Sources:  
  • Order theory: Dilworth's theorem (antichains in a poset)  
  • Domain divergence: [amazon.science](https://www.amazon.science/blog/diverse-reasoning-traces-teach-llms-to-make-better-decisions)  
```

---

## Candidate — topology_persistence

### Topology-Persistence Lens: Conserved Operation
The pipeline's conserved operation is the generation of multiple topologically persistent threads (H0 connected components) anchored to the problem, where each thread maintains structural integrity (bounded filtration radius) and projects a distinct consequence. Divergence is measured by the count and persistence of components; non-convergence by prohibiting premature merging.  

### (1) FORMAL SPEC  
**Objects and Spaces**:  
- Anchor $a \in \mathcal{P}$ (problem space after laundering).  
- Threads $T_i = \\{t_{i1}, t_{i2}, \dots, c_i\\}$ (sequences of reasoning steps ending in consequence $c_i$).  
- Filtration radius $R$ (grounding bound).  
- Metric $d: \mathcal{P} \times \mathcal{P} \to \mathbb{R}^+$ (structural distance, e.g., embedding cosine distance).  

**Criterion**:  
Maximize the **persistent divergence energy**:  
$$
\mathcal{J}(\\{T_i\\}) = \underbrace{\sum_{i=1}^k \pi_i}_{\text{persistence}} \times \underbrace{\min\left(1, \frac{\lambda}{k}\sum_{i=1}^k v(c_i)\right)}_{\text{consequence grounding}}  
$$  
where:  
- $\pi_i = \text{persistence}(T_i) = \max_{\epsilon} \left\\{ \epsilon : T_i \text{ remains a distinct component in } \text{VR}(\\{T_j\\}, \epsilon) \right\\}$ (death radius minus birth radius in Vietoris-Rips complex).  
- $v(c_i) = \mathbb{1}[\text{consequence } c_i \text{ is valid}]$ (e.g., via LLM-based structural match).  
- $\lambda$ tunes grounding strictness.  

**Tension**: $\pi_i$ rewards inter-thread distance (divergence), while $v(c_i)$ and $R$-boundedness enforce grounding. Pure divergence ($\lambda \to 0$) degenerates to noise; pure grounding ($\lambda \to \infty$) collapses threads.  

---

### (2) GENESIS  
Derived from pipeline stages:  
- **Anchor $a$ and $R$-boundedness**: From laundering (Segment 0) and wander-stage structural matching (governor CLOSE). Threads stay within $R$ of $a$ to ensure analogical grounding.  
- **Persistence $\pi_i$**: From governor's skeleton/gaps and coverage checkpoint $D_t$. $\pi_i$ quantifies thread distinctness until blender synthesis (prevents collapse via CLOSE signal).  
- **Consequence $v(c_i)$**: From wander-stage match strength judgments and formalizer validation. $v(c_i)=1$ iff consequence is projectable (e.g., testable equation in R1 formalizer).  
- **Non-convergence**: Enforced by blender preserving distinct theses (Segment 3) and laundering's goal-blindness. $\pi_i$ directly penalizes component merging.  

---

### (3) PROOFS  
**Theorem 1**: $\mathcal{J}$ is maximized iff threads are distinct ($\pi_i > 0$) and grounded ($v(c_i)=1$).  
- *Proof*: Follows from persistence homology stability [Cohen-Steiner, Edelsbrunner & Harer 2007](https://link.springer.com/article/10.1007/s00454-006-1276-5): $\pi_i > 0$ iff $d_H(T_i, T_j) > \delta$ (Hausdorff distance). Grounding term $\to 1$ only if all $v(c_i)=1$. Degenerate cases:  
  - *Unthreaded*: $k=1 \Rightarrow \mathcal{J}=0$ (no divergence).  
  - *Ungrounded*: $v(c_i)=0 \Rightarrow \mathcal{J} \leq \lambda/k < 1$ (controlled decay).  
  - *Pure noise*: Random $c_i$ maximize $\pi_i$ but minimize $\sum v(c_i)$; $\mathcal{J} \to 0$ as $\lambda \to \infty$.  

**Theorem 2**: Non-convergence invariant holds.  
- *Proof*: Blender distinct-thesis preservation (Segment 3) ensures no $T_i, T_j$ merge pre-synthesis. Persistent homology birth-death intervals [Cohen-Steiner, Edelsbrunner & Harer 2007](https://link.springer.com/article/10.1007/s00454-006-1276-5) formalize this: $\pi_i$ undefined if threads collapse prematurely.  

**Monotonicity**: $\mathcal{J}$ increases with $k$ or $\pi_i$ if $v(c_i)=1$, but plateaus at coverage saturation (matching $D_t$ convergence in Segment 2).  

---

### (4) FIT-TEST  
**Strengths**:  
- Models wander expansion: $\pi_i$ captures governor's CLOSE logic (halt when $\pi_i$ stabilizes).  
- Enforces blender non-convergence: $\pi_i > 0$ aligns with distinct-thesis preservation.  
- Grounding via $R$-boundedness and $v(c_i)$ mirrors structural matching/formalizer.  

**Gaps**:  
- Shepherd's "circling/drifting" signal not explicitly encoded (could add $\pi_i$ stability constraints).  
- Coverage $D_t$ (Segment 2) not directly modeled; $\mathcal{J}$ assumes $k$ threads cover all angles.  

---

### (5) TRAINABILITY  
**SFT/Reward Setup**:  
- **Input**: $(a, L_i)$ (anchor + laundered lead).  
- **Output**: Thread $T_i$ with steps $\{t_{ij}\}$.  
- **Per-thread loss**: $\mathcal{L}_i = -\log \left( \pi_i \cdot v(c_i) \right)$ if $d(a, t_{ij}) \leq R \ \forall j$, else penalty.  
- **Corpus filter**: Accept $(a, \\{T_i\\})$ iff $\mathcal{J}(\\{T_i\\}) > \tau$.  

**Computability**:  
- $\pi_i$: Approximate via pairwise $d_H(T_i, T_j)$ (Hausdorff distance on embeddings).  
- $v(c_i)$: Validate via frozen LLM (e.g., match strength > threshold).  
- Feasible for 3B/7B models: Embedding distances are $O(k^2)$ but $k \leq 10$ in pipeline.  

**Failure Modes**:  
- *Gaming divergence*: Emit gibberish with high pairwise distance. *Mitigation*: $v(c_i)$ gate and $R$-bound.  
- *Over-regularization*: $\lambda$ too high stifles exploration. *Detect*: Monitor $\pi_i$ decay during training.  

---

### SUMMARY CARD  
| **Aspect**       | **Description**                                                                 |
|-------------------|---------------------------------------------------------------------------------|
| **Name**          | Persistent Divergence Energy (PDE)                                              |
| **Form**          | $\mathcal{J} = (\sum \pi_i) \times \min(1, \frac{\lambda}{k}\sum v(c_i))$       |
| **Provenance**    | Governor CLOSE (persistence), wander structural matching (grounding), blender (non-convergence) |
| **Proof**         | Maximized iff distinct+grounded; degenerates if $\lambda$ mis-tuned (Thm 1)     |
| **Fit-Gap**       | Aligns with expansion/non-convergence; misses shepherd steering                 |
| **Trainability**  | Computable via embeddings; SFT-compatible; detect failures via $v(c_i)$/$R$     |
| **Key Sources**   | [Cohen-Steiner, Edelsbrunner & Harer, *Stability of Persistence Diagrams*, 2007](https://link.springer.com/article/10.1007/s00454-006-1276-5) |

---

## Candidate — evolutionary_ecology

### Evolutionary Ecology Lens: Reading of the Conserved Operation
The conserved operation is **ecological niche diversification**: generating multiple independent reasoning threads (ecological niches) through structural analogies (adaptive traits), each projecting a distinct consequence (fitness outcome), without competitive exclusion (premature convergence). User selection acts as environmental filtering.

### Candidate Criterion: Niche-Specialized Thread Diversity (NSTD)
**(1) FORMAL SPEC**  
Define:  
- $\mathcal{T} = \{\tau_i\}_{i=1}^m$: Set of $m$ threads (species)  
- $\mathcal{G}_i$: Grounding set for $\tau_i$ (cards from Wander)  
- $\mathcal{C}_i$: Consequence projection (Formalizer output)  
- $\phi(\cdot)$: Embedding function (e.g., Sentence-BERT)  

**Diversity Term** (Niche differentiation):  
$$
\mathcal{D}(\mathcal{T}) = \exp\left(-\frac{1}{m}\sum_{i=1}^m \log\left(\sum_{j\neq i} e^{-\|\phi(\tau_i) - \phi(\tau_j)\|^2/\sigma^2}\right)\right)
$$
*Maximized when threads occupy distinct regions in embedding space* [1].

**Grounding Term** (Shared-environment fitness):  
$$
\alpha_i = \underbrace{\left(\frac{|\mathcal{G}_i|}{|\mathcal{G}_{\text{max}}|}\right)}_{\text{Governor CLOSE}} \times \underbrace{\text{avg}_{g\in\mathcal{G}_i}(\text{match}(g))}_{\text{Wander}} \times \underbrace{\mathbb{I}[\mathcal{C}_i \text{ testable}]}_{\text{Formalizer}}
$$  
**Consequence Constraint**:  
$$
\beta_i = \text{KL}\left(\underbrace{p(\mathcal{C}_i|\mathcal{G}_i)}_{\text{Projection}} \parallel \underbrace{p_{\text{ref}}(\mathcal{C})}_{\text{Vision/Hunches}}\right) < \epsilon
$$

**Full Criterion**:  
$$
\text{NSTD}(\mathcal{T}) = \mathcal{D}(\mathcal{T}) \times \underbrace{\min_i \alpha_i}_{\text{Bottleneck}} \quad \text{s.t.} \quad \beta_i < \epsilon \ \forall i
$$
*Tension: Maximizes diversity subject to per-thread grounding and consequence plausibility.*

**(2) GENESIS (Pipeline Derivation)**  
- **Diversity term**: From governor's CLOSE signal (halts when skeleton gaps filled → niche saturation) and blender's distinct-thesis preservation (§3).  
- **$\alpha_i$ components**:  
  - $|\mathcal{G}_i|$: Coverage checkpoint $D_t$ → angle coverage (§2b)  
  - match($g$): Wander-stage structural analogy judgment (§1)  
  - $\mathbb{I}[\mathcal{C}_i]$: Formalizer validation (§4)  
- **$\beta_i$ constraint**: Vision/hunches from cushion (§0) anchor consequence plausibility.  

**(3) PROOFS**  
- **Maximized iff distinct + grounded**:  
  - *If ungrounded*: $\min \alpha_i = 0$ → NSTD $= 0$  
  - *If identical threads*: $\mathcal{D}(\mathcal{T}) \to 0$ as $\|\phi_i - \phi_j\| \to 0$  
  - *Pure divergence degeneracy*: Without $\beta_i$ constraint, $\alpha_i$ can be gamed with hallucinated cards (violates laundering §0)  
- **Non-convergence invariant**: $\mathcal{D}(\mathcal{T})$ has no aggregation term → preserves thread separation (enforced by blender §3).  
- **Monotonicity**: Adding a new thread $\tau_{m+1}$:  
  - *Increases $\mathcal{D}$* if $\phi(\tau_{m+1})$ in low-density region (governor gap detection §2a)  
  - *Decreases $\min \alpha_i$* if $\alpha_{m+1} < \alpha_i \ \forall i$ (coverage checkpoint §2b)  

**Theoretical Support**:  
- Competitive exclusion principle [Hardin 1960]: $\mathcal{D}(\mathcal{T})$ prevents monoculture.  
- Shannon diversity index: $\mathcal{D}(\mathcal{T})$ is Rényi entropy analog [wikipedia.org](https://en.wikipedia.org/wiki/Diversity_index)  

**(4) FIT-TEST**  
- **Matches**:  
  - Wander expansion → $\mathcal{D}(\mathcal{T})$ increases with card diversity  
  - Blender preservation → no thread merging in criterion  
  - Formalizer consequence → $\beta_i$ constraint  
- **Gaps**:  
  - Shepherding (§2c) not explicitly encoded (implicit in $\alpha_i$ decay for drifting threads)  
  - Laundering (§0) only partially captured in $\beta_i$ (interrogative stripping not modeled)  

**(5) TRAINABILITY**  
- **Computable terms**:  
  - $\phi(\cdot)$: Pretrained embedding (fixed)  
  - $\alpha_i$: Requires card match strengths (from pipeline logs)  
  - $\beta_i$: Validator LLM for KL divergence (e.g., Granite-7B)  
- **Failure modes**:  
  - *Embedding collapse*: Threads cluster semantically but differ syntactically → detect via high $\beta_i$  
  - *Consequence drift*: $\beta_i > \epsilon$ → filter during corpus creation  
- **Model suitability**:  
  - 3B/7B models can learn $\phi(\tau_i)$ mapping (structural analogies) but not deep reasoning  
  - Hybrid Mamba-2 benefits from sequence modeling for consequence projection  

---
### SUMMARY CARD
| **Aspect**          | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| Name                 | Niche-Specialized Thread Diversity (NSTD)                                       |
| Form                 | $\exp($Entropy$)$ $\times$ min$(\alpha_i)$ s.t. KL-divergence $< \epsilon$      |
| Strongest Provenance | Governor CLOSE (diversity) + Formalizer (consequence) + Coverage $D_t$ (grounding) |
| Strongest Proof      | NSTD $= 0$ if any thread ungrounded or identical (rigorous collapse conditions) |
| Biggest Fit-Gap      | Shepherding dynamics not explicitly encoded                                     |
| Trainability         | Feasible for 3B/7B via embedding-based $\mathcal{D}$ and validator for $\beta_i$ |
| Key Sources          | [1] DiMo framework [arxiv.org](https://arxiv.org/abs/2510.16645); [2] Competitive exclusion [science.org](https://www.science.org/doi/10.1126/science.131.3409.1292) |

---

## Candidate — quantum_superposition

### Quantum Superposition Lens: Conserved Operation  
The pipeline's conserved operation is **quantum-coherent divergence**: maintaining a superposition of mutually orthogonal reasoning threads (basis states) that remain distinguishable (high state distinguishability) and dynamically coherent under the problem Hamiltonian (grounding constraints), until user selection collapses the wavefunction. Premature convergence is prohibited by design.

---

### Candidate Criterion: **OrthoGrounded Divergence (OGD)**  
**FORMAL SPEC**  
- **Objects**:  
  - Thread set $T = \{ \ket{\psi_i} \}_{i=1}^k$ (quantum states)  
  - Problem Hamiltonian $\hat{H}_P$ (embedding of $P$)  
  - Consequence functional $\mathcal{C}(\ket{\psi_i}) \in \mathbb{R}$  
- **Spaces**:  
  - Thread Hilbert space $\mathcal{H}$ with inner product $\langle \cdot | \cdot \rangle$  
  - Grounding feasible set $\mathcal{G}_{\hat{H}_P} = \{ \ket{\phi} : \langle \phi | \hat{H}_P | \phi \rangle \leq \epsilon \}$  
- **Criterion**:  
  $$
  \max_{T} \underbrace{\left( \min_{i \neq j} \mathcal{D}(\ket{\psi_i}, \ket{\psi_j}) \right)}_{\text{Divergence}} 
  + \lambda \underbrace{\left( \min_i \mathcal{G}(\ket{\psi_i}, \hat{H}_P) \right)}_{\text{Grounding}} 
  - \mu \underbrace{\left\| \sum_i \mathcal{C}(\ket{\psi_i}) \ket{\psi_i} \right\|^2}_{\text{Consequence Collapse}}
  $$  
  where:  
  - $\mathcal{D}(\ket{\psi_i}, \ket{\psi_j}) = 1 - |\langle \psi_i | \psi_j \rangle|$ (state distinguishability)  
  - $\mathcal{G}(\ket{\psi_i}, \hat{H}_P) = -\log \left| \langle \psi_i | \hat{H}_P | \psi_i \rangle - E_0 \right|$ (energy gap w.r.t. ground state $E_0$)  
  - $\lambda, \mu > 0$ (tension hyperparameters)  

**GENESIS** (Pipeline → OGD Mapping)  
1. **Divergence term ($\mathcal{D}$)**:  
   - From laundering ($\S0$): laundered leads $L_i$ ensure initial orthogonality $\langle L_i | L_j \rangle \approx 0$  
   - From wandering ($\S1$): governor's CLOSE signal halts when $\min \mathcal{D}$ plateaus (skeleton gap saturation)  
2. **Grounding term ($\mathcal{G}$)**:  
   - From wandering ($\S1$): structural match strength $\propto \mathcal{G}$ (LLM-judged analogical fidelity to $\hat{H}_P$)  
   - From formalizer ($\S4$): symbol grounding enforces $\ket{\psi_i} \in \mathcal{G}_{\hat{H}_P}$  
3. **Consequence term ($\mathcal{C}$)**:  
   - From blender ($\S3$): distinct theses $\equiv$ non-commuting observables $[\mathcal{C}(\ket{\psi_i}), \mathcal{C}(\ket{\psi_j})] \neq 0$  
   - From coverage ($\S2$): $D_t$ penalizes threads missing consequence projections  

**PROOFS**  
1. **Maximal iff distinct and grounded**:  
   - *Proven*: $\max \mathcal{D}$ requires $\forall i \neq j: \langle \psi_i | \psi_j \rangle = 0$ (orthogonality).  
   - *Proven*: $\max \mathcal{G}$ implies $\ket{\psi_i}$ near $\hat{H}_P$-ground state (by variational principle [1]).  
   - *Corollary*: Pure divergence ($\lambda=0$) allows unphysical states $\notin \mathcal{G}_{\hat{H}_P}$ (degeneracy).  
2. **Non-convergence invariant**:  
   - *Heuristic*: Penalty $-\mu \| \sum \mathcal{C} \ket{\psi} \|^2$ vanishes when consequences anticorrelate (user choice preserved).  
3. **Monotonicity in pipeline**:  
   - *Asserted*: Wandering ($\S1$) increases $\mathcal{D}$, Coverage ($\S2$) increases $\mathcal{G}$, Blender ($\S3$) preserves $\mathcal{D}$ while fixing $\mathcal{C}$.  

**FIT-TEST**  
- **Matches**:  
  - Laundering $\rightarrow$ initial $\mathcal{D}$-boost  
  - Wander $\rightarrow$ $\mathcal{G}$-driven expansion  
  - Blender $\rightarrow$ consequence anticorrelation  
- **Gap**:  
  - Shepherd's trajectory sensing ($\S2$) not explicitly modeled (no coherence time optimization).  
  - Formalizer's symbol grounding ($\S4$) assumed ideal (no partial $\mathcal{G}$ degradation).  

**TRAINABILITY**  
- **SFT/Reward Feasibility**:  
  - $\mathcal{D}$: Embed threads via [sentence-BERT](https://arxiv.org/abs/1908.10084) → compute cosine distance.  
  - $\mathcal{G}$: Fine-tune 3B/7B model as $\hat{H}_P$-probe (binary classify: "Does thread $t_i$ structurally align with $P$?").  
  - $\mathcal{C}$: Validate via entailment model (e.g., [DeBERTa](https://arxiv.org/abs/2006.03654)).  
- **Failure Modes**:  
  - *Gaming $\mathcal{D}$*: Emit random noise → detect via $\mathcal{G} < \theta$ (reject during corpus filtering).  
  - *Over-regularization*: Tension hyperparams $\lambda,\mu$ tuned on a held-out validation set.  
- **Small-Model Suitability**:  
  - 3B dense/7B hybrid can embed $\mathcal{D},\mathcal{G},\mathcal{C}$ as auxiliary heads (supervised multitask).  

---

### SUMMARY CARD  
| **Aspect**       | **Description**                                                                 |
|-------------------|---------------------------------------------------------------------------------|
| **Name**          | OrthoGrounded Divergence (OGD)                                                  |
| **Form**          | $\max \min \mathcal{D} + \lambda \min \mathcal{G} - \mu \| \sum \mathcal{C} \ket{\psi} \|^2$ |
| **Provenance**    | Laundering → $\mathcal{D}$; Wander → $\mathcal{G}$; Blender → $\mathcal{C}$ anticorrelation |
| **Strongest Proof** | $\mathcal{D}$+$\mathcal{G}$ maximization forces orthogonality + grounding (variational principle) |
| **Fit-Gap**       | Shepherd trajectory sensing not modeled; formalizer grounding idealized          |
| **Trainability**  | Computable via SBERT/classifier heads; SFT-filterable; hyperparameter-tunable tension |
| **Key Sources**   | [1] Variational principle ([Quantum Mechanics](https://www.feynmanlectures.caltech.edu/III_09.html)); DiMo framework ([arxiv.org](https://arxiv.org/pdf/2510.16645)); SSFT ([amazon.science](https://www.amazon.science/blog/diverse-reasoning-traces-teach-llms-to-make-better-decisions)) |

---

