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