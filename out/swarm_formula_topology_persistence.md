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