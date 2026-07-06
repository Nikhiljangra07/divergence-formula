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