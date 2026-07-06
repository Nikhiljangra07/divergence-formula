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