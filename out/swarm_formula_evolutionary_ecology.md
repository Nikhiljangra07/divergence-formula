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