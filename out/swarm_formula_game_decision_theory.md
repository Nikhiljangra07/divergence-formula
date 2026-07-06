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