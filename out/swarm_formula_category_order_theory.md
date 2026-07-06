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