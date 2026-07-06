**Conserved Operation**:  
*Multi-Angle Divergent Consequence Projection*  
Given a problem, generate a set of reasoning threads where each thread:  
(i) originates from a structurally distinct angle (maximal pairwise inter-thread divergence),  
(ii) maintains grounded structural analogy to the problem's core,  
(iii) explicitly projects consequences along its unique trajectory,  
(iv) avoids convergence/collapse between threads.  

**Candidate Criteria**:  

1. **Information-Theoretic Anchored Diversity**  
   $\mathcal{L} = \underbrace{\sum_{i \neq j} I(T_i; T_j)^{-1}}_{\text{Divergence}} - \lambda \underbrace{\sum_i [D_{KL}(T_i \| A) + D_{KL}(C_i \| A)]}_{\text{Grounding Penalty}}$  
   *Maximize inverse mutual information between threads while minimizing KL-divergence of each thread ($T_i$) and its projected consequences ($C_i$) from the anchor ($A$).*  
   **Stage**: Wander (structural match) + Coverage audit  

2. **Geometric Tension via Determinantal Margin**  
   $\mathcal{L} = \log \det(\mathbf{V}_T^\top \mathbf{V}_T) + \alpha \min_i \cos(\mathbf{v}_{T_i}, \mathbf{v}_A)$  
   *Maximize determinant (volume) of thread embedding matrix $\mathbf{V}_T$ while ensuring minimum cosine similarity between any thread and anchor $\mathbf{v}_A$.*  
   **Stage**: Blender (fusion diversity) + Formalizer grounding  

3. **Adversarial Min-Max Coverage**  
   $\mathcal{L} = \underbrace{\mathbb{E}_{T_i \sim \theta}[\text{Coverage}(T_i)]]}_{\text{Adversary minimizes}} - \beta \underbrace{\mathbb{E}_{T_i,T_j}[\text{Sim}(T_i,T_j)]}_{\text{Generator maximizes}}$  
   *Adversary tries to collapse threads to minimally cover required angles; generator maximizes inter-thread dissimilarity under this pressure.*  
   **Stage**: Middle (coverage checkpoint) + Shepherd trajectory  

**Key Tension Handling**:  
All three enforce divergence *through* their first term while the second term (explicitly or via adversary) maintains grounding. IT uses information compression, Geometric uses embedding geometry, Min-Max uses strategic equilibrium—three distinct mathematical lenses.