# ARKWELL OPU: Hardware-Aware Conflict Model

## 1. Overview and Problem Definition
A critical bottleneck in parallel local search architectures is managing race conditions and mutual state corruption. The **ARKWELL Conflict Model** provides the formal theoretical basis for determining when two or more candidate updates can execute concurrently.

> **Research Warning:** A conflict graph does not constitute a patentable invention simply by implementing graph coloring. The research novelty lies in the precise separation and synthesis of distinct dependency classes, higher-order interaction analysis, and dynamic constraint-slack-aware conflict elimination.

---

## 2. Taxonomy of Conflict Classes

Let $u_i = (\Delta x_i, i)$ and $u_j = (\Delta x_j, j)$ be candidate state transitions on variables $x_i$ and $x_j$ ($i \neq j$).

### 2.1 Read/Write (RAW / WAR) Conflict
- **Definition:** Occurs when the evaluation of candidate $u_i$ reads a state variable that candidate $u_j$ writes.
- **Form:** Let $\mathcal{R}(u_i)$ be the read set of $u_i$, and $\mathcal{W}(u_j) = \{j\}$ be the write set of $u_j$.
- A RAW/WAR conflict exists if:
  $$\mathcal{R}(u_i) \cap \mathcal{W}(u_j) \neq \emptyset \iff j \in \mathcal{R}(u_i)$$

### 2.2 Objective Coupling Conflict
- **Definition:** Occurs when the joint objective delta does not equal the sum of independent objective deltas:
  $$\Delta f(\{u_i, u_j\}) \neq \Delta f_i + \Delta f_j$$
- For linear objectives:
  $$f(\mathbf{x}) = \sum c_k x_k \implies \Delta f(\{u_i, u_j\}) = \Delta f_i + \Delta f_j$$
  Hence, linear objectives produce zero objective coupling conflicts.
- For quadratic / polynomial objectives (e.g. QUBO $x_i Q_{ij} x_j$), an objective conflict exists whenever $Q_{ij} \neq 0$.

### 2.3 Constraint Dependency Conflict
- **Definition:** Two updates share one or more constraints in the bipartite constraint graph:
  $$\mathcal{C}(i) \cap \mathcal{C}(j) \neq \emptyset$$
  Where $\mathcal{C}(i) = \{k \mid x_i \text{ is incident to constraint } g_k\}$.

### 2.4 Joint Feasibility Violation Conflict (Feasibility Incompatibility)
- **Crucial Architectural Insight:** Constraint sharing ($\mathcal{C}(i) \cap \mathcal{C}(j) \neq \emptyset$) **does NOT strictly imply** an execution conflict!
- Let constraint $k$ be $x_i + x_j \le 1$.
- Case 1: Both $u_i$ and $u_j$ propose $0 \to 1$.
  - Joint outcome: $x'_i + x'_j = 2 > 1$. **Violates constraint! This is an active conflict.**
- Case 2: $u_i$ proposes $1 \to 0$ and $u_j$ proposes $0 \to 1$.
  - Joint outcome: $x'_i + x'_j = 0 + 1 = 1 \le 1$. **Feasible! No violation!**
- Case 3: Both propose $1 \to 0$.
  - Joint outcome: $x'_i + x'_j = 0 \le 1$. **Feasible! No violation!**

**Rule:** Two updates sharing constraint $g_k$ conflict if and only if their concurrent execution drives $g_k$ into violation:
$$g_k(\mathbf{x} + \Delta \mathbf{x}_{i,j}) > 0$$

### 2.5 Noncommuting Updates
- **Definition:** The resulting state depends on the order of execution:
  $$u_i(u_j(\mathbf{x})) \neq u_j(u_i(\mathbf{x}))$$
- In pure binary coordinate flips, coordinate updates commute under bitwise XOR:
  $$\mathbf{x} \oplus \mathbf{e}_i \oplus \mathbf{e}_j = \mathbf{x} \oplus \mathbf{e}_j \oplus \mathbf{e}_i$$
  Hence, commutativity is preserved at the bit-level, but evaluation decisions made asynchronously based on stale state may diverge.

### 2.6 Resource Conflicts
- **Definition:** Physical PE or memory bank allocation collisions. Two updates assigned to the same PE or competing for the same memory bank cannot be scheduled simultaneously.

---

## 3. Pairwise Conflict Graph $\mathcal{G}_{\text{conf}}$

Given a set of candidate updates $\mathcal{U} = \{u_1, \dots, u_K\}$:
- Vertices: $V_{\text{conf}} = \mathcal{U}$.
- Edges: $(u_a, u_b) \in E_{\text{conf}} \iff \text{Conflict}(u_a, u_b) = \text{True}$.

### 3.1 Pairwise vs. Higher-Order Interactions
- While pairwise edge conflicts are sufficient for binary pairwise constraints ($x_i + x_j \le 1$), generalized constraints (e.g. $\sum_{i \in S} x_i \le K$) exhibit **hyper-conflicts**.
- For example, if three variables can each be set to 1 individually, but only at most two simultaneously, pairwise checks do not suffice.
- In ABS V1.0 (MIS), every constraint is pairwise, so pairwise conflict graph analysis is both necessary and sufficient. For higher-order extensions, the verifier performs hyper-clique validation before commit.
