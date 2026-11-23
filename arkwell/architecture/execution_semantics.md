# ARKWELL OPU: Execution Semantics Specification (ABS V1.0)

## 1. Scope and Research Context
This document specifies the formal mathematical and computational execution semantics for the **Optimization Processing Unit (OPU)** architecture under research in the **ARKWELL** project.

> **Status:** Research Hypothesis / Formal Model Specification (ABS V1.0)  
> **Target:** Massively parallel, digital, constraint-aware discrete optimization.

---

## 2. Problem Representation & Binary Variable Model

An optimization problem instance is defined on a discrete binary state space:

$$\mathbf{x} = (x_0, x_1, \dots, x_{N-1})^\top \in \{0, 1\}^N$$

Where:
- $N \in \mathbb{N}$ denotes the number of binary decision variables.
- Each coordinate $x_i \in \{0, 1\}$ represents the selection or activation state of variable $i$.
- $\mathcal{X} = \{0, 1\}^N$ is the full state configuration space ($|\mathcal{X}| = 2^N$).

### 2.1 State Representation in Hardware
- **Physical Bitvector:** The state vector $\mathbf{x}$ is managed within the OPU State Manager in compact word-aligned bitvector representations.
- **Atomic Read/Write Granularity:** The smallest unit of state modification is a single bit flip $\Delta x_i \in \{-1, +1\}$.
- **Coordinate Indexing:** $i \in \{0, \dots, N-1\}$.

---

## 3. Objective Function Formulation

The global objective function maps a configuration to a real-valued scalar:

$$f: \{0, 1\}^N \to \mathbb{R}$$

For the Maximum Independent Set (MIS) canonical workload:

$$\max_{\mathbf{x} \in \mathcal{X}} \quad f(\mathbf{x}) = \sum_{i=0}^{N-1} x_i$$

### 3.1 Local Objective Delta ($\Delta f_i$)
When variable $x_i$ undergoes a candidate transition $x_i \to 1 - x_i$:

$$\Delta x_i = (1 - x_i) - x_i = 1 - 2x_i \in \{-1, +1\}$$

The exact delta in objective value is:

$$\Delta f_i(\mathbf{x}) = f(\mathbf{x} \oplus \mathbf{e}_i) - f(\mathbf{x}) = \Delta x_i$$

Where $\mathbf{e}_i$ is the $i$-th standard basis vector. In linear objectives, $\Delta f_i$ depends purely on the current state $x_i$ and is decoupled from $x_j$ ($j \neq i$).

---

## 4. Constraint Formulation & Feasibility Semantics

Constraints are represented as a system of $M$ inequality or equality relations:

$$\mathbf{g}(\mathbf{x}) \le \mathbf{0}, \quad g_k: \{0, 1\}^N \to \mathbb{R}, \quad k \in \{0, \dots, M-1\}$$

For Maximum Independent Set defined on an undirected graph $G = (V, E)$ with $|V| = N$ and $|E| = M$:

$$\forall (i, j) \in E: \quad g_{i,j}(\mathbf{x}) = x_i + x_j - 1 \le 0$$

Equivalently:

$$x_i \cdot x_j = 0 \quad \forall (i, j) \in E$$

### 4.1 Constraint State Vector
Let $v_k(\mathbf{x}) = \max(0, g_k(\mathbf{x}))$ denote the violation magnitude of constraint $k$.
The configuration $\mathbf{x}$ is **feasible** if and only if:

$$\sum_{k=0}^{M-1} v_k(\mathbf{x}) = 0 \iff \forall (i, j) \in E, \; x_i + x_j \le 1$$

The feasible subspace is defined as:

$$\mathcal{F} = \{\mathbf{x} \in \mathcal{X} \mid \forall (i, j) \in E, \; x_i + x_j \le 1\}$$

---

## 5. Candidate Updates & Local Search Semantics

A candidate update $u_i$ is a proposed bit flip on variable $x_i$:

$$u_i: \mathbf{x} \mapsto \mathbf{x} \oplus \mathbf{e}_i$$

### 5.1 Local Feasibility Delta
For edge $(i, j) \in E$:
- If $x_i = 0 \to 1$:
  - If $x_j = 0$, edge $(i, j)$ remains satisfied ($0 + 1 \le 1$).
  - If $x_j = 1$, edge $(i, j)$ becomes **violated** ($1 + 1 = 2 > 1$).
- If $x_i = 1 \to 0$:
  - If $x_j = 1$, edge $(i, j)$ transitions from violated to satisfied.
  - If $x_j = 0$, edge $(i, j)$ transitions from satisfied to satisfied with slack.

### 5.2 Feasible Neighborhood Moves
A move $u_i$ from a feasible state $\mathbf{x} \in \mathcal{F}$ is **strictly feasible** if:

$$\mathbf{x} \oplus \mathbf{e}_i \in \mathcal{F}$$

For MIS:
- Adding vertex $i$ ($0 \to 1$) is strictly feasible iff $\forall j \in \mathcal{N}(i), x_j = 0$.
- Removing vertex $i$ ($1 \to 0$) is always feasible (preserves $\mathcal{F}$).

---

## 6. Concurrent Update Semantics & Conflict Model

When multiple Processing Elements (PEs) evaluate updates $\{u_{i_1}, u_{i_2}, \dots, u_{i_k}\}$ concurrently:

### 6.1 State Transition Commutativity
A batch of updates $B = \{u_{i_1}, \dots, u_{i_k}\}$ applied to state $\mathbf{x}$ produces:

$$\mathbf{x}' = \mathbf{x} \oplus \sum_{r=1}^k \mathbf{e}_{i_r}$$

### 6.2 Strict Feasibility Invariant
The batch $B$ is **valid and safe** if and only if:

$$\mathbf{x} \in \mathcal{F} \implies \mathbf{x}' \in \mathcal{F}$$

### 6.3 Conflict Condition for MIS
Two candidate updates $u_i$ and $u_j$ (both proposing $0 \to 1$) **conflict** if and only if:

$$(i, j) \in E$$

If both are applied simultaneously from a state where $x_i = 0$ and $x_j = 0$, then:

$$x'_i + x'_j = 1 + 1 = 2 > 1 \implies \mathbf{x}' \notin \mathcal{F}$$

Even though individually $u_i$ preserves feasibility (if $x_k = 0 \; \forall k \in \mathcal{N}(i)$) and $u_j$ preserves feasibility (if $x_l = 0 \; \forall l \in \mathcal{N}(j)$), their joint application creates a catastrophic violation of constraint $g_{i,j}$.

### 6.4 Safe Concurrent Batch Rule
A set of candidate additions $B \subseteq \{i \in V \mid x_i = 0\}$ is safe for concurrent application if and only if:

$$\forall i, j \in B, \; (i, j) \notin E \quad (B \text{ is an independent set in } G)$$

---

## 7. Execution Assumptions
1. **Discrete Synchronous Phases:** Time in the OPU simulator is partitioned into evaluation, conflict analysis, scheduling, dispatch, commit, and propagation phases.
2. **Correctness Invariant:** Under no circumstance may an unverified or conflicting batch commit to the authoritative state vector.
3. **Traceability:** Every state transition must record its batch composition, conflict analysis verdict, and verifier confirmation.
