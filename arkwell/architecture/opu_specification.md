# ARKWELL OPU: Architectural Specification

## 1. Architectural Overview
The **Optimization Processing Unit (OPU)** is a specialized computing architecture dedicated to discrete and combinatorial optimization problems. Unlike general-purpose CPUs (optimized for low-latency serial pipelines) or GPUs (optimized for uniform SIMD/SIMT dense tensor throughput), the OPU is architected around:

1. **Hardware-Native Dependency Graphs:** Explicit compilation of objective and constraint structures into physical dependency graphs.
2. **Dynamic Conflict Analyzer:** Hardware logic to detect read/write, objective, and constraint coupling across candidate local search updates.
3. **Safe Concurrent Scheduler:** Conflict-free batch generation unit that dispatches maximal safe concurrent sets to parallel Processing Elements (PEs).
4. **Selective Propagation Interconnect:** State multicast fabric that forwards state updates only to PEs subscribed to modified constraints.

```
                      +-----------------------------+
                      |     Problem Compiler        |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
        +------------v------------+     +------------v------------+
        | Objective Dependency    |     | Constraint Dependency   |
        | Graph Engine            |     | Graph Engine            |
        +------------+------------+     +------------+------------+
                     |                               |
                     +---------------+---------------+
                                     |
                      +--------------v--------------+
                      | Hardware Conflict Analyzer  |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      |   Safe Concurrent Scheduler |
                      +--------------+--------------+
                                     |
                 +-------------------+-------------------+
                 |                   |                   |
        +--------v-------+  +--------v-------+  +--------v-------+
        | Processing     |  | Processing     |  | Processing     |
        | Element (PE 0) |  | Element (PE 1) |  | Element (PE k) |
        +--------+-------+  +--------+-------+  +--------+-------+
                 |                   |                   |
                 +-------------------+-------------------+
                                     |
                      +--------------v--------------+
                      | Selective State Interconnect|
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      | Authoritative State Manager |
                      +-----------------------------+
```

---

## 2. Subsystem Definitions

### 2.1 Compiler
Transforms the mathematical formulation of an optimization problem (e.g. MIS, Max-SAT, QUBO, Integer Programming) into:
- The Variable Allocation Table (VAT).
- The Objective Dependency Graph ($\mathcal{G}_{\text{obj}}$).
- The Constraint Dependency Graph ($\mathcal{G}_{\text{con}}$).

### 2.2 Objective & Constraint Dependency Graphs
- **Objective Graph ($\mathcal{G}_{\text{obj}}$):** Direct hypergraph where hyperedges connect variables contributing to common nonlinear or coupled objective terms. For linear objectives (such as MIS), each node is an isolated component.
- **Constraint Graph ($\mathcal{G}_{\text{con}}$):** Bipartite graph between variables $V$ and constraints $C$. An edge $(i, k)$ exists if variable $x_i$ appears in constraint $g_k(\mathbf{x})$.

### 2.3 Conflict Analyzer
Computes pairwise and higher-order incompatibilities between candidate updates. It determines whether concurrent execution of $\{u_a, u_b\}$ will cause race conditions, inconsistent constraint reads, or joint feasibility violations.

### 2.4 Scheduler
Partitions active candidate updates into discrete execution groups (batches):
$$\mathcal{B} = \{B_1, B_2, \dots, B_m\}$$
Such that for every batch $B_r$, all updates in $B_r$ can commit simultaneously without synchronization violation or feasibility corruption.

### 2.5 Processing Elements (PE Array)
Parallel SIMD/MIMD execution units. Each PE:
- Maintains a local view of assigned variables and incident constraints.
- Computes candidate move evaluations ($\Delta f_i, \Delta g_k$).
- Proposes candidate updates to the Scheduler.

### 2.6 State Manager & Verifier
- **State Manager:** Holds the authoritative configuration bitvector $\mathbf{x} \in \{0, 1\}^N$.
- **Verifier:** An independent hardware sanity checker that verifies feasibility and objective delta before retiring state transitions to the architectural register file.

---

## 3. Simulator Target & Research Purpose
The ABS V1.0 software simulator models this pipeline cycle-accurately in terms of algorithmic transitions, message passing counts, synchronization barriers, and memory traffic, evaluating whether the architectural hypothesis holds before silicon synthesis.
