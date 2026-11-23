# ARKWELL ABS V1.0 — BENCHMARK B02: CONFLICT SCHEDULING EFFICIENCY REPORT

**Document ID:** REP-ABS-V1-B02-01  
**Project:** ARKWELL (Optimization Processing Unit)  
**Milestone:** Milestone 2  
**Date:** 2026-11-23  
**Status:** Complete — Empirical Benchmark Evaluation  

---

## 1. Executive Summary

Benchmark **B02 (Conflict Scheduling Efficiency)** investigates the computational complexity, graph density, degree distribution, batch partitioning, and software simulator bottlenecks associated with the OPU Conflict Analyzer and Safe Concurrent Scheduler.

### Key Empirical Findings
1. **Algorithmic Speedup:** The newly introduced `OptimizedConflictAnalyzer` (sparse adjacency neighbor intersection + vectorized intra-PE collision) achieves an average **27.67x speedup** (up to **44.93x**) over the reference $O(K^2)$ analyzer while preserving bit-for-bit mathematical equivalence across all test matrices.
2. **Conflict Graph Density:** Conflict density remains remarkably stable across graph sizes ($N=512$ to $N=1024$), averaging **$D \approx 0.264$** on benchmark instances. Average candidate conflict degree scales linearly with problem size ($\bar{d} = 134.8$ at $N=512$, $\bar{d} = 270.8$ at $N=1024$).
3. **Execution Group Partitioning:** The scheduler successfully packs active candidates into dense, safe concurrent batches, maintaining an average batch width of $8.1$ to $8.5$ concurrent moves.
4. **Hardware Processing Element Utilization:** PE utilization across scheduled execution groups consistently exceeds **$97.9\%$ to $98.6\%$**, confirming that candidate partitioning effectively keeps parallel execution pipelines saturated.
5. **Software vs. Silicon Bottleneck:** At $N=1024$, conflict construction was reduced from $295.80\text{ ms}$ down to $8.41\text{ ms}$. In a physical OPU, conflict graph construction is mapped directly to spatial hardware crossbars, avoiding serial instruction fetch overhead.

---

## 2. Experimental Setup & Workload Matrix

* **Instance Generator:** Erdős-Rényi $G(N, p)$ with deterministic seeds (42, 101, 2024).
* **Problem Sizes ($N$):** 512, 1024 variables.
* **Graph Densities ($d$):** 0.02, 0.10, 0.30.
* **Processing Elements ($P$):** 4, 8, 16 PEs.
* **Candidate Pool:** $K = N$ active candidate additions evaluated at Round 1.
* **Analyzers Evaluated:**
  - `ReferenceConflictAnalyzer`: $O(K^2)$ exhaustive pair check with Python hash sets and dictionary tracking.
  - `OptimizedConflictAnalyzer`: $O(K \cdot \bar{d})$ sparse neighbor indexing with direct integer lookup arrays and vectorized NumPy bitwise masks.

---

## 3. Empirical Results

### 3.1 Conflict Graph Construction Performance (Ref vs. Opt)

| Size ($N$) | Density ($d$) | PEs | Conflicts ($|E_{\text{conf}}|$) | Conflict Density | Avg Degree ($\bar{d}$) | Ref Time (ms) | Opt Time (ms) | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 0.02 | 4 | 33,627 | 0.2571 | 131.4 | 55.77 | 3.32 | **16.8x** |
| **512** | 0.02 | 16 | 9,075 | 0.0694 | 35.4 | 39.46 | 1.84 | **21.4x** |
| **512** | 0.10 | 4 | 44,570 | 0.3407 | 174.1 | 70.36 | 3.44 | **20.5x** |
| **512** | 0.10 | 16 | 20,018 | 0.1530 | 78.2 | 53.64 | 2.12 | **25.3x** |
| **512** | 0.30 | 4 | 70,736 | 0.5407 | 276.3 | 108.97 | 5.25 | **20.8x** |
| **512** | 0.30 | 16 | 46,184 | 0.3530 | 180.4 | 89.28 | 3.27 | **27.3x** |
| **1024** | 0.02 | 4 | 133,969 | 0.2558 | 261.7 | 239.51 | 8.24 | **29.1x** |
| **1024** | 0.02 | 16 | 35,665 | 0.0681 | 69.7 | 170.83 | 4.67 | **36.6x** |
| **1024** | 0.10 | 4 | 177,758 | 0.3394 | 347.2 | 313.25 | 9.07 | **34.5x** |
| **1024** | 0.10 | 16 | 79,454 | 0.1517 | 155.2 | 235.10 | 5.86 | **40.1x** |
| **1024** | 0.30 | 4 | 282,427 | 0.5392 | 551.6 | 486.29 | 14.88 | **32.7x** |
| **1024** | 0.30 | 16 | 184,123 | 0.3515 | 359.6 | 397.66 | 8.85 | **44.9x** |

**Summary Across All 54 Configurations:**
- Mean Reference Construction Time: **181.21 ms**
- Mean Optimized Construction Time: **5.83 ms**
- **Average Algorithmic Speedup: 27.67x (Maximum: 44.93x)**

---

### 3.2 Scheduling Quality & Processing Element Utilization

| Size ($N$) | Mean Batches (Groups) | Mean Batch Width | Max Batch Width | Scheduling Latency (ms) | PE Utilization |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 80.0 | 8.1 | 9.1 | 3.00 ms | **97.9%** |
| **1024** | 155.3 | 8.5 | 9.3 | 10.58 ms | **98.6%** |

*Analysis:* The bitmask-accelerated greedy independent set scheduler partitions up to 1024 candidates into conflict-free batches in ~10 ms. In each primary cycle, PEs achieve >97% utilization, meaning parallel functional units are almost never left idling due to scheduling fragmentation.

---

## 4. Architectural Conclusions

1. **Bottleneck Resolution:** The previous $O(K^2)$ Python loop bottleneck has been completely mitigated by sparse adjacency indexing. Conflict graph construction time dropped from over $480\text{ ms}$ to under $15\text{ ms}$ at $N=1024$.
2. **Equivalence Preserved:** Regression test `test_conflict_model_equivalence` proved 100% equivalence between optimized and reference outputs, with zero discrepancies in conflict edges or types.
3. **Hardware Crossbar Feasibility:** Because conflict analysis can be formulated as sparse neighbor bitwise masking, dedicated OPU hardware can implement this via bit-parallel content-addressable memory (CAM) or sparse switch matrices in sub-microsecond latency.
