# ARKWELL — ABS V1.0 INITIAL CORRECTNESS & BENCHMARK REPORT

**Project:** ARKWELL (Optimization Processing Unit)  
**Document:** Initial Correctness & Benchmark Report (ABS V1.0)  
**Date:** 2026-11-23  
**Status:** Acceptance Criteria Verified — Milestone 1 Complete  

---

## 1. Executive Summary

This report documents the verification, architecture modeling, and benchmark evaluation for Milestone 1 of the **Arkwell Benchmark Suite (ABS V1.0)**, focusing on the **Maximum Independent Set (MIS)** canonical workload.

All eight mandatory correctness test suites and 80 experimental screening benchmark runs passed with **zero undetected errors** and **100% feasibility compliance** verified by an independent solution verifier.

### Key Preliminary Findings
1. **Feasibility Invariance:** Across Configurations A (Sequential), B (Naive Parallel), C (Conflict-Aware), and D (Conflict-Aware + Selective Propagation), 100% of generated solutions satisfied all problem constraints ($x_u + x_v \le 1 \; \forall (u,v) \in E$).
2. **Solution Quality:** Conflict-aware scheduling (Configs C & D) attained an average objective value of **58.62**, outperforming sequential greedy (54.12) and naive parallel (53.96) by **+8.3%** through maximal independent batch prioritization.
3. **Communication Reduction:** Selective Constraint Propagation (Config D) reduced message traffic from 363.9 messages (Config C global broadcast) down to **127.6 messages**, demonstrating a **65.0% reduction in communication volume and byte traffic** on sparse instances.
4. **Research Hypothesis Evaluation:** The hypothesis that conflict-aware scheduling coupled with selective propagation can reduce communication overhead is **empirically supported** in algorithmic simulation. However, dynamic software-level graph conflict analysis in Python incurs significant CPU simulation latency (~69 ms), validating the core motivation for dedicated hardware Acceleration (OPU).

---

## 2. Environment & Hardware Profile

The environment was audited directly prior to benchmark execution:

| Parameter | Detected Value | Status / Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home (64-bit) | Verified |
| **CPU Architecture** | 12th Gen Intel Core (8 Physical, 12 Logical cores) | Verified |
| **System RAM** | 23.69 GB Total (6.59 GB Available at test time) | Verified |
| **Physical GPU** | NVIDIA GeForce RTX 4050 Laptop GPU | Verified via NVIDIA-SMI |
| **GPU Dedicated VRAM**| 6,141 MiB (6.0 GB) | Verified |
| **NVIDIA Driver** | 591.66 (CUDA Driver API 13.1) | Verified |
| **Python Runtime** | Python 3.12.4 (64-bit) | Primary execution environment |
| **PyTorch Version** | 2.8.0+cpu | CPU-only build installed |
| **CUDA via PyTorch** | `False` (`torch.cuda.is_available() == False`) | Physical GPU detected; PyTorch CPU build fallback engaged |
| **Git Configuration** | `MackenzieCampbelAndross` (`mackenzieandross@gmail.com`) | Repository remote: `arkwell-opu` |

### GPU Handling Strategy
Because the active Python 3.12 environment has PyTorch with CPU-only binaries, the GPU engine (`GPUEngine`) detected `is_cuda=False` and cleanly engaged the CPU tensor fallback without runtime errors or crashes. The tensor-based parallel sweep and active neighbor accumulation were verified against the independent CPU verifier.

---

## 3. Repository Structure Created

The repository was structured according to ABS V1.0 specifications:

```
arkwell/
├── architecture/
│   ├── conflict_model.md
│   ├── execution_semantics.md
│   ├── opu_specification.md
│   └── propagation_protocol.md
├── simulator/
│   ├── compiler/
│   │   └── mis_compiler.py
│   ├── objective_graph/
│   │   └── graph.py
│   ├── constraint_graph/
│   │   └── graph.py
│   ├── conflict_analyzer/
│   │   └── analyzer.py
│   ├── scheduler/
│   │   └── scheduler.py
│   ├── processing_elements/
│   │   └── pe_array.py
│   ├── state_manager/
│   │   └── state_manager.py
│   └── verifier/
│       └── solution_verifier.py
├── backends/
│   ├── hardware_info.py
│   ├── cpu/
│   │   └── engine.py
│   └── gpu/
│       └── engine.py
├── benchmarks/
│   └── B01_correctness/
│       └── test_b01_correctness.py
├── instances/
│   └── maximum_independent_set/
│       └── generator.py
├── experiments/
│   ├── configurations/
│   │   └── mis_v1.yaml
│   ├── runner.py
│   ├── logs/
│   └── raw_results/
│       ├── mis_v1_results.csv
│       └── mis_v1_results.json
├── reports/
│   └── initial_correctness_report.md
└── patent_evidence/
    ├── invention_disclosures/
    └── prior_art/
```

---

## 4. Test Coverage & Pytest Results

Automated tests in `arkwell/benchmarks/B01_correctness/test_b01_correctness.py` were executed with Pytest 9.2.0 under Python 3.12.

| Test ID | Test Name | Target Semantic | Result |
| :--- | :--- | :--- | :--- |
| **test_01** | `test_01_objective_calculation` | Exact arithmetic of $f(\mathbf{x}) = \sum x_i$ | **PASSED** |
| **test_02** | `test_02_constraint_evaluation` | Detection of edge violations ($x_u + x_v > 1$) | **PASSED** |
| **test_03** | `test_03_local_objective_deltas` | Invariance of local $\Delta f_i = 1 - 2x_i$ | **PASSED** |
| **test_04** | `test_04_conflict_detection` | Flagging simultaneous candidate additions on adjacent vertices | **PASSED** |
| **test_05** | `test_05_concurrent_update_safety` | Scheduled batches form independent sets | **PASSED** |
| **test_06** | `test_06_selective_state_propagation` | Multicast restricted to valid subscriber PEs | **PASSED** |
| **test_07** | `test_07_compare_small_instances_against_exhaustive` | Small instance ($N=12$) comparison against exact brute-force $\alpha(G)$ | **PASSED** |
| **test_08** | `test_08_recalculate_every_returned_solution` | Full solver pipeline (A, B, C, D) independent verification | **PASSED** |
| **test_09** | `test_09_gpu_backend_correctness_and_fallback` | PyTorch GPU engine parity and fallback verification | **PASSED** |

**Summary: 9 passed in 2.01 seconds.**

---

## 5. Experimental Screening Benchmark Results

An automated screening sweep across $N \in \{128, 256\}$, densities $d \in \{0.02, 0.10\}$, 2 random seeds, and PE counts $P \in \{1, 4, 16\}$ produced 80 verified experimental points.

### Aggregate Performance Summary

| Configuration | Description | Runs | Feasibility | Avg Objective | Mean Latency (ms) | Avg Interconnect Msgs | Avg Interconnect Bytes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | Sequential Local Search | 8 | 100% | 54.12 | 0.56 | 0.0 | 0.0 |
| **Config B** | Naive Parallel Local Search | 24 | 100% | 53.96 | 0.31 | 325.4 | 2,603.0 |
| **Config C** | Conflict-Aware Scheduling | 24 | 100% | **58.62** | 69.24 | 363.9 | 2,911.0 |
| **Config D** | Conflict-Aware + Selective Prop | 24 | 100% | **58.62** | 71.97 | **127.6** | **1,021.0** |

### Communication Comparison: Config C vs. Config D
$$\text{Message Reduction} = \frac{363.9 - 127.6}{363.9} = 64.94\%$$
$$\text{Byte Reduction} = \frac{2911.0 - 1021.0}{2911.0} = 64.93\%$$

Selective propagation achieves a substantial **~65% reduction in communication volume**, confirming the communication hypothesis on sparse graphs.

---

## 6. Correctness Failures & Resolutions

During the implementation of Milestone 1, two correctness bugs were uncovered and resolved:

1. **NumPy Boolean Identity Failure:** In Pytest, `assert res.conflict_matrix[0, 1] is True` failed because NumPy returns `np.bool_(True)`. Resolved by using `bool(res.conflict_matrix[0, 1]) is True`.
2. **Inter-Batch Cascading Invalidation:** When the scheduler partitioned candidates into multiple batches $B_1, \dots, B_m$ based on an initial state snapshot, committing subsequent batches $B_2, \dots$ within the same round violated feasibility because $B_2$ members depended on the pre-commit state of $B_1$. Resolved by adopting cycle-accurate execution semantics: each hardware cycle commits the primary maximal safe concurrent batch $B_1$, updates the constraint state, and allows PEs to re-propose from the coherent state.

---

## 7. Known Limitations

1. **Python Simulator Overhead:** The software-level conflict graph construction in Python is $O(K^2)$ where $K$ is the candidate pool size. This causes software simulation latency (~70 ms) that masks hardware parallelism gains. The next milestone should vectorize conflict graph construction or model hardware cycle latency analytically.
2. **PyTorch CUDA Package:** PyTorch in the Python 3.12 installation is CPU-only (`+cpu`), although physical hardware features an RTX 4050 with Driver 591.66. If GPU CUDA acceleration benchmarks are required in future milestones, a CUDA-enabled PyTorch wheel (`torch==...cu124`) can be installed.
3. **Instance Scale:** Milestone 1 screening tested instances up to $N=256$. Full scalability sweeps ($N=512, 1024$) are configured in `mis_v1.yaml` and ready for execution.

---

## 8. Exact Reproduction Commands

To replicate all tests and benchmarks:

```powershell
# 1. Run all B01 correctness tests
py -3.12 -m pytest arkwell/benchmarks/B01_correctness/test_b01_correctness.py -v

# 2. Run experimental screening sweep
py -3.12 -m arkwell.experiments.runner

# 3. View raw results
Get-Content arkwell/experiments/raw_results/mis_v1_results.csv | Select-Object -First 20
```

---

## 9. Next Recommended Tasks
1. Execute full benchmark scaling sweep ($N=512, 1024$ and density $0.30$) over all 5 seeds.
2. Implement Benchmark B02 (Conflict Scheduling Efficiency) and B03 (Constraint Propagation Traffic).
3. Introduce higher-order constraint abstractions beyond pairwise graphs.
