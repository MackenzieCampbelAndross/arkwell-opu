# ARKWELL — Optimization Processing Unit (OPU) Architecture

Arkwell is a research project exploring a digital, massively parallel, constraint-aware discrete optimization computing architecture.

## Primary Research Hypotheses
1. A compiler that generates objective and constraint dependency graphs, constructs a hardware-aware conflict graph, and schedules safe concurrent local search updates can reduce synchronization and communication overhead compared to naive parallelization.
2. A selective constraint propagation mechanism can further minimize interconnect traffic and memory invalidations.

> **Status:** These are unverified research hypotheses under ongoing experimental investigation in the Arkwell Benchmark Suite (ABS V1.0).

---

## Repository Organization

```
arkwell/
    architecture/            # Architectural specifications and execution semantics
    simulator/               # Core OPU cycle & state simulator
        compiler/            # Mathematical formulation -> graph compiler
        objective_graph/     # Objective dependency graph
        constraint_graph/    # Bipartite constraint hypergraph
        conflict_analyzer/   # Hardware conflict analyzer
        scheduler/           # Safe concurrent update scheduler
        processing_elements/ # Simulated PE array
        state_manager/       # Authoritative state register file
        verifier/            # Hardware-independent solution verifier
    backends/                # Execution backends
        cpu/                 # High-performance NumPy/SciPy reference engine
        gpu/                 # PyTorch CUDA backend with CPU fallback
    benchmarks/              # Arkwell Benchmark Suite (ABS V1.0)
        B01_correctness/     # Exhaustive & sanity verification tests
        B02_conflict_scheduling/
        B03_constraint_propagation/
        B04_optimization_performance/
        B05_scalability/
        B06_parallel_efficiency/
        B07_resource_efficiency/
    instances/               # Benchmark problem instance generators
        maximum_independent_set/
    experiments/             # Automated experiments and sweeps
        configurations/      # YAML configuration files
        raw_results/         # Structured JSON & CSV data
        logs/                # Run logs
    analysis/                # Result analysis and visualization scripts
    reports/                 # Formal verification and benchmark reports
    patent_evidence/         # Technical disclosures and prior art references
```

---

## Requirements and Setup
- Python 3.12 (or 3.11+)
- NumPy, SciPy
- PyTorch (CUDA supported with CPU fallback)
- Pytest, PyYAML, Matplotlib

Run the test suite:
```bash
py -3.12 -m pytest arkwell/benchmarks/B01_correctness -v
```
