# ARKWELL OPU: CUDA Environment Verification & Hardware Audit Report

**Document ID:** REP-ABS-V1-CUDA-01  
**Project:** ARKWELL (Optimization Processing Unit)  
**Milestone:** Milestone 2  
**Date:** 2026-11-23  
**Status:** Audited — Verified CPU Fallback Operational  

---

## 1. Executive Summary

As mandated by Section 6 and Section 9 of the Arkwell Milestone 2 specification, a comprehensive investigation was conducted into the host hardware, GPU subsystem, NVIDIA driver, CUDA driver capabilities, and PyTorch runtime installation.

**Key Findings:**
1. **Physical Hardware:** An **NVIDIA GeForce RTX 4050 Laptop GPU** (6,141 MiB dedicated VRAM, 75W TGP) is present and active.
2. **Driver Capabilities:** NVIDIA Display Driver **591.66** (WDDM 3.2) is installed, which supports CUDA runtime APIs up to **CUDA 13.1** and backward-compatible versions (CUDA 12.8, 12.6, 12.4, 12.1, 11.8).
3. **PyTorch Installation Status:** The active Python 3.12.4 runtime contains `torch==2.8.0` compiled as a **CPU-only** binary (`torch.cuda.is_available() == False`).
4. **PyTorch CUDA Wheel Compatibility:** Binary wheels matching this environment exist on the PyTorch index (specifically `torch==2.5.1+cu124` or `torch==2.6.0+cu124` for `cp312-win_amd64`). The binary wheel is approximately 2.4 GB in size.
5. **CPU Fallback Integrity:** The PyTorch tensor engine in `arkwell/backends/gpu/engine.py` cleanly detects `torch.cuda.is_available() == False` and engages the CPU tensor backend. Zero crashes or unhandled exceptions occurred, and mathematical parity was verified against the CPU reference engine.
6. **Benchmark Integrity Rule:** Pursuant to strict research guidelines, **all benchmark data reported in ABS V1.0 Milestones 1 and 2 are explicitly designated as CPU-based algorithmic simulations**, and no CPU fallback measurements are represented as GPU hardware acceleration benchmarks.

---

## 2. Hardware Inspection Telemetry

Direct telemetry queried via `nvidia-smi` and system introspection:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.66                 Driver Version: 591.66         CUDA Version: 13.1     |
| GPU  Name: NVIDIA GeForce RTX 4050 Laptop GPU (AD107M, Ada Lovelace)                    |
| VRAM: 6,141 MiB Total (approx. 2,000 MiB utilized by display compositor/active apps)   |
| Bus Type: PCIe Gen4 x8 | Compute Mode: Default | Architecture: CC 8.9                   |
+-----------------------------------------------------------------------------------------+
```

System Memory and CPU Telemetry:
* **Host CPU:** 12th Gen Intel Core (8 Physical Cores, 12 Logical Processors)
* **Host RAM:** 23.69 GB Total, 6.59 GB Available at test start
* **Storage:** 40.01 GB Free on C:\ drive, 76.05 GB Free on D:\ drive

---

## 3. PyTorch & CUDA Runtime Matrix

| Metric | Measured Value | Analysis |
| :--- | :--- | :--- |
| `sys.version` | `3.12.4 [MSC v.1940 64 bit (AMD64)]` | Supported Python version |
| `torch.__version__` | `2.8.0` | Standard PyPI CPU-only wheel |
| `torch.cuda.is_available()` | `False` | No compiled CUDA kernels in binary |
| `torch.cuda.device_count()` | `0` | PyTorch sees 0 CUDA devices |
| Standalone CUDA Toolkit (`nvcc`)| Not on system PATH | No standalone developer toolkit |

### Remote Wheel Availability
Inspection of `https://download.pytorch.org/whl/torch/` confirmed the availability of compatible CUDA wheels for `cp312-win_amd64`:
- `torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl` (2.4 GB)
- `torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl` (2.5 GB)

Bandwidth probing measured a transfer rate of 2.61 MB/s, indicating a complete package installation would require ~15 minutes of uninterrupted network download. As directed by Section 2 and Section 6 ("Do not blindly reinstall dependencies"), the existing environment was preserved and CPU fallback thoroughly validated.

---

## 4. Backend Architecture & Fallback Behavior

The Arkwell OPU simulator maintains strict separation between the algorithmic OPU state machine and the physical execution backend.

```
                      +-----------------------------+
                      |       OPU Simulator         |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
          +----------v----------+         +----------v----------+
          |     CPUEngine       |         |     GPUEngine       |
          |  (NumPy / C-arrays) |         |  (PyTorch Tensors)  |
          +----------+----------+         +----------+----------+
                     |                               |
                     |                   +-----------+-----------+
                     |                   |                       |
                     |         +---------v--------+    +---------v--------+
                     |         | CUDA Available?  |    | CUDA Available?  |
                     |         |      YES         |    |       NO         |
                     |         +---------+--------+    +---------+--------+
                     |                   |                       |
                     |         +---------v--------+    +---------v--------+
                     |         |  Device: cuda:0  |    |   Device: cpu    |
                     |         | (RTX 4050 VRAM)  |    | (Tensor Fallback)|
                     |         +------------------+    +------------------+
```

When `GPUEngine` runs in CPU fallback:
1. `self.device` evaluates to `torch.device("cpu")`.
2. Tensor operations (`index_add_`, `scatter`, logical mask operations) execute on the host CPU using vectorized SIMD kernels (AVX2).
3. `torch.cuda.synchronize()` degrades gracefully to a no-op.
4. Peak memory reporting returns `0` bytes rather than fabricating non-existent VRAM allocations.
5. Feasibility evaluation and state vector updates produce bit-for-bit identical results to `CPUEngine`.

---

## 5. Conclusions & Next Steps

1. The hardware environment is fully characterized and logged.
2. The CPU fallback mechanism is operational, stable, and verified by automated unit tests (`test_09_gpu_backend_correctness_and_fallback`).
3. For Milestone 2 performance scaling, CPU algorithmic optimizations (e.g. Sparse Adjacency Conflict Detection) provide the highest immediate performance leverage.
