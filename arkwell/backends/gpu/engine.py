"""GPU execution backend for OPU local search using PyTorch."""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem


@dataclass
class GPUEngineResult:
    """Metrics returned by GPU backend execution."""
    device: str
    cuda_available: bool
    peak_memory_bytes: int
    execution_time_seconds: float
    final_objective: int
    is_feasible: bool


class GPUEngine:
    """
    PyTorch-accelerated backend for parallel state evaluation and update.
    Supports CUDA with automatic, transparent CPU fallback when CUDA is not present.
    """

    def __init__(self, compiled: OPUCompiledProblem, device_str: Optional[str] = None):
        self.compiled = compiled
        self.n = compiled.num_variables
        self.m = compiled.num_constraints

        # Device determination
        if device_str is not None:
            self.device = torch.device(device_str)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.is_cuda = self.device.type == "cuda"

        # Edge index for constraint evaluations
        if self.m > 0:
            edges = compiled.constraint_graph.constraint_edges
            # Undirected edge index (2, 2*M) for symmetric neighbor scatter
            u = np.concatenate([edges[:, 0], edges[:, 1]])
            v = np.concatenate([edges[:, 1], edges[:, 0]])
            self.src_nodes = torch.tensor(u, dtype=torch.int64, device=self.device)
            self.dst_nodes = torch.tensor(v, dtype=torch.int64, device=self.device)
        else:
            self.src_nodes = torch.empty(0, dtype=torch.int64, device=self.device)
            self.dst_nodes = torch.empty(0, dtype=torch.int64, device=self.device)

    def synchronize(self) -> None:
        """Synchronizes GPU device pipeline if on CUDA."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def compute_active_neighbors_parallel(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes active neighbor counts for all vertices simultaneously on device.
        active_counts[v] = sum_{u in N(v)} state[u]
        """
        active_counts = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        if self.src_nodes.numel() > 0:
            active_src = state_tensor[self.src_nodes]
            # Accumulate active src into dst nodes
            active_counts.index_add_(0, self.dst_nodes, active_src.to(torch.int32))
        return active_counts

    def evaluate_state_feasibility(self, state_tensor: torch.Tensor) -> Tuple[bool, int]:
        """Parallel check of all edge constraints on device."""
        if self.m == 0:
            return True, int(state_tensor.sum().item())
        edges = self.compiled.constraint_graph.constraint_edges
        u_idx = torch.tensor(edges[:, 0], dtype=torch.int64, device=self.device)
        v_idx = torch.tensor(edges[:, 1], dtype=torch.int64, device=self.device)
        edge_sum = state_tensor[u_idx] + state_tensor[v_idx]
        has_violations = bool((edge_sum > 1).any().item())
        obj = int(state_tensor.sum().item())
        return (not has_violations), obj

    def run_parallel_greedy_sweep(
        self,
        initial_state: np.ndarray,
        candidate_order: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, GPUEngineResult]:
        """
        Executes parallel candidate evaluation on GPU with careful timing and memory tracking.
        """
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.synchronize()

        start_time = time.perf_counter()

        # Transfer state to device once
        state_t = torch.tensor(initial_state, dtype=torch.int32, device=self.device)

        if candidate_order is None:
            order = list(range(self.n))
        else:
            order = list(candidate_order)

        # Incrementally update active counts
        active_counts = self.compute_active_neighbors_parallel(state_t)

        for v in order:
            if state_t[v] == 0 and active_counts[v] == 0:
                state_t[v] = 1
                # Update neighbors
                nbrs = self.compiled.adjacency_list[v]
                if len(nbrs) > 0:
                    nbr_tensor = torch.tensor(nbrs, dtype=torch.int64, device=self.device)
                    active_counts.index_add_(0, nbr_tensor, torch.ones_like(nbr_tensor, dtype=torch.int32))

        self.synchronize()
        end_time = time.perf_counter()
        exec_time = end_time - start_time

        peak_mem = 0
        if self.is_cuda:
            peak_mem = torch.cuda.max_memory_allocated(self.device)

        final_state_cpu = state_t.cpu().numpy()
        is_feas, obj = self.evaluate_state_feasibility(state_t)

        result = GPUEngineResult(
            device=str(self.device),
            cuda_available=self.is_cuda,
            peak_memory_bytes=peak_mem,
            execution_time_seconds=exec_time,
            final_objective=obj,
            is_feasible=is_feas,
        )
        return final_state_cpu, result
