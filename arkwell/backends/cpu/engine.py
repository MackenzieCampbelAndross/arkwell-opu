"""CPU execution engine for local search and constraint tracking."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem


@dataclass
class CPUEngineState:
    """Fast cache-aligned state for CPU local search."""
    state_vector: np.ndarray  # Shape (N,), int32: x_i in {0, 1}
    active_neighbor_counts: np.ndarray  # Shape (N,), int32: number of active neighbors
    objective_value: int


class CPUEngine:
    """CPU backend for OPU local search."""

    def __init__(self, compiled: OPUCompiledProblem):
        self.compiled = compiled
        self.n = compiled.num_variables
        self.adj = compiled.adjacency_list

    def initialize_state(self, initial_state: np.ndarray) -> CPUEngineState:
        """Initializes state and computes active neighbor counts."""
        state = np.asarray(initial_state, dtype=np.int32).copy()
        active_counts = np.zeros(self.n, dtype=np.int32)

        for u in range(self.n):
            if state[u] == 1:
                nbrs = self.adj[u]
                if len(nbrs) > 0:
                    active_counts[nbrs] += 1

        obj = int(np.sum(state))
        return CPUEngineState(
            state_vector=state,
            active_neighbor_counts=active_counts,
            objective_value=obj,
        )

    def is_addition_feasible(self, state: CPUEngineState, var_idx: int) -> bool:
        """Adding vertex var_idx (0 -> 1) is feasible iff active_neighbor_counts == 0."""
        return state.state_vector[var_idx] == 0 and state.active_neighbor_counts[var_idx] == 0

    def apply_flip(self, state: CPUEngineState, var_idx: int) -> None:
        """In-place updates state vector, objective, and neighbor counts."""
        curr = state.state_vector[var_idx]
        nbrs = self.adj[var_idx]

        if curr == 0:
            # 0 -> 1
            state.state_vector[var_idx] = 1
            state.objective_value += 1
            if len(nbrs) > 0:
                state.active_neighbor_counts[nbrs] += 1
        else:
            # 1 -> 0
            state.state_vector[var_idx] = 0
            state.objective_value -= 1
            if len(nbrs) > 0:
                state.active_neighbor_counts[nbrs] -= 1

    def apply_batch_flips(self, state: CPUEngineState, flip_indices: List[int]) -> None:
        """Applies multiple flips safely."""
        for idx in flip_indices:
            self.apply_flip(state, idx)
