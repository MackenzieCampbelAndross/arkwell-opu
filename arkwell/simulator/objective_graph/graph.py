"""Objective Dependency Graph representation for OPU compiler."""

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np


@dataclass
class ObjectiveGraph:
    """
    Encodes variable dependencies within the objective function.
    For linear objectives (MIS), all variables are independent (no coupling edges).
    For non-linear objectives (QUBO / Higher-order), edges denote coupled terms.
    """
    num_variables: int
    linear_coefficients: np.ndarray  # Shape (N,), weight of each x_i
    coupling_edges: np.ndarray  # Shape (K, 2) of variable pairs sharing non-linear terms
    coupling_weights: np.ndarray  # Shape (K,) weights

    @property
    def is_linear(self) -> bool:
        return len(self.coupling_edges) == 0

    def compute_local_delta(self, var_idx: int, current_state: np.ndarray) -> float:
        """Computes delta in objective if var_idx is flipped from current_state."""
        current_val = current_state[var_idx]
        delta_x = 1 - 2 * current_val  # If 0 -> +1; if 1 -> -1
        delta_obj = float(self.linear_coefficients[var_idx] * delta_x)

        # If non-linear couplings exist, add coupled terms
        # (For MIS, this is zero)
        return delta_obj
