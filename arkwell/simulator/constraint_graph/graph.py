"""Constraint Dependency Graph representation for OPU compiler."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ConstraintGraph:
    """
    Bipartite representation between Variables and Constraints.
    In MIS:
      Variables: 0 .. N-1
      Constraints: 0 .. M-1, each is (u, v) representing x_u + x_v <= 1.
    """
    num_variables: int
    num_constraints: int
    # Constraint definitions: constraint_edges[k] = (u, v)
    constraint_edges: np.ndarray  # Shape (M, 2), int64
    # Bipartite mapping: var_to_constraints[i] = list of constraint indices incident to variable i
    var_to_constraints: List[np.ndarray]

    def evaluate_all_slacks(self, state: np.ndarray) -> np.ndarray:
        """
        Computes slack s_k = 1 - (x_u + x_v) for all constraints k in [0, M-1].
        Feasible iff s_k >= 0 for all k.
        """
        if self.num_constraints == 0:
            return np.empty(0, dtype=np.int32)
        u_vals = state[self.constraint_edges[:, 0]]
        v_vals = state[self.constraint_edges[:, 1]]
        slacks = 1 - (u_vals + v_vals)
        return slacks

    def get_incident_constraints(self, var_idx: int) -> np.ndarray:
        return self.var_to_constraints[var_idx]
