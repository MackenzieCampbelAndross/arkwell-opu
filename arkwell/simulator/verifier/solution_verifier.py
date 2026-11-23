"""Independent correctness and feasibility verifier for optimization solutions."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from arkwell.instances.maximum_independent_set.generator import MISInstance


@dataclass
class VerificationResult:
    """Rigorous verification report produced independently of any solver."""
    is_feasible: bool
    objective_value: int
    num_violations: int
    violated_constraints: List[Tuple[int, int]]
    state_vector: np.ndarray

    def assert_valid(self) -> None:
        if not self.is_feasible:
            raise ValueError(
                f"Verification failed: {self.num_violations} constraint violations detected! "
                f"First 5 violations: {self.violated_constraints[:5]}"
            )


class IndependentVerifier:
    """Hardware-independent verification engine."""

    @staticmethod
    def verify_mis(instance: MISInstance, state: np.ndarray) -> VerificationResult:
        """
        Independently checks all constraints and evaluates objective.
        Uses raw instance edge list; does not rely on any solver-internal data.
        """
        state = np.asarray(state, dtype=np.int32)
        if state.ndim != 1 or len(state) != instance.num_vertices:
            raise ValueError(f"State dimension mismatch: expected ({instance.num_vertices},), got {state.shape}")

        # Check binary values
        if not np.all((state == 0) | (state == 1)):
            raise ValueError("State vector contains non-binary values!")

        # Objective is sum of selected vertices
        obj_val = int(np.sum(state))

        # Check constraint feasibility: for each edge (u, v), state[u] + state[v] <= 1
        violations: List[Tuple[int, int]] = []
        if instance.num_edges > 0:
            u_vals = state[instance.edges[:, 0]]
            v_vals = state[instance.edges[:, 1]]
            edge_sums = u_vals + v_vals
            violated_mask = edge_sums > 1
            if np.any(violated_mask):
                violated_indices = np.where(violated_mask)[0]
                for idx in violated_indices:
                    u, v = instance.edges[idx]
                    violations.append((int(u), int(v)))

        return VerificationResult(
            is_feasible=(len(violations) == 0),
            objective_value=obj_val,
            num_violations=len(violations),
            violated_constraints=violations,
            state_vector=state.copy(),
        )

    @staticmethod
    def verify_batch_transition(
        instance: MISInstance,
        initial_state: np.ndarray,
        batch_updates: List[int],  # List of vertex indices being flipped
        final_state: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Verifies that a batch state transition was atomic, correct, and preserved feasibility.
        """
        expected_state = initial_state.copy()
        for v in batch_updates:
            expected_state[v] = 1 - expected_state[v]

        if not np.array_equal(final_state, expected_state):
            return False, "Final state does not match bit-flip expectation"

        # Check final state feasibility
        res = IndependentVerifier.verify_mis(instance, final_state)
        if not res.is_feasible:
            return False, f"Batch transition created {res.num_violations} feasibility violations"

        return True, "Batch transition verified successfully"


class ExhaustiveVerifier:
    """Exact ground-truth solver for small instances (N <= 20) via exhaustive state enumeration."""

    @staticmethod
    def solve_mis_exact(instance: MISInstance) -> Tuple[int, np.ndarray]:
        """
        Exhaustively tests all 2^N state configurations to find the provable global maximum independent set.
        """
        n = instance.num_vertices
        if n > 22:
            raise ValueError(f"Exhaustive verification is computationally prohibitive for N={n} > 22")

        max_obj = -1
        best_state = np.zeros(n, dtype=np.int32)
        total_states = 1 << n

        for mask in range(total_states):
            state = np.array([(mask >> bit) & 1 for bit in range(n)], dtype=np.int32)
            # Fast feasibility check
            if instance.num_edges > 0:
                u_vals = state[instance.edges[:, 0]]
                v_vals = state[instance.edges[:, 1]]
                if np.any(u_vals + v_vals > 1):
                    continue

            obj = int(np.sum(state))
            if obj > max_obj:
                max_obj = obj
                best_state = state

        return max_obj, best_state
