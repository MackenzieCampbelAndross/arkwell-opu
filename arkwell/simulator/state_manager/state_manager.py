"""Authoritative State Manager for the OPU architecture."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class StateTransitionRecord:
    """Historical record of an applied state update."""
    step: int
    batch_size: int
    modified_variables: List[int]
    delta_objective: int
    resulting_objective: int


class StateManager:
    """
    Authoritative register file holding the architectural state vector x in {0, 1}^N.
    Enforces atomic commit semantics and maintains state audit history.
    """

    def __init__(self, num_variables: int, initial_state: Optional[np.ndarray] = None):
        self.num_variables = num_variables
        if initial_state is not None:
            self.state = np.asarray(initial_state, dtype=np.int32).copy()
        else:
            self.state = np.zeros(num_variables, dtype=np.int32)
        
        self.step_counter = 0
        self.history: List[StateTransitionRecord] = []

    def get_state(self) -> np.ndarray:
        """Returns a read-only copy of the current state vector."""
        return self.state.copy()

    def commit_batch(self, flip_indices: List[int]) -> StateTransitionRecord:
        """
        Atomically commits a batch of bit flips to the state vector.
        """
        if len(flip_indices) == 0:
            return StateTransitionRecord(
                step=self.step_counter,
                batch_size=0,
                modified_variables=[],
                delta_objective=0,
                resulting_objective=int(np.sum(self.state)),
            )

        # Ensure unique variables in batch
        unique_indices = list(set(flip_indices))
        if len(unique_indices) != len(flip_indices):
            raise ValueError("Duplicate variable flip detected in single commit batch!")

        old_obj = int(np.sum(self.state))
        for idx in unique_indices:
            self.state[idx] = 1 - self.state[idx]

        new_obj = int(np.sum(self.state))
        delta_obj = new_obj - old_obj

        self.step_counter += 1
        record = StateTransitionRecord(
            step=self.step_counter,
            batch_size=len(unique_indices),
            modified_variables=unique_indices,
            delta_objective=delta_obj,
            resulting_objective=new_obj,
        )
        self.history.append(record)
        return record
