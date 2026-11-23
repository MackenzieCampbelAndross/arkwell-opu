"""Safe Concurrent Update Scheduler for OPU architecture."""

from dataclasses import dataclass
import time
from typing import List
import numpy as np
from arkwell.simulator.conflict_analyzer.analyzer import CandidateUpdate, ConflictGraphResult


@dataclass
class SchedulePlan:
    """The partitioned sequence of conflict-free execution batches."""
    batches: List[List[CandidateUpdate]]
    num_batches: int
    avg_batch_width: float
    max_batch_width: int
    scheduling_time_seconds: float


class SafeConcurrentScheduler:
    """
    Schedules candidate updates into safe concurrent execution groups
    such that each group forms an independent set in the conflict graph.
    """

    @staticmethod
    def schedule(conflict_result: ConflictGraphResult) -> SchedulePlan:
        start_time = time.perf_counter()
        k = conflict_result.num_candidates
        if k == 0:
            return SchedulePlan(
                batches=[],
                num_batches=0,
                avg_batch_width=0.0,
                max_batch_width=0,
                scheduling_time_seconds=0.0,
            )

        conflict_mat = conflict_result.conflict_matrix
        candidates = conflict_result.candidates

        # Compute conflict degrees
        conflict_degrees = np.sum(conflict_mat, axis=1)

        # Priority order: highest delta objective first, then lowest conflict degree
        # To break ties deterministically, sort by (delta_objective DESC, conflict_degree ASC, var_idx ASC)
        order = sorted(
            range(k),
            key=lambda idx: (-candidates[idx].delta_objective, conflict_degrees[idx], candidates[idx].var_idx),
        )

        batches: List[List[CandidateUpdate]] = []
        assigned = np.zeros(k, dtype=bool)

        for idx in order:
            if assigned[idx]:
                continue

            current_batch_indices = [idx]
            assigned[idx] = True
            incompatible = conflict_mat[idx].copy()

            for candidate_idx in order:
                if not assigned[candidate_idx] and not incompatible[candidate_idx]:
                    current_batch_indices.append(candidate_idx)
                    assigned[candidate_idx] = True
                    incompatible |= conflict_mat[candidate_idx]

            batches.append([candidates[i] for i in current_batch_indices])

        elapsed = time.perf_counter() - start_time
        batch_widths = [len(b) for b in batches]
        avg_w = float(np.mean(batch_widths)) if len(batch_widths) > 0 else 0.0
        max_w = int(max(batch_widths)) if len(batch_widths) > 0 else 0

        return SchedulePlan(
            batches=batches,
            num_batches=len(batches),
            avg_batch_width=avg_w,
            max_batch_width=max_w,
            scheduling_time_seconds=elapsed,
        )
