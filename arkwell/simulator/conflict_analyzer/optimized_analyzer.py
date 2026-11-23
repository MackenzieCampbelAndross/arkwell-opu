"""Optimized conflict analyzer using sparse adjacency indexing and vectorized evaluation."""

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem
from arkwell.simulator.conflict_analyzer.analyzer import CandidateUpdate, ConflictGraphResult, ConflictType


class OptimizedConflictAnalyzer:
    """
    High-performance conflict analyzer.
    Preserves exact mathematical equivalence with Reference ConflictAnalyzer
    while eliminating the O(K^2) Python dictionary and set-lookup bottlenecks.
    
    Optimizations:
    1. Direct-index candidate map: cand_lookup[v] -> candidate_index
    2. Sparse adjacency neighbor intersection: queries only physical edges in G
    3. Vectorized intra-PE resource collision evaluation
    4. Optional lazy/vectorized conflict classification
    """

    def __init__(self, compiled: OPUCompiledProblem):
        self.compiled = compiled
        self.n = compiled.num_variables
        self.m = compiled.num_constraints
        self.adj = compiled.adjacency_list
        self._cand_lookup = np.full(self.n, -1, dtype=np.int32)

    def analyze_candidates(
        self,
        candidates: List[CandidateUpdate],
        current_state: np.ndarray,
        build_type_dict: bool = True,
    ) -> ConflictGraphResult:
        """
        Builds the pairwise conflict graph over candidate updates with O(K * avg_degree) complexity.
        """
        start_time = time.perf_counter()
        k = len(candidates)
        if k == 0:
            return ConflictGraphResult(
                candidates=[],
                num_candidates=0,
                conflict_matrix=np.empty((0, 0), dtype=bool),
                conflict_types={},
                num_conflicts=0,
                conflict_density=0.0,
                analysis_time_seconds=0.0,
            )

        conflict_matrix = np.zeros((k, k), dtype=bool)
        conflict_types: Dict[Tuple[int, int], ConflictType] = {}

        # Extract numpy arrays for fast vectorized access
        var_indices = np.array([c.var_idx for c in candidates], dtype=np.int32)
        pe_ids = np.array([c.pe_id for c in candidates], dtype=np.int32)
        proposed_vals = np.array([c.proposed_val for c in candidates], dtype=np.int8)

        # 1. Vectorized Resource Collision: intra-PE move conflicts
        # Two updates assigned to the same PE conflict in a single cycle
        pe_match = pe_ids[:, None] == pe_ids[None, :]
        np.fill_diagonal(pe_match, False)
        
        # 2. Vectorized Read/Write: duplicate updates on the same variable
        var_match = var_indices[:, None] == var_indices[None, :]
        np.fill_diagonal(var_match, False)

        # 3. Sparse Adjacency Neighbor Conflict Detection
        # Populate cand_lookup table
        self._cand_lookup[var_indices] = np.arange(k, dtype=np.int32)

        edge_conflicts = np.zeros((k, k), dtype=bool)
        both_adding = (proposed_vals == 1)

        for a in range(k):
            v_a = var_indices[a]
            nbrs = self.adj[v_a]
            if len(nbrs) == 0:
                continue

            # Query candidate table for neighbors
            nbr_cands = self._cand_lookup[nbrs]
            valid_cands = nbr_cands[nbr_cands != -1]

            if len(valid_cands) > 0:
                if both_adding[a]:
                    # Candidates b that are also adding (0 -> 1) have FEASIBILITY_VIOLATION
                    valid_adders = valid_cands[both_adding[valid_cands]]
                    edge_conflicts[a, valid_adders] = True

        # Clear lookup table
        self._cand_lookup[var_indices] = -1

        # Combine conflict matrix
        conflict_matrix = pe_match | var_match | edge_conflicts

        # Optional classification dictionary for report/debugging
        if build_type_dict:
            # Resource collisions
            r_rows, r_cols = np.where(pe_match)
            for r, c in zip(r_rows, r_cols):
                conflict_types[(int(r), int(c))] = ConflictType.RESOURCE_COLLISION

            # Read-write collisions
            rw_rows, rw_cols = np.where(var_match & (~pe_match))
            for r, c in zip(rw_rows, rw_cols):
                conflict_types[(int(r), int(c))] = ConflictType.READ_WRITE

            # Feasibility violations
            f_rows, f_cols = np.where(edge_conflicts & (~pe_match) & (~var_match))
            for r, c in zip(f_rows, f_cols):
                conflict_types[(int(r), int(c))] = ConflictType.FEASIBILITY_VIOLATION

        total_possible = k * (k - 1) // 2
        actual_conflicts = int(np.sum(conflict_matrix)) // 2
        density = (actual_conflicts / total_possible) if total_possible > 0 else 0.0
        elapsed = time.perf_counter() - start_time

        return ConflictGraphResult(
            candidates=candidates,
            num_candidates=k,
            conflict_matrix=conflict_matrix,
            conflict_types=conflict_types,
            num_conflicts=actual_conflicts,
            conflict_density=density,
            analysis_time_seconds=elapsed,
        )
