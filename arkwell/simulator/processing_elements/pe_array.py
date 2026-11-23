"""Processing Element (PE) array and interconnect modeling."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np
from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem
from arkwell.simulator.conflict_analyzer.analyzer import CandidateUpdate


@dataclass
class InterconnectStats:
    """Interconnect communication and synchronization metrics."""
    broadcast_messages: int = 0
    broadcast_bytes: int = 0
    selective_messages: int = 0
    selective_bytes: int = 0
    synchronization_barriers: int = 0


class PEArray:
    """
    Models an array of P parallel Processing Elements with local partitions
    and interconnect message routing.
    """

    BYTES_PER_UPDATE_MESSAGE = 8  # 4 bytes variable ID + 4 bytes delta value

    def __init__(self, num_pes: int, compiled: OPUCompiledProblem):
        if num_pes <= 0:
            raise ValueError("num_pes must be >= 1")
        self.num_pes = num_pes
        self.compiled = compiled
        self.n = compiled.num_variables
        self.adj = compiled.adjacency_list

        # Partition variables round-robin across PEs
        self.pe_to_vars: List[List[int]] = [[] for _ in range(num_pes)]
        self.var_to_pe: Dict[int, int] = {}
        for v in range(self.n):
            pe_id = v % num_pes
            self.pe_to_vars[pe_id].append(v)
            self.var_to_pe[v] = pe_id

        # Precompute subscription map for Selective Propagation:
        # subscribers[u] = set of PE IDs that host at least one neighbor of u (excluding owner)
        self.subscribers: List[List[int]] = []
        for u in range(self.n):
            owner = self.var_to_pe[u]
            nbr_pes = set()
            for nbr in self.adj[u]:
                nbr_pe = self.var_to_pe[int(nbr)]
                if nbr_pe != owner:
                    nbr_pes.add(nbr_pe)
            self.subscribers.append(sorted(nbr_pes))

        self.stats = InterconnectStats()

    def generate_candidate_moves(
        self,
        current_state: np.ndarray,
        active_counts: np.ndarray,
    ) -> List[CandidateUpdate]:
        """
        Each PE inspects its assigned variables and proposes candidate moves.
        In MIS local search: each PE proposes adding a currently inactive vertex (x_i = 0)
        that has 0 active neighbors (active_counts[i] == 0).
        """
        candidates: List[CandidateUpdate] = []
        for pe_id in range(self.num_pes):
            for v in self.pe_to_vars[pe_id]:
                if current_state[v] == 0 and active_counts[v] == 0:
                    candidates.append(
                        CandidateUpdate(
                            pe_id=pe_id,
                            var_idx=v,
                            current_val=0,
                            proposed_val=1,
                            delta_objective=1.0,
                        )
                    )
        return candidates

    def route_broadcast_updates(self, modified_vars: List[int]) -> Tuple[int, int]:
        """Global broadcast: every update is broadcast to all other P-1 PEs."""
        if self.num_pes <= 1 or len(modified_vars) == 0:
            return 0, 0
        msgs = len(modified_vars) * (self.num_pes - 1)
        bytes_vol = msgs * self.BYTES_PER_UPDATE_MESSAGE
        self.stats.broadcast_messages += msgs
        self.stats.broadcast_bytes += bytes_vol
        return msgs, bytes_vol

    def route_selective_updates(self, modified_vars: List[int]) -> Tuple[int, int]:
        """Selective propagation: update on v is sent ONLY to subscriber PEs."""
        if self.num_pes <= 1 or len(modified_vars) == 0:
            return 0, 0
        msgs = 0
        for v in modified_vars:
            msgs += len(self.subscribers[v])
        bytes_vol = msgs * self.BYTES_PER_UPDATE_MESSAGE
        self.stats.selective_messages += msgs
        self.stats.selective_bytes += bytes_vol
        return msgs, bytes_vol
