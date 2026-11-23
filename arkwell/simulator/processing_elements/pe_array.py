"""Processing Element (PE) array and interconnect modeling with ABS V1.0 B03 protocols."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np
from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem
from arkwell.simulator.conflict_analyzer.analyzer import CandidateUpdate


@dataclass
class PropagationMetrics:
    """Detailed telemetry for interconnect propagation protocols (P0, P1, P2)."""
    protocol: str  # 'P0_broadcast', 'P1_selective', 'P2_batched_selective'
    messages: int = 0
    header_bytes: int = 0
    payload_bytes: int = 0
    total_bytes: int = 0
    affected_pe_deliveries: int = 0
    unnecessary_updates: int = 0
    synchronization_barriers: int = 0
    state_invalidations: int = 0
    propagation_time_seconds: float = 0.0


class PEArray:
    """
    Models an array of P parallel Processing Elements with local partitions
    and multi-protocol interconnect routing (P0 Broadcast, P1 Selective, P2 Batched Selective).
    """

    # Protocol framing constants (in bytes)
    HEADER_BYTES_PER_UNBATCHED_MSG = 4   # 2 bytes src/dst PE, 1 byte type, 1 byte seq
    PAYLOAD_BYTES_PER_UPDATE = 4         # 2 bytes var ID, 2 bytes delta/val
    HEADER_BYTES_PER_BATCHED_PACKET = 8  # 4 bytes packet header + 4 bytes batch descriptor/mask

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
        # subscribers[u] = sorted list of unique PE IDs hosting at least one neighbor of u (excluding owner)
        self.subscribers: List[List[int]] = []
        for u in range(self.n):
            owner = self.var_to_pe[u]
            nbr_pes = set()
            for nbr in self.adj[u]:
                nbr_pe = self.var_to_pe[int(nbr)]
                if nbr_pe != owner:
                    nbr_pes.add(nbr_pe)
            self.subscribers.append(sorted(nbr_pes))

        # Separate protocol metrics
        self.p0_stats = PropagationMetrics(protocol="P0_broadcast")
        self.p1_stats = PropagationMetrics(protocol="P1_selective")
        self.p2_stats = PropagationMetrics(protocol="P2_batched_selective")

    def generate_candidate_moves(
        self,
        current_state: np.ndarray,
        active_counts: np.ndarray,
    ) -> List[CandidateUpdate]:
        """
        Each PE inspects its assigned partition and proposes candidate moves.
        In MIS: proposes adding an inactive vertex (x_i = 0) that has 0 active neighbors.
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

    def route_p0_broadcast(self, modified_vars: List[int]) -> PropagationMetrics:
        """
        P0: Full state broadcast.
        Every update is sent to all other P-1 PEs.
        """
        if self.num_pes <= 1 or len(modified_vars) == 0:
            return PropagationMetrics(protocol="P0_broadcast")

        msgs = len(modified_vars) * (self.num_pes - 1)
        h_bytes = msgs * self.HEADER_BYTES_PER_UNBATCHED_MSG
        p_bytes = msgs * self.PAYLOAD_BYTES_PER_UPDATE
        tot_bytes = h_bytes + p_bytes

        # Calculate unnecessary updates: deliveries to PEs that have NO incident constraints
        unnecessary = 0
        state_inval = 0
        for v in modified_vars:
            subs = set(self.subscribers[v])
            owner = self.var_to_pe[v]
            for target_pe in range(self.num_pes):
                if target_pe == owner:
                    continue
                if target_pe not in subs:
                    unnecessary += 1
                else:
                    state_inval += 1

        self.p0_stats.messages += msgs
        self.p0_stats.header_bytes += h_bytes
        self.p0_stats.payload_bytes += p_bytes
        self.p0_stats.total_bytes += tot_bytes
        self.p0_stats.affected_pe_deliveries += msgs
        self.p0_stats.unnecessary_updates += unnecessary
        self.p0_stats.state_invalidations += state_inval
        self.p0_stats.synchronization_barriers += 1

        return PropagationMetrics(
            protocol="P0_broadcast",
            messages=msgs,
            header_bytes=h_bytes,
            payload_bytes=p_bytes,
            total_bytes=tot_bytes,
            affected_pe_deliveries=msgs,
            unnecessary_updates=unnecessary,
            synchronization_barriers=1,
            state_invalidations=state_inval,
        )

    def route_p1_selective(self, modified_vars: List[int]) -> PropagationMetrics:
        """
        P1: Selective dependency-based propagation (unbatched).
        Updates are sent strictly to PEs hosting incident constraints.
        """
        if self.num_pes <= 1 or len(modified_vars) == 0:
            return PropagationMetrics(protocol="P1_selective")

        msgs = 0
        for v in modified_vars:
            msgs += len(self.subscribers[v])

        h_bytes = msgs * self.HEADER_BYTES_PER_UNBATCHED_MSG
        p_bytes = msgs * self.PAYLOAD_BYTES_PER_UPDATE
        tot_bytes = h_bytes + p_bytes

        self.p1_stats.messages += msgs
        self.p1_stats.header_bytes += h_bytes
        self.p1_stats.payload_bytes += p_bytes
        self.p1_stats.total_bytes += tot_bytes
        self.p1_stats.affected_pe_deliveries += msgs
        self.p1_stats.unnecessary_updates += 0  # Zero unnecessary updates by construction
        self.p1_stats.state_invalidations += msgs
        self.p1_stats.synchronization_barriers += 1

        return PropagationMetrics(
            protocol="P1_selective",
            messages=msgs,
            header_bytes=h_bytes,
            payload_bytes=p_bytes,
            total_bytes=tot_bytes,
            affected_pe_deliveries=msgs,
            unnecessary_updates=0,
            synchronization_barriers=1,
            state_invalidations=msgs,
        )

    def route_p2_batched_selective(self, modified_vars: List[int]) -> PropagationMetrics:
        """
        P2: Selective propagation with batching.
        Coalesces all variable updates in a commit batch destined for the same PE into a single packet.
        """
        if self.num_pes <= 1 or len(modified_vars) == 0:
            return PropagationMetrics(protocol="P2_batched_selective")

        # Map recipient PE -> list of variables affecting it in this batch
        pe_payloads: Dict[int, List[int]] = {}
        for v in modified_vars:
            for sub_pe in self.subscribers[v]:
                if sub_pe not in pe_payloads:
                    pe_payloads[sub_pe] = []
                pe_payloads[sub_pe].append(v)

        packets = len(pe_payloads)
        h_bytes = packets * self.HEADER_BYTES_PER_BATCHED_PACKET
        total_items = sum(len(items) for items in pe_payloads.values())
        p_bytes = total_items * self.PAYLOAD_BYTES_PER_UPDATE
        tot_bytes = h_bytes + p_bytes

        self.p2_stats.messages += packets
        self.p2_stats.header_bytes += h_bytes
        self.p2_stats.payload_bytes += p_bytes
        self.p2_stats.total_bytes += tot_bytes
        self.p2_stats.affected_pe_deliveries += packets
        self.p2_stats.unnecessary_updates += 0
        self.p2_stats.state_invalidations += total_items
        self.p2_stats.synchronization_barriers += 1

        return PropagationMetrics(
            protocol="P2_batched_selective",
            messages=packets,
            header_bytes=h_bytes,
            payload_bytes=p_bytes,
            total_bytes=tot_bytes,
            affected_pe_deliveries=packets,
            unnecessary_updates=0,
            synchronization_barriers=1,
            state_invalidations=total_items,
        )

    # Backward compatibility bridges
    def route_broadcast_updates(self, modified_vars: List[int]) -> Tuple[int, int]:
        m = self.route_p0_broadcast(modified_vars)
        return m.messages, m.total_bytes

    def route_selective_updates(self, modified_vars: List[int]) -> Tuple[int, int]:
        m = self.route_p1_selective(modified_vars)
        return m.messages, m.total_bytes
