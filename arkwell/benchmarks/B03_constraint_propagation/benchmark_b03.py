"""B03 Constraint Propagation Traffic Benchmark Suite."""

from dataclasses import dataclass
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from arkwell.instances.maximum_independent_set.generator import generate_mis_instance
from arkwell.simulator.compiler.mis_compiler import compile_mis_to_opu
from arkwell.simulator.processing_elements.pe_array import PEArray, PropagationMetrics
from arkwell.simulator.solvers import (
    solve_config_c_conflict_aware,
    solve_config_d_selective_propagation,
)
from arkwell.simulator.verifier.solution_verifier import IndependentVerifier


@dataclass
class B03ProtocolComparisonRecord:
    instance_name: str
    num_variables: int
    density: float
    seed: int
    num_pes: int
    protocol: str  # 'P0_broadcast', 'P1_selective', 'P2_batched_selective'

    messages: int
    header_bytes: int
    payload_bytes: int
    total_bytes: int
    affected_pe_deliveries: int
    unnecessary_updates: int
    state_invalidations: int
    synchronization_barriers: int
    message_reduction_vs_p0_pct: float
    byte_reduction_vs_p0_pct: float

    def to_dict(self) -> Dict:
        return self.__dict__


def run_b03_benchmark(
    problem_sizes: List[int] = [512, 1024],
    densities: List[float] = [0.02, 0.10, 0.30],
    seeds: List[int] = [42, 101, 2024],
    pe_counts: List[int] = [4, 8, 16],
) -> List[Dict]:
    """Runs B03 Constraint Propagation Traffic comparison between P0, P1, and P2."""
    records: List[B03ProtocolComparisonRecord] = []

    for n in problem_sizes:
        for d in densities:
            for s in seeds:
                inst = generate_mis_instance(n, d, s)
                for pes in pe_counts:
                    # Run Config C (P0 broadcast)
                    res_p0 = solve_config_c_conflict_aware(inst, num_pes=pes, seed=s, use_optimized_analyzer=True)
                    # Run Config D (P1 unbatched selective)
                    res_p1 = solve_config_d_selective_propagation(
                        inst, num_pes=pes, seed=s, use_optimized_analyzer=True, use_batched_propagation=False
                    )
                    # Run Config D (P2 batched selective)
                    res_p2 = solve_config_d_selective_propagation(
                        inst, num_pes=pes, seed=s, use_optimized_analyzer=True, use_batched_propagation=True
                    )

                    # Verify correctness of all 3
                    assert res_p0.is_feasible, f"P0 infeasible on {inst.name}"
                    assert res_p1.is_feasible, f"P1 infeasible on {inst.name}"
                    assert res_p2.is_feasible, f"P2 infeasible on {inst.name}"
                    assert res_p0.final_objective == res_p1.final_objective == res_p2.final_objective

                    p0_msgs = res_p0.communication_messages
                    p0_bytes = res_p0.communication_bytes

                    # Record P0
                    records.append(
                        B03ProtocolComparisonRecord(
                            instance_name=inst.name,
                            num_variables=n,
                            density=d,
                            seed=s,
                            num_pes=pes,
                            protocol="P0_broadcast",
                            messages=p0_msgs,
                            header_bytes=res_p0.communication_header_bytes,
                            payload_bytes=res_p0.communication_payload_bytes,
                            total_bytes=p0_bytes,
                            affected_pe_deliveries=res_p0.communication_messages,
                            unnecessary_updates=res_p0.unnecessary_updates,
                            state_invalidations=res_p0.state_invalidations,
                            synchronization_barriers=res_p0.synchronization_count,
                            message_reduction_vs_p0_pct=0.0,
                            byte_reduction_vs_p0_pct=0.0,
                        )
                    )

                    # Record P1
                    p1_msgs = res_p1.communication_messages
                    p1_bytes = res_p1.communication_bytes
                    msg_red_p1 = ((p0_msgs - p1_msgs) / p0_msgs * 100.0) if p0_msgs > 0 else 0.0
                    byte_red_p1 = ((p0_bytes - p1_bytes) / p0_bytes * 100.0) if p0_bytes > 0 else 0.0

                    records.append(
                        B03ProtocolComparisonRecord(
                            instance_name=inst.name,
                            num_variables=n,
                            density=d,
                            seed=s,
                            num_pes=pes,
                            protocol="P1_selective",
                            messages=p1_msgs,
                            header_bytes=res_p1.communication_header_bytes,
                            payload_bytes=res_p1.communication_payload_bytes,
                            total_bytes=p1_bytes,
                            affected_pe_deliveries=p1_msgs,
                            unnecessary_updates=res_p1.unnecessary_updates,
                            state_invalidations=res_p1.state_invalidations,
                            synchronization_barriers=res_p1.synchronization_count,
                            message_reduction_vs_p0_pct=msg_red_p1,
                            byte_reduction_vs_p0_pct=byte_red_p1,
                        )
                    )

                    # Record P2
                    p2_msgs = res_p2.communication_messages
                    p2_bytes = res_p2.communication_bytes
                    msg_red_p2 = ((p0_msgs - p2_msgs) / p0_msgs * 100.0) if p0_msgs > 0 else 0.0
                    byte_red_p2 = ((p0_bytes - p2_bytes) / p0_bytes * 100.0) if p0_bytes > 0 else 0.0

                    records.append(
                        B03ProtocolComparisonRecord(
                            instance_name=inst.name,
                            num_variables=n,
                            density=d,
                            seed=s,
                            num_pes=pes,
                            protocol="P2_batched_selective",
                            messages=p2_msgs,
                            header_bytes=res_p2.communication_header_bytes,
                            payload_bytes=res_p2.communication_payload_bytes,
                            total_bytes=p2_bytes,
                            affected_pe_deliveries=p2_msgs,
                            unnecessary_updates=res_p2.unnecessary_updates,
                            state_invalidations=res_p2.state_invalidations,
                            synchronization_barriers=res_p2.synchronization_count,
                            message_reduction_vs_p0_pct=msg_red_p2,
                            byte_reduction_vs_p0_pct=byte_red_p2,
                        )
                    )

    return [r.to_dict() for r in records]
