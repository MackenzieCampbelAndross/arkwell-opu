"""B02 Conflict Scheduling Efficiency Benchmark Suite."""

from dataclasses import dataclass
import json
import os
import time
from typing import Dict, List
import numpy as np

from arkwell.instances.maximum_independent_set.generator import generate_mis_instance
from arkwell.simulator.compiler.mis_compiler import compile_mis_to_opu
from arkwell.simulator.conflict_analyzer.analyzer import ConflictAnalyzer, CandidateUpdate
from arkwell.simulator.conflict_analyzer.optimized_analyzer import OptimizedConflictAnalyzer
from arkwell.simulator.scheduler.scheduler import SafeConcurrentScheduler


@dataclass
class B02BenchmarkRecord:
    instance_name: str
    num_variables: int
    density: float
    seed: int
    num_pes: int
    num_candidates: int
    
    # Conflict Graph Metrics
    ref_construction_time_ms: float
    opt_construction_time_ms: float
    construction_speedup: float
    num_conflicts: int
    conflict_density: float
    average_conflict_degree: float
    
    # Scheduling Metrics
    num_execution_groups: int
    avg_batch_width: float
    max_batch_width: int
    scheduling_time_ms: float
    pe_utilization: float

    def to_dict(self) -> Dict:
        return self.__dict__


def run_b02_benchmark(
    problem_sizes: List[int] = [512, 1024],
    densities: List[float] = [0.02, 0.10, 0.30],
    seeds: List[int] = [42, 101, 2024],
    pe_counts: List[int] = [4, 8, 16],
    run_ref_timing: bool = True,
) -> List[Dict]:
    """Runs B02 Conflict Scheduling Benchmark across instance matrix."""
    records: List[B02BenchmarkRecord] = []
    
    for n in problem_sizes:
        for d in densities:
            for s in seeds:
                inst = generate_mis_instance(n, d, s)
                compiled = compile_mis_to_opu(inst)
                ref_analyzer = ConflictAnalyzer(compiled)
                opt_analyzer = OptimizedConflictAnalyzer(compiled)
                scheduler = SafeConcurrentScheduler()
                state = np.zeros(n, dtype=np.int32)

                for pes in pe_counts:
                    # Realistic candidate pool: active PEs proposing moves from round 1
                    cands = [
                        CandidateUpdate(
                            pe_id=i % pes,
                            var_idx=i,
                            current_val=0,
                            proposed_val=1,
                            delta_objective=1.0,
                        )
                        for i in range(n)
                    ]
                    k = len(cands)

                    # Reference analyzer timing (skip for n > 1024 to save time if desired)
                    if run_ref_timing and n <= 1024:
                        t0 = time.perf_counter()
                        c_ref = ref_analyzer.analyze_candidates(cands, state)
                        t_ref_ms = (time.perf_counter() - t0) * 1000.0
                    else:
                        t_ref_ms = 0.0

                    # Optimized analyzer timing
                    t0 = time.perf_counter()
                    c_opt = opt_analyzer.analyze_candidates(cands, state, build_type_dict=False)
                    t_opt_ms = (time.perf_counter() - t0) * 1000.0

                    speedup = (t_ref_ms / t_opt_ms) if t_opt_ms > 0 and t_ref_ms > 0 else 1.0

                    # Scheduler evaluation
                    t0 = time.perf_counter()
                    plan = scheduler.schedule(c_opt)
                    t_sched_ms = (time.perf_counter() - t0) * 1000.0

                    avg_deg = (2.0 * c_opt.num_conflicts / k) if k > 0 else 0.0
                    pe_util = (len(plan.batches[0]) / pes) if len(plan.batches) > 0 else 0.0

                    rec = B02BenchmarkRecord(
                        instance_name=inst.name,
                        num_variables=n,
                        density=d,
                        seed=s,
                        num_pes=pes,
                        num_candidates=k,
                        ref_construction_time_ms=t_ref_ms,
                        opt_construction_time_ms=t_opt_ms,
                        construction_speedup=speedup,
                        num_conflicts=c_opt.num_conflicts,
                        conflict_density=c_opt.conflict_density,
                        average_conflict_degree=avg_deg,
                        num_execution_groups=plan.num_batches,
                        avg_batch_width=plan.avg_batch_width,
                        max_batch_width=plan.max_batch_width,
                        scheduling_time_ms=t_sched_ms,
                        pe_utilization=min(1.0, pe_util),
                    )
                    records.append(rec)

    return [r.to_dict() for r in records]
