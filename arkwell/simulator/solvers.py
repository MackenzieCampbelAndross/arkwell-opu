"""Arkwell OPU Solvers implementing Configurations A, B, C, and D with Milestone 2 instrumentation."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import psutil

from arkwell.instances.maximum_independent_set.generator import MISInstance
from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem, compile_mis_to_opu
from arkwell.simulator.conflict_analyzer.analyzer import ConflictAnalyzer, ConflictType
from arkwell.simulator.conflict_analyzer.optimized_analyzer import OptimizedConflictAnalyzer
from arkwell.simulator.scheduler.scheduler import SafeConcurrentScheduler
from arkwell.simulator.processing_elements.pe_array import PEArray, PropagationMetrics
from arkwell.simulator.state_manager.state_manager import StateManager
from arkwell.simulator.verifier.solution_verifier import IndependentVerifier
from arkwell.backends.cpu.engine import CPUEngine


@dataclass
class SolverTimingBreakdown:
    """Detailed phase timing breakdown separating algorithmic execution from simulator instrumentation."""
    algorithmic_time_seconds: float = 0.0
    conflict_construction_time_seconds: float = 0.0
    scheduling_time_seconds: float = 0.0
    propagation_time_seconds: float = 0.0
    verification_time_seconds: float = 0.0
    total_execution_time_seconds: float = 0.0


@dataclass
class SolverResult:
    """Standardized result schema adhering to Section 8 & Milestone 2 benchmark metrics."""
    configuration: str  # 'A', 'B', 'C', or 'D'
    instance_name: str
    num_variables: int
    num_constraints: int
    num_pes: int
    seed: int

    # Optimization metrics
    initial_objective: int
    final_objective: int
    is_feasible: bool
    num_violations: int

    # Timing metrics
    execution_time_seconds: float
    time_to_first_feasible_seconds: float
    time_to_target_quality_seconds: Optional[float]
    scheduling_latency_seconds: float

    # Architectural & Hardware metrics
    conflict_count: int
    conflict_density: float
    average_conflict_degree: float
    num_execution_groups: int
    avg_batch_width: float
    max_batch_width: int
    pe_utilization: float
    communication_messages: int
    communication_bytes: int
    communication_header_bytes: int
    communication_payload_bytes: int
    unnecessary_updates: int
    state_invalidations: int
    synchronization_count: int
    memory_consumption_bytes: int
    gpu_memory_consumption_bytes: int

    # Structured details with defaults
    timings: SolverTimingBreakdown = field(default_factory=SolverTimingBreakdown)
    final_state: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    def to_dict(self) -> Dict:
        return {
            "configuration": self.configuration,
            "instance_name": self.instance_name,
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "num_pes": self.num_pes,
            "seed": self.seed,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "is_feasible": self.is_feasible,
            "num_violations": self.num_violations,
            "execution_time_seconds": self.execution_time_seconds,
            "algorithmic_time_seconds": self.timings.algorithmic_time_seconds,
            "conflict_construction_time_seconds": self.timings.conflict_construction_time_seconds,
            "scheduling_time_seconds": self.timings.scheduling_time_seconds,
            "propagation_time_seconds": self.timings.propagation_time_seconds,
            "verification_time_seconds": self.timings.verification_time_seconds,
            "conflict_count": self.conflict_count,
            "conflict_density": self.conflict_density,
            "average_conflict_degree": self.average_conflict_degree,
            "num_execution_groups": self.num_execution_groups,
            "avg_batch_width": self.avg_batch_width,
            "max_batch_width": self.max_batch_width,
            "pe_utilization": self.pe_utilization,
            "communication_messages": self.communication_messages,
            "communication_bytes": self.communication_bytes,
            "communication_header_bytes": self.communication_header_bytes,
            "communication_payload_bytes": self.communication_payload_bytes,
            "unnecessary_updates": self.unnecessary_updates,
            "state_invalidations": self.state_invalidations,
            "synchronization_count": self.synchronization_count,
            "memory_consumption_bytes": self.memory_consumption_bytes,
            "gpu_memory_consumption_bytes": self.gpu_memory_consumption_bytes,
        }


def solve_config_a_sequential(
    instance: MISInstance,
    max_iterations: int = 10000,
    random_order: bool = True,
    seed: int = 42,
) -> SolverResult:
    """Configuration A: Sequential Local Search (Single PE baseline reference)."""
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())

    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = instance.num_vertices
    order = np.arange(n)
    if random_order:
        rng.shuffle(order)

    t_algo_start = time.perf_counter()
    for v in order:
        if cpu_engine.is_addition_feasible(engine_state, int(v)):
            cpu_engine.apply_flip(engine_state, int(v))
            state_mgr.commit_batch([int(v)])
    t_algo = time.perf_counter() - t_algo_start

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()

    t_verif_start = time.perf_counter()
    verification = IndependentVerifier.verify_mis(instance, final_state)
    t_verif = time.perf_counter() - t_verif_start

    timings = SolverTimingBreakdown(
        algorithmic_time_seconds=t_algo,
        verification_time_seconds=t_verif,
        total_execution_time_seconds=elapsed,
    )

    mem_bytes = psutil.Process().memory_info().rss

    return SolverResult(
        configuration="A",
        instance_name=instance.name,
        num_variables=instance.num_vertices,
        num_constraints=instance.num_edges,
        num_pes=1,
        seed=seed,
        initial_objective=0,
        final_objective=verification.objective_value,
        is_feasible=verification.is_feasible,
        num_violations=verification.num_violations,
        execution_time_seconds=elapsed,
        time_to_first_feasible_seconds=0.0,
        time_to_target_quality_seconds=elapsed,
        scheduling_latency_seconds=0.0,
        timings=timings,
        conflict_count=0,
        conflict_density=0.0,
        average_conflict_degree=0.0,
        num_execution_groups=verification.objective_value,
        avg_batch_width=1.0,
        max_batch_width=1,
        pe_utilization=1.0,
        communication_messages=0,
        communication_bytes=0,
        communication_header_bytes=0,
        communication_payload_bytes=0,
        unnecessary_updates=0,
        state_invalidations=0,
        synchronization_count=verification.objective_value,
        memory_consumption_bytes=mem_bytes,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )


def solve_config_b_naive_parallel(
    instance: MISInstance,
    num_pes: int = 4,
    seed: int = 42,
) -> SolverResult:
    """Configuration B: Naive Parallel Local Search (Broadcast interconnect)."""
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())
    pe_array = PEArray(num_pes, compiled)

    start_time = time.perf_counter()
    round_count = 0
    total_committed = 0
    rejected_moves = 0
    t_algo_accum = 0.0
    t_prop_accum = 0.0

    while True:
        round_count += 1
        t0 = time.perf_counter()
        candidates = pe_array.generate_candidate_moves(
            engine_state.state_vector,
            engine_state.active_neighbor_counts,
        )
        if len(candidates) == 0:
            break

        committed_this_round: List[int] = []
        for cand in candidates:
            v = cand.var_idx
            if cpu_engine.is_addition_feasible(engine_state, v):
                cpu_engine.apply_flip(engine_state, v)
                committed_this_round.append(v)
            else:
                rejected_moves += 1
        t_algo_accum += time.perf_counter() - t0

        if len(committed_this_round) > 0:
            state_mgr.commit_batch(committed_this_round)
            t_p0 = time.perf_counter()
            pe_array.route_p0_broadcast(committed_this_round)
            t_prop_accum += time.perf_counter() - t_p0
            total_committed += len(committed_this_round)

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()

    t_verif_start = time.perf_counter()
    verification = IndependentVerifier.verify_mis(instance, final_state)
    t_verif = time.perf_counter() - t_verif_start

    pe_util = (total_committed / (round_count * num_pes)) if round_count > 0 else 0.0
    p0 = pe_array.p0_stats

    timings = SolverTimingBreakdown(
        algorithmic_time_seconds=t_algo_accum,
        propagation_time_seconds=t_prop_accum,
        verification_time_seconds=t_verif,
        total_execution_time_seconds=elapsed,
    )

    return SolverResult(
        configuration="B",
        instance_name=instance.name,
        num_variables=instance.num_vertices,
        num_constraints=instance.num_edges,
        num_pes=num_pes,
        seed=seed,
        initial_objective=0,
        final_objective=verification.objective_value,
        is_feasible=verification.is_feasible,
        num_violations=verification.num_violations,
        execution_time_seconds=elapsed,
        time_to_first_feasible_seconds=0.0,
        time_to_target_quality_seconds=elapsed,
        scheduling_latency_seconds=0.0,
        timings=timings,
        conflict_count=rejected_moves,
        conflict_density=0.0,
        average_conflict_degree=0.0,
        num_execution_groups=round_count,
        avg_batch_width=(total_committed / round_count) if round_count > 0 else 0.0,
        max_batch_width=num_pes,
        pe_utilization=min(1.0, pe_util),
        communication_messages=p0.messages,
        communication_bytes=p0.total_bytes,
        communication_header_bytes=p0.header_bytes,
        communication_payload_bytes=p0.payload_bytes,
        unnecessary_updates=p0.unnecessary_updates,
        state_invalidations=p0.state_invalidations,
        synchronization_count=p0.synchronization_barriers,
        memory_consumption_bytes=psutil.Process().memory_info().rss,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )


def solve_config_c_conflict_aware(
    instance: MISInstance,
    num_pes: int = 4,
    seed: int = 42,
    use_optimized_analyzer: bool = True,
) -> SolverResult:
    """
    Configuration C: Conflict-Aware Scheduling.
    Uses global broadcast (P0) interconnect.
    """
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())
    pe_array = PEArray(num_pes, compiled)

    analyzer = OptimizedConflictAnalyzer(compiled) if use_optimized_analyzer else ConflictAnalyzer(compiled)
    scheduler = SafeConcurrentScheduler()

    start_time = time.perf_counter()
    total_conflicts = 0
    t_conflict_accum = 0.0
    t_sched_accum = 0.0
    t_algo_accum = 0.0
    t_prop_accum = 0.0
    all_batch_widths: List[int] = []
    total_committed = 0
    rounds = 0
    total_cands_evaluated = 0

    while True:
        rounds += 1
        t_gen = time.perf_counter()
        candidates = pe_array.generate_candidate_moves(
            engine_state.state_vector,
            engine_state.active_neighbor_counts,
        )
        t_algo_accum += time.perf_counter() - t_gen
        if len(candidates) == 0:
            break

        total_cands_evaluated += len(candidates)

        # 1. Conflict analysis
        t_c0 = time.perf_counter()
        if isinstance(analyzer, OptimizedConflictAnalyzer):
            c_res = analyzer.analyze_candidates(candidates, engine_state.state_vector, build_type_dict=False)
        else:
            c_res = analyzer.analyze_candidates(candidates, engine_state.state_vector)
        t_conflict_accum += time.perf_counter() - t_c0
        total_conflicts += c_res.num_conflicts

        # 2. Scheduling
        t_s0 = time.perf_counter()
        plan = scheduler.schedule(c_res)
        t_sched_accum += time.perf_counter() - t_s0

        # 3. Commit primary maximal safe concurrent batch
        if len(plan.batches) > 0:
            batch = plan.batches[0]
            batch_vars = [cand.var_idx for cand in batch]

            t_a0 = time.perf_counter()
            cpu_engine.apply_batch_flips(engine_state, batch_vars)
            state_mgr.commit_batch(batch_vars)
            t_algo_accum += time.perf_counter() - t_a0

            # Route via global broadcast (P0)
            t_p0 = time.perf_counter()
            pe_array.route_p0_broadcast(batch_vars)
            t_prop_accum += time.perf_counter() - t_p0

            all_batch_widths.append(len(batch_vars))
            total_committed += len(batch_vars)

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()

    t_verif_start = time.perf_counter()
    verification = IndependentVerifier.verify_mis(instance, final_state)
    t_verif = time.perf_counter() - t_verif_start

    num_groups = len(all_batch_widths)
    avg_width = float(np.mean(all_batch_widths)) if num_groups > 0 else 0.0
    max_width = int(max(all_batch_widths)) if num_groups > 0 else 0
    pe_util = (total_committed / (num_groups * num_pes)) if num_groups > 0 else 0.0
    avg_degree = (2.0 * total_conflicts / total_cands_evaluated) if total_cands_evaluated > 0 else 0.0

    p0 = pe_array.p0_stats
    timings = SolverTimingBreakdown(
        algorithmic_time_seconds=t_algo_accum,
        conflict_construction_time_seconds=t_conflict_accum,
        scheduling_time_seconds=t_sched_accum,
        propagation_time_seconds=t_prop_accum,
        verification_time_seconds=t_verif,
        total_execution_time_seconds=elapsed,
    )

    return SolverResult(
        configuration="C",
        instance_name=instance.name,
        num_variables=instance.num_vertices,
        num_constraints=instance.num_edges,
        num_pes=num_pes,
        seed=seed,
        initial_objective=0,
        final_objective=verification.objective_value,
        is_feasible=verification.is_feasible,
        num_violations=verification.num_violations,
        execution_time_seconds=elapsed,
        time_to_first_feasible_seconds=0.0,
        time_to_target_quality_seconds=elapsed,
        scheduling_latency_seconds=t_sched_accum,
        timings=timings,
        conflict_count=total_conflicts,
        conflict_density=0.0,
        average_conflict_degree=avg_degree,
        num_execution_groups=num_groups,
        avg_batch_width=avg_width,
        max_batch_width=max_width,
        pe_utilization=min(1.0, pe_util),
        communication_messages=p0.messages,
        communication_bytes=p0.total_bytes,
        communication_header_bytes=p0.header_bytes,
        communication_payload_bytes=p0.payload_bytes,
        unnecessary_updates=p0.unnecessary_updates,
        state_invalidations=p0.state_invalidations,
        synchronization_count=p0.synchronization_barriers,
        memory_consumption_bytes=psutil.Process().memory_info().rss,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )


def solve_config_d_selective_propagation(
    instance: MISInstance,
    num_pes: int = 4,
    seed: int = 42,
    use_optimized_analyzer: bool = True,
    use_batched_propagation: bool = False,  # If True uses P2, else P1
) -> SolverResult:
    """
    Configuration D: Conflict-Aware Scheduling with Selective Propagation (P1 or P2).
    """
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())
    pe_array = PEArray(num_pes, compiled)

    analyzer = OptimizedConflictAnalyzer(compiled) if use_optimized_analyzer else ConflictAnalyzer(compiled)
    scheduler = SafeConcurrentScheduler()

    start_time = time.perf_counter()
    total_conflicts = 0
    t_conflict_accum = 0.0
    t_sched_accum = 0.0
    t_algo_accum = 0.0
    t_prop_accum = 0.0
    all_batch_widths: List[int] = []
    total_committed = 0
    rounds = 0
    total_cands_evaluated = 0

    while True:
        rounds += 1
        t_gen = time.perf_counter()
        candidates = pe_array.generate_candidate_moves(
            engine_state.state_vector,
            engine_state.active_neighbor_counts,
        )
        t_algo_accum += time.perf_counter() - t_gen
        if len(candidates) == 0:
            break

        total_cands_evaluated += len(candidates)

        t_c0 = time.perf_counter()
        if isinstance(analyzer, OptimizedConflictAnalyzer):
            c_res = analyzer.analyze_candidates(candidates, engine_state.state_vector, build_type_dict=False)
        else:
            c_res = analyzer.analyze_candidates(candidates, engine_state.state_vector)
        t_conflict_accum += time.perf_counter() - t_c0
        total_conflicts += c_res.num_conflicts

        t_s0 = time.perf_counter()
        plan = scheduler.schedule(c_res)
        t_sched_accum += time.perf_counter() - t_s0

        if len(plan.batches) > 0:
            batch = plan.batches[0]
            batch_vars = [cand.var_idx for cand in batch]

            t_a0 = time.perf_counter()
            cpu_engine.apply_batch_flips(engine_state, batch_vars)
            state_mgr.commit_batch(batch_vars)
            t_algo_accum += time.perf_counter() - t_a0

            # Route selectively
            t_p0 = time.perf_counter()
            if use_batched_propagation:
                pe_array.route_p2_batched_selective(batch_vars)
            else:
                pe_array.route_p1_selective(batch_vars)
            t_prop_accum += time.perf_counter() - t_p0

            all_batch_widths.append(len(batch_vars))
            total_committed += len(batch_vars)

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()

    t_verif_start = time.perf_counter()
    verification = IndependentVerifier.verify_mis(instance, final_state)
    t_verif = time.perf_counter() - t_verif_start

    num_groups = len(all_batch_widths)
    avg_width = float(np.mean(all_batch_widths)) if num_groups > 0 else 0.0
    max_width = int(max(all_batch_widths)) if num_groups > 0 else 0
    pe_util = (total_committed / (num_groups * num_pes)) if num_groups > 0 else 0.0
    avg_degree = (2.0 * total_conflicts / total_cands_evaluated) if total_cands_evaluated > 0 else 0.0

    active_stats = pe_array.p2_stats if use_batched_propagation else pe_array.p1_stats
    timings = SolverTimingBreakdown(
        algorithmic_time_seconds=t_algo_accum,
        conflict_construction_time_seconds=t_conflict_accum,
        scheduling_time_seconds=t_sched_accum,
        propagation_time_seconds=t_prop_accum,
        verification_time_seconds=t_verif,
        total_execution_time_seconds=elapsed,
    )

    return SolverResult(
        configuration="D",
        instance_name=instance.name,
        num_variables=instance.num_vertices,
        num_constraints=instance.num_edges,
        num_pes=num_pes,
        seed=seed,
        initial_objective=0,
        final_objective=verification.objective_value,
        is_feasible=verification.is_feasible,
        num_violations=verification.num_violations,
        execution_time_seconds=elapsed,
        time_to_first_feasible_seconds=0.0,
        time_to_target_quality_seconds=elapsed,
        scheduling_latency_seconds=t_sched_accum,
        timings=timings,
        conflict_count=total_conflicts,
        conflict_density=0.0,
        average_conflict_degree=avg_degree,
        num_execution_groups=num_groups,
        avg_batch_width=avg_width,
        max_batch_width=max_width,
        pe_utilization=min(1.0, pe_util),
        communication_messages=active_stats.messages,
        communication_bytes=active_stats.total_bytes,
        communication_header_bytes=active_stats.header_bytes,
        communication_payload_bytes=active_stats.payload_bytes,
        unnecessary_updates=active_stats.unnecessary_updates,
        state_invalidations=active_stats.state_invalidations,
        synchronization_count=active_stats.synchronization_barriers,
        memory_consumption_bytes=psutil.Process().memory_info().rss,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )
