"""Arkwell OPU Solvers implementing Configurations A, B, C, and D."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import psutil

from arkwell.instances.maximum_independent_set.generator import MISInstance
from arkwell.simulator.compiler.mis_compiler import OPUCompiledProblem, compile_mis_to_opu
from arkwell.simulator.conflict_analyzer.analyzer import ConflictAnalyzer, ConflictType
from arkwell.simulator.scheduler.scheduler import SafeConcurrentScheduler
from arkwell.simulator.processing_elements.pe_array import PEArray
from arkwell.simulator.state_manager.state_manager import StateManager
from arkwell.simulator.verifier.solution_verifier import IndependentVerifier
from arkwell.backends.cpu.engine import CPUEngine


@dataclass
class SolverResult:
    """Standardized result schema adhering to Section 8 benchmark metrics."""
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
    num_execution_groups: int
    avg_batch_width: float
    max_batch_width: int
    pe_utilization: float
    communication_messages: int
    communication_bytes: int
    synchronization_count: int
    memory_consumption_bytes: int
    gpu_memory_consumption_bytes: int

    # State
    final_state: np.ndarray

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
            "time_to_first_feasible_seconds": self.time_to_first_feasible_seconds,
            "time_to_target_quality_seconds": self.time_to_target_quality_seconds,
            "scheduling_latency_seconds": self.scheduling_latency_seconds,
            "conflict_count": self.conflict_count,
            "conflict_density": self.conflict_density,
            "num_execution_groups": self.num_execution_groups,
            "avg_batch_width": self.avg_batch_width,
            "max_batch_width": self.max_batch_width,
            "pe_utilization": self.pe_utilization,
            "communication_messages": self.communication_messages,
            "communication_bytes": self.communication_bytes,
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
    """
    Configuration A: Sequential Local Search.
    Single PE reference solver.
    """
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())

    start_time = time.perf_counter()
    time_to_feasible = 0.0

    rng = np.random.default_rng(seed)
    n = instance.num_vertices
    order = np.arange(n)
    if random_order:
        rng.shuffle(order)

    # Greedy maximal independent set construction
    for v in order:
        if cpu_engine.is_addition_feasible(engine_state, int(v)):
            cpu_engine.apply_flip(engine_state, int(v))
            state_mgr.commit_batch([int(v)])

    # Optional local 2-improvement / neighborhood passes can be applied here
    # For baseline reference, maximal independent set is guaranteed maximal

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()

    # Independent verification
    verification = IndependentVerifier.verify_mis(instance, final_state)

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
        time_to_first_feasible_seconds=time_to_feasible,
        time_to_target_quality_seconds=elapsed,
        scheduling_latency_seconds=0.0,
        conflict_count=0,
        conflict_density=0.0,
        num_execution_groups=verification.objective_value,
        avg_batch_width=1.0,
        max_batch_width=1,
        pe_utilization=1.0,
        communication_messages=0,
        communication_bytes=0,
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
    """
    Configuration B: Naive Parallel Local Search.
    Multiple PEs evaluate candidate moves on snapshot state.
    Commits are serialized or filtered with naive lock-free validation to preserve correctness.
    Uses global broadcast interconnect.
    """
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())
    pe_array = PEArray(num_pes, compiled)

    start_time = time.perf_counter()
    round_count = 0
    total_committed = 0
    total_proposed = 0
    rejected_moves = 0

    while True:
        round_count += 1
        # PEs inspect state and propose moves
        candidates = pe_array.generate_candidate_moves(
            engine_state.state_vector,
            engine_state.active_neighbor_counts,
        )
        if len(candidates) == 0:
            break

        total_proposed += len(candidates)
        # Naive parallel commit protocol:
        # PEs attempt to commit in parallel. To maintain correctness without conflict graph,
        # an arbiter checks each candidate in arrival order against live state.
        committed_this_round: List[int] = []
        for cand in candidates:
            v = cand.var_idx
            if cpu_engine.is_addition_feasible(engine_state, v):
                cpu_engine.apply_flip(engine_state, v)
                committed_this_round.append(v)
            else:
                rejected_moves += 1

        if len(committed_this_round) > 0:
            state_mgr.commit_batch(committed_this_round)
            # Global broadcast to all PEs
            pe_array.route_broadcast_updates(committed_this_round)
            total_committed += len(committed_this_round)

        pe_array.stats.synchronization_barriers += 1

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()
    verification = IndependentVerifier.verify_mis(instance, final_state)

    pe_util = (total_committed / (round_count * num_pes)) if round_count > 0 else 0.0

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
        conflict_count=rejected_moves,
        conflict_density=0.0,
        num_execution_groups=round_count,
        avg_batch_width=(total_committed / round_count) if round_count > 0 else 0.0,
        max_batch_width=num_pes,
        pe_utilization=min(1.0, pe_util),
        communication_messages=pe_array.stats.broadcast_messages,
        communication_bytes=pe_array.stats.broadcast_bytes,
        synchronization_count=pe_array.stats.synchronization_barriers,
        memory_consumption_bytes=psutil.Process().memory_info().rss,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )


def solve_config_c_conflict_aware(
    instance: MISInstance,
    num_pes: int = 4,
    seed: int = 42,
) -> SolverResult:
    """
    Configuration C: Conflict-Aware Scheduling.
    Constructs hardware conflict graph over candidate updates.
    Schedules safe concurrent batches.
    Uses global broadcast interconnect.
    """
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())
    pe_array = PEArray(num_pes, compiled)
    analyzer = ConflictAnalyzer(compiled)
    scheduler = SafeConcurrentScheduler()

    start_time = time.perf_counter()
    total_conflicts = 0
    total_scheduling_time = 0.0
    all_batch_widths: List[int] = []
    total_committed = 0
    rounds = 0

    while True:
        rounds += 1
        candidates = pe_array.generate_candidate_moves(
            engine_state.state_vector,
            engine_state.active_neighbor_counts,
        )
        if len(candidates) == 0:
            break

        # 1. Conflict analysis
        c_res = analyzer.analyze_candidates(candidates, engine_state.state_vector)
        total_conflicts += c_res.num_conflicts

        # 2. Scheduling into safe batches
        plan = scheduler.schedule(c_res)
        total_scheduling_time += plan.scheduling_time_seconds

        # 3. Commit the primary maximal safe concurrent batch for this cycle
        if len(plan.batches) > 0:
            batch = plan.batches[0]
            batch_vars = [cand.var_idx for cand in batch]

            # Apply all batch flips simultaneously
            cpu_engine.apply_batch_flips(engine_state, batch_vars)
            state_mgr.commit_batch(batch_vars)

            # Route via global broadcast
            pe_array.route_broadcast_updates(batch_vars)
            pe_array.stats.synchronization_barriers += 1

            all_batch_widths.append(len(batch_vars))
            total_committed += len(batch_vars)

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()
    verification = IndependentVerifier.verify_mis(instance, final_state)

    num_groups = len(all_batch_widths)
    avg_width = float(np.mean(all_batch_widths)) if num_groups > 0 else 0.0
    max_width = int(max(all_batch_widths)) if num_groups > 0 else 0
    pe_util = (total_committed / (num_groups * num_pes)) if num_groups > 0 else 0.0

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
        scheduling_latency_seconds=total_scheduling_time,
        conflict_count=total_conflicts,
        conflict_density=0.0,
        num_execution_groups=num_groups,
        avg_batch_width=avg_width,
        max_batch_width=max_width,
        pe_utilization=min(1.0, pe_util),
        communication_messages=pe_array.stats.broadcast_messages,
        communication_bytes=pe_array.stats.broadcast_bytes,
        synchronization_count=pe_array.stats.synchronization_barriers,
        memory_consumption_bytes=psutil.Process().memory_info().rss,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )


def solve_config_d_selective_propagation(
    instance: MISInstance,
    num_pes: int = 4,
    seed: int = 42,
) -> SolverResult:
    """
    Configuration D: Conflict-Aware Scheduling with Selective Propagation.
    Safe concurrent batches committed with selective state multicast.
    """
    compiled = compile_mis_to_opu(instance)
    cpu_engine = CPUEngine(compiled)
    state_mgr = StateManager(instance.num_vertices)
    engine_state = cpu_engine.initialize_state(state_mgr.get_state())
    pe_array = PEArray(num_pes, compiled)
    analyzer = ConflictAnalyzer(compiled)
    scheduler = SafeConcurrentScheduler()

    start_time = time.perf_counter()
    total_conflicts = 0
    total_scheduling_time = 0.0
    all_batch_widths: List[int] = []
    total_committed = 0
    rounds = 0

    while True:
        rounds += 1
        candidates = pe_array.generate_candidate_moves(
            engine_state.state_vector,
            engine_state.active_neighbor_counts,
        )
        if len(candidates) == 0:
            break

        c_res = analyzer.analyze_candidates(candidates, engine_state.state_vector)
        total_conflicts += c_res.num_conflicts

        plan = scheduler.schedule(c_res)
        total_scheduling_time += plan.scheduling_time_seconds

        # Commit the primary maximal safe concurrent batch for this cycle
        if len(plan.batches) > 0:
            batch = plan.batches[0]
            batch_vars = [cand.var_idx for cand in batch]

            cpu_engine.apply_batch_flips(engine_state, batch_vars)
            state_mgr.commit_batch(batch_vars)

            # Selective multicast to subscribers only
            pe_array.route_selective_updates(batch_vars)
            pe_array.stats.synchronization_barriers += 1

            all_batch_widths.append(len(batch_vars))
            total_committed += len(batch_vars)

    elapsed = time.perf_counter() - start_time
    final_state = state_mgr.get_state()
    verification = IndependentVerifier.verify_mis(instance, final_state)

    num_groups = len(all_batch_widths)
    avg_width = float(np.mean(all_batch_widths)) if num_groups > 0 else 0.0
    max_width = int(max(all_batch_widths)) if num_groups > 0 else 0
    pe_util = (total_committed / (num_groups * num_pes)) if num_groups > 0 else 0.0

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
        scheduling_latency_seconds=total_scheduling_time,
        conflict_count=total_conflicts,
        conflict_density=0.0,
        num_execution_groups=num_groups,
        avg_batch_width=avg_width,
        max_batch_width=max_width,
        pe_utilization=min(1.0, pe_util),
        communication_messages=pe_array.stats.selective_messages,
        communication_bytes=pe_array.stats.selective_bytes,
        synchronization_count=pe_array.stats.synchronization_barriers,
        memory_consumption_bytes=psutil.Process().memory_info().rss,
        gpu_memory_consumption_bytes=0,
        final_state=final_state,
    )
