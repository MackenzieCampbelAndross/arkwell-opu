"""ABS V1.0 Milestone 2 Regression Test Suite."""

import pytest
import numpy as np

from arkwell.instances.maximum_independent_set.generator import generate_mis_instance
from arkwell.simulator.compiler.mis_compiler import compile_mis_to_opu
from arkwell.simulator.conflict_analyzer.analyzer import ConflictAnalyzer, CandidateUpdate
from arkwell.simulator.conflict_analyzer.optimized_analyzer import OptimizedConflictAnalyzer
from arkwell.simulator.scheduler.scheduler import SafeConcurrentScheduler
from arkwell.simulator.processing_elements.pe_array import PEArray
from arkwell.backends.cpu.engine import CPUEngine
from arkwell.simulator.state_manager.state_manager import StateManager
from arkwell.simulator.verifier.solution_verifier import IndependentVerifier
from arkwell.simulator.solvers import (
    solve_config_a_sequential,
    solve_config_b_naive_parallel,
    solve_config_c_conflict_aware,
    solve_config_d_selective_propagation,
)


class TestMilestone2Regressions:
    """Milestone 2 regression tests verifying equivalence, safety, message accounting, and consistency."""

    def test_conflict_model_equivalence(self):
        """Verifies that OptimizedConflictAnalyzer exactly matches Reference ConflictAnalyzer."""
        for n in [64, 128]:
            for d in [0.05, 0.20]:
                inst = generate_mis_instance(n, d, seed=42)
                compiled = compile_mis_to_opu(inst)
                ref = ConflictAnalyzer(compiled)
                opt = OptimizedConflictAnalyzer(compiled)

                # Generate arbitrary candidate pool
                cands = [
                    CandidateUpdate(
                        pe_id=i % 4,
                        var_idx=i,
                        current_val=0,
                        proposed_val=1,
                        delta_objective=1.0,
                    )
                    for i in range(n)
                ]
                state = np.zeros(n, dtype=np.int32)

                res_ref = ref.analyze_candidates(cands, state)
                res_opt = opt.analyze_candidates(cands, state, build_type_dict=True)

                assert res_ref.num_conflicts == res_opt.num_conflicts
                assert np.array_equal(res_ref.conflict_matrix, res_opt.conflict_matrix)
                assert res_ref.conflict_density == pytest.approx(res_opt.conflict_density)

                # Check conflict types match
                for pair, c_type in res_ref.conflict_types.items():
                    assert res_opt.conflict_types[pair] == c_type

    def test_batch_safety_invariant(self):
        """Verifies that all scheduled batches are strictly independent sets in the problem graph."""
        inst = generate_mis_instance(128, 0.15, seed=101)
        compiled = compile_mis_to_opu(inst)
        opt = OptimizedConflictAnalyzer(compiled)
        scheduler = SafeConcurrentScheduler()

        cands = [
            CandidateUpdate(pe_id=i % 8, var_idx=i, current_val=0, proposed_val=1, delta_objective=1.0)
            for i in range(128)
        ]
        c_res = opt.analyze_candidates(cands, np.zeros(128, dtype=np.int32), build_type_dict=False)
        plan = scheduler.schedule(c_res)

        edge_set = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in inst.edges)

        for batch in plan.batches:
            batch_vars = [c.var_idx for c in batch]
            for i in range(len(batch_vars)):
                for j in range(i + 1, len(batch_vars)):
                    pair = (min(batch_vars[i], batch_vars[j]), max(batch_vars[i], batch_vars[j]))
                    assert pair not in edge_set, f"Batch safety violated! Edge {pair} found within batch."

    def test_selective_propagation_consistency(self):
        """Verifies that P0 (Broadcast), P1 (Selective), and P2 (Batched Selective) produce identical final states."""
        inst = generate_mis_instance(256, 0.08, seed=777)

        res_p0 = solve_config_c_conflict_aware(inst, num_pes=8, seed=42, use_optimized_analyzer=True)
        res_p1 = solve_config_d_selective_propagation(
            inst, num_pes=8, seed=42, use_optimized_analyzer=True, use_batched_propagation=False
        )
        res_p2 = solve_config_d_selective_propagation(
            inst, num_pes=8, seed=42, use_optimized_analyzer=True, use_batched_propagation=True
        )

        assert res_p0.is_feasible is True
        assert res_p1.is_feasible is True
        assert res_p2.is_feasible is True
        assert np.array_equal(res_p0.final_state, res_p1.final_state)
        assert np.array_equal(res_p1.final_state, res_p2.final_state)
        assert res_p0.final_objective == res_p1.final_objective == res_p2.final_objective

    def test_message_accounting(self):
        """Verifies that header_bytes + payload_bytes == total_bytes across all routing modes."""
        inst = generate_mis_instance(64, 0.10, seed=42)
        compiled = compile_mis_to_opu(inst)
        pe_array = PEArray(num_pes=4, compiled=compiled)

        batch_vars = [0, 5, 12, 20]
        m0 = pe_array.route_p0_broadcast(batch_vars)
        assert m0.header_bytes + m0.payload_bytes == m0.total_bytes
        assert m0.messages == len(batch_vars) * (4 - 1)

        m1 = pe_array.route_p1_selective(batch_vars)
        assert m1.header_bytes + m1.payload_bytes == m1.total_bytes
        assert m1.messages <= m0.messages
        assert m1.unnecessary_updates == 0

        m2 = pe_array.route_p2_batched_selective(batch_vars)
        assert m2.header_bytes + m2.payload_bytes == m2.total_bytes
        assert m2.messages <= pe_array.num_pes
        assert m2.unnecessary_updates == 0

    def test_incremental_vs_full_state_recomputation(self):
        """Verifies that incremental active neighbor count tracking matches full recomputation."""
        inst = generate_mis_instance(128, 0.10, seed=999)
        compiled = compile_mis_to_opu(inst)
        cpu_engine = CPUEngine(compiled)
        state_mgr = StateManager(128)
        engine_state = cpu_engine.initialize_state(state_mgr.get_state())

        flips = [1, 4, 10, 25, 50, 75, 100]
        for v in flips:
            if cpu_engine.is_addition_feasible(engine_state, v):
                cpu_engine.apply_flip(engine_state, v)

                # Ground truth recomputation from scratch
                ground_truth_counts = np.zeros(128, dtype=np.int32)
                for u in range(128):
                    if engine_state.state_vector[u] == 1:
                        nbrs = inst.adjacency_list[u]
                        ground_truth_counts[nbrs] += 1

                assert np.array_equal(engine_state.active_neighbor_counts, ground_truth_counts)

    def test_timing_separation_reporting(self):
        """Verifies that all 6 required timing metrics are non-negative and properly separated."""
        inst = generate_mis_instance(128, 0.05, seed=42)
        res = solve_config_c_conflict_aware(inst, num_pes=4, seed=42, use_optimized_analyzer=True)

        t = res.timings
        assert t.algorithmic_time_seconds >= 0.0
        assert t.conflict_construction_time_seconds >= 0.0
        assert t.scheduling_time_seconds >= 0.0
        assert t.propagation_time_seconds >= 0.0
        assert t.verification_time_seconds >= 0.0
        assert t.total_execution_time_seconds >= 0.0
        assert res.execution_time_seconds == t.total_execution_time_seconds
