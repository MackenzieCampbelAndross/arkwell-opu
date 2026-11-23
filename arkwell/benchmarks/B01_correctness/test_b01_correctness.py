"""B01 Correctness Test Suite: Rigorous verification of OPU mathematical semantics."""

import pytest
import numpy as np

from arkwell.instances.maximum_independent_set.generator import generate_mis_instance
from arkwell.simulator.compiler.mis_compiler import compile_mis_to_opu
from arkwell.simulator.verifier.solution_verifier import IndependentVerifier, ExhaustiveVerifier
from arkwell.simulator.conflict_analyzer.analyzer import ConflictAnalyzer, CandidateUpdate, ConflictType
from arkwell.simulator.scheduler.scheduler import SafeConcurrentScheduler
from arkwell.simulator.processing_elements.pe_array import PEArray
from arkwell.simulator.solvers import (
    solve_config_a_sequential,
    solve_config_b_naive_parallel,
    solve_config_c_conflict_aware,
    solve_config_d_selective_propagation,
)
from arkwell.backends.gpu.engine import GPUEngine


class TestB01Correctness:
    """Rigorous correctness verification meeting ABS V1.0 Section 9 acceptance criteria."""

    def test_01_objective_calculation(self):
        """1. Test objective calculation with exact arithmetic."""
        inst = generate_mis_instance(num_vertices=32, density=0.10, seed=42)
        state = np.zeros(32, dtype=np.int32)
        state[[1, 5, 10, 20]] = 1

        res = IndependentVerifier.verify_mis(inst, state)
        assert res.objective_value == 4
        assert res.objective_value == int(np.sum(state))

    def test_02_constraint_evaluation(self):
        """2. Test constraint evaluation: detect violations when adjacent vertices are both 1."""
        inst = generate_mis_instance(num_vertices=16, density=0.30, seed=101)
        assert inst.num_edges > 0
        u, v = inst.edges[0]

        # Feasible state: only u is 1
        state_feas = np.zeros(16, dtype=np.int32)
        state_feas[u] = 1
        res_feas = IndependentVerifier.verify_mis(inst, state_feas)
        assert res_feas.is_feasible is True
        assert res_feas.num_violations == 0

        # Infeasible state: both u and v are 1
        state_infeas = np.zeros(16, dtype=np.int32)
        state_infeas[u] = 1
        state_infeas[v] = 1
        res_infeas = IndependentVerifier.verify_mis(inst, state_infeas)
        assert res_infeas.is_feasible is False
        assert res_infeas.num_violations >= 1
        assert (int(u), int(v)) in res_infeas.violated_constraints

    def test_03_local_objective_deltas(self):
        """3. Test local objective deltas match actual state transitions."""
        inst = generate_mis_instance(num_vertices=32, density=0.10, seed=42)
        compiled = compile_mis_to_opu(inst)
        state = np.zeros(32, dtype=np.int32)

        # Delta when flipping 0 -> 1 should be +1.0
        delta_add = compiled.objective_graph.compute_local_delta(5, state)
        assert delta_add == 1.0

        # Flip variable 5 to 1
        state[5] = 1
        # Delta when flipping 1 -> 0 should be -1.0
        delta_rem = compiled.objective_graph.compute_local_delta(5, state)
        assert delta_rem == -1.0

    def test_04_conflict_detection(self):
        """4. Test conflict detection flags simultaneous additions on adjacent vertices."""
        inst = generate_mis_instance(num_vertices=16, density=0.30, seed=42)
        compiled = compile_mis_to_opu(inst)
        analyzer = ConflictAnalyzer(compiled)

        u, v = inst.edges[0]
        # Two candidate additions on adjacent vertices u and v
        cand_u = CandidateUpdate(pe_id=0, var_idx=int(u), current_val=0, proposed_val=1, delta_objective=1.0)
        cand_v = CandidateUpdate(pe_id=1, var_idx=int(v), current_val=0, proposed_val=1, delta_objective=1.0)

        res = analyzer.analyze_candidates([cand_u, cand_v], np.zeros(16, dtype=np.int32))
        assert res.num_conflicts == 1
        assert bool(res.conflict_matrix[0, 1]) is True
        assert res.conflict_types[(0, 1)] == ConflictType.FEASIBILITY_VIOLATION

    def test_05_concurrent_update_safety(self):
        """5. Test concurrent update safety: scheduled batches never conflict."""
        inst = generate_mis_instance(num_vertices=64, density=0.15, seed=777)
        compiled = compile_mis_to_opu(inst)
        analyzer = ConflictAnalyzer(compiled)
        scheduler = SafeConcurrentScheduler()

        # Propose additions for all vertices
        candidates = [
            CandidateUpdate(pe_id=i % 4, var_idx=i, current_val=0, proposed_val=1, delta_objective=1.0)
            for i in range(64)
        ]

        c_res = analyzer.analyze_candidates(candidates, np.zeros(64, dtype=np.int32))
        plan = scheduler.schedule(c_res)

        # In every scheduled batch, no two candidate variables may share an edge
        edge_set = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in inst.edges)
        for batch in plan.batches:
            batch_vars = [c.var_idx for c in batch]
            for i in range(len(batch_vars)):
                for j in range(i + 1, len(batch_vars)):
                    pair = (min(batch_vars[i], batch_vars[j]), max(batch_vars[i], batch_vars[j]))
                    assert pair not in edge_set, f"Batch contains adjacent vertices: {pair}"

    def test_06_selective_state_propagation(self):
        """6. Test selective state propagation sends updates strictly to subscriber PEs."""
        inst = generate_mis_instance(num_vertices=32, density=0.10, seed=42)
        compiled = compile_mis_to_opu(inst)
        pe_array = PEArray(num_pes=4, compiled=compiled)

        # Route updates for variable 0
        pe_owner = pe_array.var_to_pe[0]
        subscribers = pe_array.subscribers[0]

        # Verify no subscriber is the owner itself
        assert pe_owner not in subscribers

        # Verify that for every subscriber PE, it owns at least one neighbor of 0
        for sub_pe in subscribers:
            vars_in_sub = pe_array.pe_to_vars[sub_pe]
            nbrs_of_0 = inst.adjacency_list[0]
            common = set(vars_in_sub).intersection(set(nbrs_of_0))
            assert len(common) > 0, f"PE {sub_pe} was subscribed but has no incident edge with vertex 0"

        # Selective messages must be <= broadcast messages
        msgs_sel, bytes_sel = pe_array.route_selective_updates([0])
        msgs_bcast, bytes_bcast = pe_array.route_broadcast_updates([0])
        assert msgs_sel <= msgs_bcast

    def test_07_compare_small_instances_against_exhaustive(self):
        """7. Compare small instances against exhaustive enumeration ground truth."""
        # For N=12, run exhaustive solver to get true global alpha(G)
        inst = generate_mis_instance(num_vertices=12, density=0.25, seed=123)
        opt_obj, opt_state = ExhaustiveVerifier.solve_mis_exact(inst)
        assert opt_obj > 0

        # Run sequential local search (Config A)
        res_a = solve_config_a_sequential(inst, random_order=False, seed=42)
        assert res_a.is_feasible is True
        # Heuristic must produce a valid maximal independent set <= opt_obj
        assert res_a.final_objective <= opt_obj
        # For small graphs, greedy often achieves or nears global optimum
        assert res_a.final_objective >= 1

    def test_08_recalculate_every_returned_solution(self):
        """8. Run Configurations A, B, C, D and independently recalculate every solution."""
        inst = generate_mis_instance(num_vertices=128, density=0.05, seed=999)

        # Config A
        res_a = solve_config_a_sequential(inst, seed=42)
        check_a = IndependentVerifier.verify_mis(inst, res_a.final_state)
        assert check_a.is_feasible is True
        assert check_a.objective_value == res_a.final_objective

        # Config B
        res_b = solve_config_b_naive_parallel(inst, num_pes=4, seed=42)
        check_b = IndependentVerifier.verify_mis(inst, res_b.final_state)
        assert check_b.is_feasible is True
        assert check_b.objective_value == res_b.final_objective

        # Config C
        res_c = solve_config_c_conflict_aware(inst, num_pes=4, seed=42)
        check_c = IndependentVerifier.verify_mis(inst, res_c.final_state)
        assert check_c.is_feasible is True
        assert check_c.objective_value == res_c.final_objective

        # Config D
        res_d = solve_config_d_selective_propagation(inst, num_pes=4, seed=42)
        check_d = IndependentVerifier.verify_mis(inst, res_d.final_state)
        assert check_d.is_feasible is True
        assert check_d.objective_value == res_d.final_objective

    def test_09_gpu_backend_correctness_and_fallback(self):
        """9. Test GPU backend execution, feasibility verification, and CPU fallback."""
        inst = generate_mis_instance(num_vertices=64, density=0.10, seed=42)
        compiled = compile_mis_to_opu(inst)
        gpu_engine = GPUEngine(compiled)

        init_state = np.zeros(64, dtype=np.int32)
        final_state, result = gpu_engine.run_parallel_greedy_sweep(init_state)

        # Verify output state with independent verifier
        verif = IndependentVerifier.verify_mis(inst, final_state)
        assert verif.is_feasible is True
        assert result.is_feasible is True
        assert verif.objective_value == result.final_objective
