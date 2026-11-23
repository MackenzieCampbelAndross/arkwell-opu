"""Compiler translating MIS problem instances into OPU dependency graphs."""

from dataclasses import dataclass
from typing import List
import numpy as np
from arkwell.instances.maximum_independent_set.generator import MISInstance
from arkwell.simulator.objective_graph.graph import ObjectiveGraph
from arkwell.simulator.constraint_graph.graph import ConstraintGraph


@dataclass
class OPUCompiledProblem:
    """Compiled hardware representation consumed by the OPU simulator."""
    instance_name: str
    num_variables: int
    num_constraints: int
    objective_graph: ObjectiveGraph
    constraint_graph: ConstraintGraph
    # Fast adjacency for local search
    adjacency_list: List[np.ndarray]


def compile_mis_to_opu(instance: MISInstance) -> OPUCompiledProblem:
    """
    Compiles an MIS problem instance into OPU dependency graphs.
    """
    n = instance.num_vertices
    m = instance.num_edges

    # 1. Objective graph: maximize sum(x_i)
    # Linear coefficients = 1.0 for all i
    linear_coeff = np.ones(n, dtype=np.float64)
    obj_graph = ObjectiveGraph(
        num_variables=n,
        linear_coefficients=linear_coeff,
        coupling_edges=np.empty((0, 2), dtype=np.int64),
        coupling_weights=np.empty(0, dtype=np.float64),
    )

    # 2. Constraint graph: M pairwise constraints
    var_to_constraints: List[List[int]] = [[] for _ in range(n)]
    for c_idx in range(m):
        u, v = instance.edges[c_idx]
        var_to_constraints[u].append(c_idx)
        var_to_constraints[v].append(c_idx)

    v2c_arrays = [np.array(c_list, dtype=np.int64) for c_list in var_to_constraints]

    con_graph = ConstraintGraph(
        num_variables=n,
        num_constraints=m,
        constraint_edges=instance.edges.copy(),
        var_to_constraints=v2c_arrays,
    )

    return OPUCompiledProblem(
        instance_name=instance.name,
        num_variables=n,
        num_constraints=m,
        objective_graph=obj_graph,
        constraint_graph=con_graph,
        adjacency_list=instance.adjacency_list,
    )
