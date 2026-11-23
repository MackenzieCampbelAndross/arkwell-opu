"""Reproducible sparse graph generator for Maximum Independent Set (MIS) instances."""

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MISInstance:
    """Represents a Maximum Independent Set benchmark problem instance."""
    name: str
    num_vertices: int
    target_density: float
    seed: int
    num_edges: int
    edges: np.ndarray  # Shape (M, 2), int64, sorted with u < v
    adjacency_list: List[np.ndarray]  # Length N, list of 1D arrays of neighbor indices
    degrees: np.ndarray  # Shape (N,), int32

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "num_vertices": self.num_vertices,
            "target_density": self.target_density,
            "seed": self.seed,
            "num_edges": self.num_edges,
            "edges": self.edges.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MISInstance":
        n = data["num_vertices"]
        edges = np.array(data["edges"], dtype=np.int64)
        if len(edges) == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        
        # Reconstruct adjacency list and degrees
        adj: List[List[int]] = [[] for _ in range(n)]
        degrees = np.zeros(n, dtype=np.int32)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degrees[u] += 1
            degrees[v] += 1
        adj_arrays = [np.array(sorted(nbrs), dtype=np.int64) for nbrs in adj]

        return cls(
            name=data["name"],
            num_vertices=n,
            target_density=data["target_density"],
            seed=data["seed"],
            num_edges=len(edges),
            edges=edges,
            adjacency_list=adj_arrays,
            degrees=degrees,
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MISInstance":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_mis_instance(num_vertices: int, density: float, seed: int) -> MISInstance:
    """
    Generates an Erdős-Rényi G(N, p) graph using deterministic sparse sampling.
    
    Guarantees:
    - Exactly deterministic given (num_vertices, density, seed).
    - Memory efficient: never constructs full N x N dense matrix.
    - Edges are undirected, without self-loops or duplicate edges.
    """
    if num_vertices <= 0:
        raise ValueError("num_vertices must be positive")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0.0, 1.0]")

    rng = np.random.default_rng(seed)
    n = num_vertices
    total_possible_edges = n * (n - 1) // 2

    if total_possible_edges == 0 or density == 0.0:
        edges = np.empty((0, 2), dtype=np.int64)
        adj_arrays = [np.empty(0, dtype=np.int64) for _ in range(n)]
        degrees = np.zeros(n, dtype=np.int32)
        name = f"mis_n{n}_d{density:.2f}_s{seed}"
        return MISInstance(name, n, density, seed, 0, edges, adj_arrays, degrees)

    # Use binomial sampling to determine exact edge count
    expected_m = int(round(total_possible_edges * density))

    if expected_m >= total_possible_edges:
        # Complete graph
        u_indices, v_indices = np.triu_indices(n, k=1)
        edges = np.column_stack((u_indices, v_indices)).astype(np.int64)
    elif density < 0.20:
        # Sparse sampling with replacement + deduplication
        # We sample indices from range(0, total_possible_edges)
        sampled_indices = set()
        while len(sampled_indices) < expected_m:
            batch_size = max(1000, expected_m - len(sampled_indices))
            candidates = rng.integers(0, total_possible_edges, size=batch_size, dtype=np.int64)
            for idx in candidates:
                sampled_indices.add(int(idx))
                if len(sampled_indices) >= expected_m:
                    break
        
        flat_indices = np.array(sorted(sampled_indices), dtype=np.int64)
        
        # Map flat upper triangular index k in [0, total_possible_edges) to (u, v)
        # Using analytical mapping:
        # row u has n-1-u elements.
        # u = n - 2 - floor(sqrt(4n(n-1) - 7 - 8k)/2 - 0.5)
        # We can compute u vectorially:
        b = 2 * n - 1
        u_arr = np.floor((b - np.sqrt(b * b - 8 * flat_indices)) / 2).astype(np.int64)
        row_starts = u_arr * (2 * n - 1 - u_arr) // 2
        v_arr = u_arr + 1 + (flat_indices - row_starts)
        edges = np.column_stack((u_arr, v_arr)).astype(np.int64)
    else:
        # For moderate/high density, sample upper triangular entries blockwise or via permutation
        flat_indices = rng.choice(total_possible_edges, size=expected_m, replace=False)
        flat_indices.sort()
        b = 2 * n - 1
        u_arr = np.floor((b - np.sqrt(b * b - 8 * flat_indices)) / 2).astype(np.int64)
        row_starts = u_arr * (2 * n - 1 - u_arr) // 2
        v_arr = u_arr + 1 + (flat_indices - row_starts)
        edges = np.column_stack((u_arr, v_arr)).astype(np.int64)

    # Build adjacency list
    adj: List[List[int]] = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        adj[u].append(int(v))
        adj[v].append(int(u))
        degrees[u] += 1
        degrees[v] += 1

    adj_arrays = [np.array(sorted(nbrs), dtype=np.int64) for nbrs in adj]
    name = f"mis_n{n}_d{density:.2f}_s{seed}"

    return MISInstance(
        name=name,
        num_vertices=n,
        target_density=density,
        seed=seed,
        num_edges=len(edges),
        edges=edges,
        adjacency_list=adj_arrays,
        degrees=degrees,
    )
