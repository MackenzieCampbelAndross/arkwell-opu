"""Milestone 2 Paired Benchmark Runner for ABS V1.0 B02 and B03 experiments."""

import csv
import json
import os
import subprocess
import time
from typing import Dict, List
import numpy as np

from arkwell.backends.hardware_info import detect_hardware
from arkwell.instances.maximum_independent_set.generator import generate_mis_instance
from arkwell.simulator.solvers import (
    solve_config_b_naive_parallel,
    solve_config_c_conflict_aware,
    solve_config_d_selective_propagation,
)
from arkwell.benchmarks.B02_conflict_scheduling.benchmark_b02 import run_b02_benchmark
from arkwell.benchmarks.B03_constraint_propagation.benchmark_b03 import run_b03_benchmark


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_milestone2_suite():
    """Executes B02, B03, and paired screening benchmarks."""
    print("=" * 60)
    print("ARKWELL ABS V1.0 MILESTONE 2: BENCHMARK SUITE EXECUTION")
    print("=" * 60)

    git_hash = get_git_commit()
    hw = detect_hardware()
    os.makedirs("arkwell/experiments/raw_results", exist_ok=True)
    os.makedirs("arkwell/experiments/logs", exist_ok=True)

    # 1. Run B02 Conflict Scheduling Benchmark
    print("\n--- Running B02: Conflict Scheduling Efficiency Benchmark ---")
    b02_records = run_b02_benchmark(
        problem_sizes=[512, 1024],
        densities=[0.02, 0.10, 0.30],
        seeds=[42, 101, 2024],
        pe_counts=[4, 8, 16],
    )
    b02_path = "arkwell/experiments/raw_results/b02_conflict_results.json"
    with open(b02_path, "w", encoding="utf-8") as f:
        json.dump(b02_records, f, indent=2)
    print(f"B02 Complete: {len(b02_records)} data points saved to {b02_path}")

    # 2. Run B03 Constraint Propagation Traffic Benchmark (P0 vs P1 vs P2)
    print("\n--- Running B03: Constraint Propagation Traffic Benchmark ---")
    b03_records = run_b03_benchmark(
        problem_sizes=[512, 1024],
        densities=[0.02, 0.10, 0.30],
        seeds=[42, 101, 2024],
        pe_counts=[4, 8, 16],
    )
    b03_path = "arkwell/experiments/raw_results/b03_propagation_results.json"
    with open(b03_path, "w", encoding="utf-8") as f:
        json.dump(b03_records, f, indent=2)
    print(f"B03 Complete: {len(b03_records)} data points saved to {b03_path}")

    # 3. Paired Configuration Screening (B vs C vs D) across Problem Sizes (512, 1024, 2048)
    print("\n--- Running Paired Configurations Sweep (B, C, D) ---")
    paired_results: List[Dict] = []
    sizes = [512, 1024, 2048]
    densities = [0.02, 0.10]
    seeds = [42, 101, 2024, 777, 999]  # 5 screening seeds
    pe_counts = [4, 16]

    for n in sizes:
        for d in densities:
            for s in seeds:
                inst = generate_mis_instance(n, d, s)
                for pes in pe_counts:
                    # Paired execution on identical instance
                    res_b = solve_config_b_naive_parallel(inst, num_pes=pes, seed=s)
                    res_c = solve_config_c_conflict_aware(inst, num_pes=pes, seed=s, use_optimized_analyzer=True)
                    res_d = solve_config_d_selective_propagation(
                        inst, num_pes=pes, seed=s, use_optimized_analyzer=True, use_batched_propagation=True
                    )

                    for res in [res_b, res_c, res_d]:
                        d_entry = res.to_dict()
                        d_entry["git_commit"] = git_hash
                        d_entry["density"] = d
                        paired_results.append(d_entry)

                    print(
                        f"N={n:4d} d={d:.2f} s={s:4d} PEs={pes:2d} | "
                        f"Obj(B/C/D)=({res_b.final_objective}/{res_c.final_objective}/{res_d.final_objective}) | "
                        f"Time(B/C/D)=({res_b.execution_time_seconds*1000:.1f}ms/{res_c.execution_time_seconds*1000:.1f}ms/{res_d.execution_time_seconds*1000:.1f}ms) | "
                        f"Msgs(P0/P2)=({res_c.communication_messages}/{res_d.communication_messages})"
                    )

    paired_json_path = "arkwell/experiments/raw_results/milestone2_paired_results.json"
    with open(paired_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "git_commit": git_hash,
                "hardware": hw.__dict__,
                "results": paired_results,
            },
            f,
            indent=2,
        )

    paired_csv_path = "arkwell/experiments/raw_results/milestone2_paired_results.csv"
    if paired_results:
        with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_results[0].keys()))
            writer.writeheader()
            writer.writerows(paired_results)

    print(f"\nPaired Sweep Complete: {len(paired_results)} data points saved to {paired_csv_path}")


if __name__ == "__main__":
    run_milestone2_suite()
