"""Automated experiment runner for the Arkwell Benchmark Suite (ABS V1.0)."""

import csv
import json
import os
import subprocess
import time
from typing import Dict, List, Optional
import yaml
import numpy as np

from arkwell.backends.hardware_info import detect_hardware
from arkwell.instances.maximum_independent_set.generator import generate_mis_instance
from arkwell.simulator.solvers import (
    SolverResult,
    solve_config_a_sequential,
    solve_config_b_naive_parallel,
    solve_config_c_conflict_aware,
    solve_config_d_selective_propagation,
)
from arkwell.simulator.verifier.solution_verifier import IndependentVerifier


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a repo."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted_initial"


def run_single_experiment(
    config_name: str,
    n: int,
    density: float,
    seed: int,
    num_pes: int,
) -> SolverResult:
    """Executes a single benchmark run."""
    instance = generate_mis_instance(n, density, seed)

    if config_name == "A":
        return solve_config_a_sequential(instance, seed=seed)
    elif config_name == "B":
        return solve_config_b_naive_parallel(instance, num_pes=num_pes, seed=seed)
    elif config_name == "C":
        return solve_config_c_conflict_aware(instance, num_pes=num_pes, seed=seed)
    elif config_name == "D":
        return solve_config_d_selective_propagation(instance, num_pes=num_pes, seed=seed)
    else:
        raise ValueError(f"Unknown configuration: {config_name}")


def run_experiments_from_config(config_path: str, quick_mode: bool = False) -> List[Dict]:
    """Runs benchmark sweep as specified in YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hardware = detect_hardware()
    git_hash = get_git_commit_hash()

    sizes = cfg["problem_sizes"]
    densities = cfg["densities"]
    seeds = cfg["random_seeds"]
    configurations = cfg["configurations"]
    pe_list = cfg["processing_elements"]

    if quick_mode:
        # Screening sample for quick execution and initial report
        sizes = [128, 256]
        densities = [0.02, 0.10]
        seeds = seeds[:2]
        pe_list = [1, 4, 16]

    results: List[Dict] = []
    os.makedirs(os.path.dirname(cfg["outputs"]["results_csv"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["outputs"]["results_json"]), exist_ok=True)
    os.makedirs(cfg["outputs"]["logs_dir"], exist_ok=True)

    log_path = os.path.join(cfg["outputs"]["logs_dir"], f"run_{int(time.time())}.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"ARKWELL ABS V1.0 EXPERIMENT RUN\n")
        log_file.write(f"Git Commit: {git_hash}\n")
        log_file.write(f"Hardware: {hardware}\n")
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n")

        for n in sizes:
            for d in densities:
                for s in seeds:
                    for conf in configurations:
                        # For Config A, only 1 PE is meaningful
                        active_pes = [1] if conf == "A" else pe_list
                        for pes in active_pes:
                            res = run_single_experiment(conf, n, d, s, pes)
                            # Rigorous verification
                            if cfg.get("verify_every_solution", True):
                                assert res.is_feasible, f"Infeasible solution detected in {conf}!"

                            entry = res.to_dict()
                            entry["git_commit"] = git_hash
                            entry["density"] = d
                            results.append(entry)

                            log_file.write(
                                f"[{conf}] N={n:4d} d={d:.2f} s={s:4d} PEs={pes:2d} | "
                                f"Obj={res.final_objective:3d} Feasible={res.is_feasible} "
                                f"Time={res.execution_time_seconds*1000:.2f}ms "
                                f"Msgs={res.communication_messages} Bytes={res.communication_bytes}\n"
                            )

    # Save to JSON
    with open(cfg["outputs"]["results_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "git_commit": git_hash,
                "hardware": hardware.__dict__,
                "results": results,
            },
            f,
            indent=2,
        )

    # Save to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(cfg["outputs"]["results_csv"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Executed {len(results)} experiment runs successfully.")
    print(f"Results saved to: {cfg['outputs']['results_csv']} and {cfg['outputs']['results_json']}")
    return results


if __name__ == "__main__":
    run_experiments_from_config("arkwell/experiments/configurations/mis_v1.yaml", quick_mode=True)
