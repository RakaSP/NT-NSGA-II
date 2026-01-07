#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from vrp_core import load_config, load_problem_from_config, ensure_dir
from vrp_core.cluster_azam import make_azam_subproblems
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine

CONFIG_PATH = "solver_config.yaml"
RUNS_PER_ALGO = 36

# ---------------------------------------------------------------------
# Define your "run configurations" here.
# Each entry defines ONE algorithm setup.
#
# For NSGA2:
#   - "crossover_rate" and "mutation_rate" inside "overrides"
#     will override the YAML's algo_params.nsga2.{crossover_rate, mutation_rate}
# ---------------------------------------------------------------------
RUN_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "nsga2_rl",
        "overrides": {
            "algorithm": "nsga2",
            "rl_enabled": True,
            "training_enabled": False,
        },
    },
]



def run_one_experiment(
    base_config: Dict[str, Any],
    run_index: int,
) -> List[Dict[str, Any]]:
    """
    Run ONE experiment index across ALL selected algorithms.

    For a given run_index:
      - Load the VRP once from base_config.
      - Build ONE set of subproblems (sub_vrps) using a single seed.
      - For each algorithm in RUN_CONFIGS:
          * create a config (base_config + algo overrides)
          * set its own output directory
          * solve the SAME sub_vrps
          * collect aggregated results

    Returns:
      List of aggregated summaries (one dict per algorithm).
    """
    all_results_for_run: List[Dict[str, Any]] = []

    # Base output directory from config (e.g. "results")
    base_output_root = Path(base_config.get("output_dir", "results")).resolve()
    ensure_dir(str(base_output_root))

    log_info("=== Global run %d/%d (all algorithms) ===", run_index + 1, RUNS_PER_ALGO)
    log_info("Base output root: %s", base_output_root)

    # ------------------------------------------------------------------
    # 1) Load the FULL problem ONCE for this run (independent of algo)
    # ------------------------------------------------------------------
    vrp_config_for_loading = copy.deepcopy(base_config)
    full_vrp = load_problem_from_config(vrp_config_for_loading)

    # ------------------------------------------------------------------
    # 2) Build subproblems ONCE (same clustering for all algorithms)
    # ------------------------------------------------------------------
    sub_vrps: List[Dict[str, Any]] = make_azam_subproblems(
        full_vrp,
        depot_id=0,
        num_depots=1,
        num_vehicles=len(full_vrp["vehicles"]),
        seed=random.randint(1, 1000000),
        capacity_per_cluster=base_config.get("cluster_capacity", None),
    )

    log_info("Built %d subproblems for run %d", len(sub_vrps), run_index + 1)

    # ------------------------------------------------------------------
    # 3) For each algorithm config, solve the SAME sub_vrps
    # ------------------------------------------------------------------
    for algo_config in RUN_CONFIGS:
        algorithm_name = algo_config["name"]
        algo_overrides: Dict[str, Any] = algo_config.get("overrides", {})

        # Start from base config and apply algorithm-specific overrides
        solver_config = copy.deepcopy(base_config)
        solver_config.update(algo_overrides)

        # ------------------------------------------------------------------
        # NSGA2-specific: allow RUN_CONFIGS to override cx/mut from YAML
        # ------------------------------------------------------------------
        algo_id = solver_config.get("algorithm")

        if algo_id == "nsga2":
            cx_override = algo_overrides.get("crossover_rate", None)
            mut_override = algo_overrides.get("mutation_rate", None)

            if cx_override is not None or mut_override is not None:
                # Make sure nested structure exists
                solver_config.setdefault("algo_params", {})
                solver_config["algo_params"].setdefault("nsga2", {})

                if cx_override is not None:
                    solver_config["algo_params"]["nsga2"]["crossover_rate"] = float(cx_override)
                if mut_override is not None:
                    solver_config["algo_params"]["nsga2"]["mutation_rate"] = float(mut_override)

                # These top-level keys are only for convenience; NSGA2
                # reads from algo_params.nsga2, so we can drop them.
                solver_config.pop("crossover_rate", None)
                solver_config.pop("mutation_rate", None)

        # Per-(algorithm, run_index) output dir
        algo_output_dir = base_output_root / f"{algorithm_name}_run{run_index + 1}"
        solver_config["output_dir"] = str(algo_output_dir)
        ensure_dir(solver_config["output_dir"])

        log_info(
            "--- Algorithm %s | run %d/%d ---",
            algorithm_name,
            run_index + 1,
            RUNS_PER_ALGO,
        )
        log_info("Output dir: %s", solver_config["output_dir"])
        log_info("Algorithm ID: %s", solver_config.get("algorithm"))

        # Create engine and solve ALL clusters at once using SAME sub_vrps
        engine = VRPSolverEngine(solver_config)
        engine.prepare(sub_vrps)
        agg = engine.run()

        # Attach metadata
        agg["algorithm_name"] = algorithm_name
        agg["algorithm_id"] = solver_config.get("algorithm")
        # Keep old key for compatibility if something else expects it
        agg["config_name"] = algorithm_name

        agg["run_index"] = run_index
        agg["output_dir"] = solver_config["output_dir"]
        agg["decode_mode"] = solver_config.get("decode_mode", "minimize")

        # Save per-(algorithm, run) summary into its output dir
        agg_path = os.path.join(
            solver_config["output_dir"], "run_summary_aggregated.json"
        )
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        log_info("Aggregated summary saved to %s", agg_path)

        all_results_for_run.append(agg)

    return all_results_for_run


def main() -> int:
    # Load base config once
    base_config: Dict[str, Any] = load_config(CONFIG_PATH)

    # Set logger
    set_log_level(str(base_config.get("logger", "INFO")).upper())
    log_info("Using config: %s", CONFIG_PATH)

    all_results: List[Dict[str, Any]] = []

    # Loop over run indices ONLY; algorithms are handled inside run_one_experiment
    for run_index in range(RUNS_PER_ALGO):
        results_for_this_run = run_one_experiment(base_config, run_index)
        for result in results_for_this_run:
            all_results.append(result)
            log_info(
                "[done] algo=%s run=%d/%d sum_distance=%.3f",
                result.get("algorithm_name"),
                run_index + 1,
                RUNS_PER_ALGO,
                result.get("sum_final_distance", 0.0),
            )

    # -----------------------------------------------------------------
    # Save combined experiment report (all algorithms × runs)
    # -----------------------------------------------------------------
    base_output_root = Path(base_config.get("output_dir", "results")).resolve()
    report_dir = base_output_root.parent
    ensure_dir(str(report_dir))

    combined_json = report_dir / "experiment_multi_run_summary.json"
    combined_csv = report_dir / "experiment_multi_run_summary.csv"

    # JSON
    combined_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # CSV
    keep = {
        "algorithm_name",
        "algorithm_id",
        "config_name",      # kept for compatibility
        "run_index",
        "output_dir",
        "clusters",
        "scorer",
        "decode_mode",
        "sum_final_distance",
        "sum_final_cost",
        "sum_final_time",
    }
    rows = [{k: s.get(k) for k in keep} for s in all_results]
    pd.DataFrame(rows).to_csv(combined_csv, index=False)

    log_info("Experiment finished. JSON: %s | CSV: %s", combined_json, combined_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
