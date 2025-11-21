#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from vrp_core import load_config, load_problem_from_config, ensure_dir
from vrp_core.cluster_azam import make_azam_subproblems
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine
import random
CONFIG_PATH = "solver_config.yaml"
RUNS_PER_ALGO = 30

# ---------------------------------------------------------------------
# Define your "run configurations" here.
# Each entry will override the base config for that experiment.
# ---------------------------------------------------------------------
RUN_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "nsga2",
        "overrides": {
            "algorithm": "nsga2",
            "rl_enabled": False,
            "training_enabled": False,
        },
    },
]


def run_one_experiment(
    base_cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    run_idx: int,
) -> Dict[str, Any]:
    """
    Do ONE full run (all clusters) for ONE run configuration and ONE run index.
    
    Now passes ALL clusters to Core at once (no loop).
    """
    name = run_cfg["name"]
    overrides: Dict[str, Any] = run_cfg.get("overrides", {})

    # Start from base config and apply overrides
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)

    # Base outdir (from base config or default)
    base_out = Path(base_cfg.get("output_dir", "results")).resolve()

    # For each (config, run_idx), make a unique root output directory
    cfg_output_dir = base_out.parent / f"{base_out.name}_{name}_run{run_idx + 1}"
    cfg["output_dir"] = str(cfg_output_dir)
    ensure_dir(cfg["output_dir"])

    log_info("=== Experiment %s | run %d/%d ===", name, run_idx + 1, RUNS_PER_ALGO)
    log_info("Output dir: %s", cfg["output_dir"])
    log_info("Algorithm: %s", cfg.get("algorithm"))

    # 1) Load the FULL problem once (for this run)
    full_vrp = load_problem_from_config(cfg)

    # 2) Build subproblems
    sub_vrps: List[Dict[str, Any]] = make_azam_subproblems(
        full_vrp,
        depot_id=0,
        num_depots=1,
        num_vehicles=len(full_vrp["vehicles"]),
        seed=random.randint(1,100000),  # you can vary this with run_idx if you want
        capacity_per_cluster=cfg.get("cluster_capacity", None),
    )
    log_info("Built %d subproblems", len(sub_vrps))

    # 3) Create ONE engine and pass ALL clusters at once
    engine = VRPSolverEngine(cfg)
    
    # Pass all clusters to Core (no loop here)
    engine.prepare(sub_vrps)
    agg = engine.run()

    # 4) Add experiment metadata to aggregated summary
    agg["config_name"] = name
    agg["run_index"] = run_idx
    agg["output_dir"] = cfg["output_dir"]
    agg["decode_mode"] = cfg.get("decode_mode", "minimize")

    # Write per-run aggregated summary into that run's output_dir
    agg_path = os.path.join(cfg["output_dir"], "run_summary_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    log_info("Aggregated summary saved to %s", agg_path)

    return agg


def main() -> int:
    # Load base config once
    base_cfg: Dict[str, Any] = load_config(CONFIG_PATH)

    # Set logger
    set_log_level(str(base_cfg.get("logger", "INFO")).upper())
    log_info("Using config: %s", CONFIG_PATH)

    all_runs: List[Dict[str, Any]] = []

    # Loop over run configurations
    for run_cfg in RUN_CONFIGS:
        name = run_cfg["name"]
        for r in range(RUNS_PER_ALGO):
            agg = run_one_experiment(base_cfg, run_cfg, r)
            all_runs.append(agg)
            log_info(
                "[done] config=%s run=%d/%d sum_distance=%.3f",
                name,
                r + 1,
                RUNS_PER_ALGO,
                agg.get("sum_final_distance", 0.0),
            )

    # -----------------------------------------------------------------
    # Save combined experiment report (all configs × runs)
    # -----------------------------------------------------------------
    base_out = Path(base_cfg.get("output_dir", "results")).resolve()
    report_dir = base_out.parent
    ensure_dir(str(report_dir))

    combined_json = report_dir / "experiment_multi_run_summary.json"
    combined_csv = report_dir / "experiment_multi_run_summary.csv"

    # JSON
    combined_json.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")

    # CSV
    keep = {
        "config_name",
        "run_index",
        "output_dir",
        "clusters",
        "algorithm",
        "scorer",
        "decode_mode",
        "sum_final_distance",
        "sum_final_cost",
        "sum_final_time",
    }
    rows = [{k: s.get(k) for k in keep} for s in all_runs]
    pd.DataFrame(rows).to_csv(combined_csv, index=False)

    log_info("Experiment finished. JSON: %s | CSV: %s", combined_json, combined_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())