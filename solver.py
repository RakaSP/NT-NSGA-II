#!/usr/bin/env python3
"""
Solver.py
---------
- Reads config (including logger level and output folder)
- Loads problem
- Selects algorithm (GA/PSO/ACO) and scorer (Cost/Distance)
- Runs solver; logs per-iteration best score on TRACE
- Saves:
  * solution_routes.json
  * solution_summary.csv
  * metrics_<algo>.csv (iteration,best,mean)
  * run_summary.json (algo, scorer, final_score, runtime_s)
No distance matrix is written to disk.
"""

import argparse
import json
import os
import sys

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

from Utils.Utils import (
    set_log_level, log_info, log_trace,
    load_problem, decode_routes, eval_routes_cost,
    write_routes_json, write_summary_csv, ensure_dir
)
from Algorithm.GA import ga_solve
from Algorithm.PSO import pso_solve
from Algorithm.ACO import aco_solve

from Scorer.Cost import score_from_perm as cost_score
from Scorer.Distance import score_from_perm as distance_score
import pandas as pd


def load_config(path: str) -> dict:
    if path.lower().endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError(
                "PyYAML not installed. Install with: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise RuntimeError(
            "Unsupported config format. Use .yaml/.yml or .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="solver_config.yaml")
    ap.add_argument("--algorithm", choices=["ga", "pso", "aco"], default=None,
                    help="override algorithm in config")
    ap.add_argument("--scorer", choices=["cost", "distance"], default=None,
                    help="override scorer in config")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Logger from config (TRACE or INFO)
    set_log_level(cfg.get("logger", "INFO"))

    # Required paths
    nodes = cfg.get("nodes", "vrp_nodes.csv")
    vehicles = cfg.get("vehicles", "vrp_vehicles.csv")

    # Output folder & file names (we always wrap outputs inside output_dir)
    output_dir = cfg.get("output_dir", "results")
    ensure_dir(output_dir)
    routes_name = cfg.get("routes_out",  "solution_routes.json")
    summary_name = cfg.get("summary_out", "solution_summary.csv")

    # Algo & scorer
    algo = (args.algorithm or cfg.get("algorithm", "ga")).lower()
    scorer_name = (args.scorer or cfg.get("scorer", "cost")).lower()
    if scorer_name == "cost":
        score_fn = cost_score
    elif scorer_name == "distance":
        score_fn = distance_score
    else:
        raise SystemExit("Unknown scorer: cost | distance")

    # Algo params
    params = (cfg.get("algo_params") or {}).get(algo, {})

    log_info("Algorithm: %s | Scorer: %s", algo.upper(), scorer_name.upper())
    log_info("Output dir: %s", output_dir)

    # Load problem (meters)
    pb = load_problem(nodes, vehicles)

    # Run selected algorithm
    if algo == "ga":
        perm, score, metrics, runtime_s = ga_solve(pb, score_fn=score_fn, **{
            k: v for k, v in params.items() if k in {"iters", "pop", "cx_rate", "mut_rate", "elite", "seed"}
        })
    elif algo == "pso":
        perm, score, metrics, runtime_s = pso_solve(pb, score_fn=score_fn, **{
            k: v for k, v in params.items() if k in {"iters", "pop", "w", "c1", "c2", "seed"}
        })
    elif algo == "aco":
        perm, score, metrics, runtime_s = aco_solve(pb, score_fn=score_fn, **{
            k: v for k, v in params.items() if k in {"iters", "ants", "alpha", "beta", "rho", "seed"}
        })
    else:
        raise SystemExit("Unknown algorithm")

    # Decode final perm and compute full COST breakdown regardless of scorer
    routes = decode_routes(perm, pb)
    total_cost, breakdown = eval_routes_cost(routes, pb)

    # Persist solution & summary
    write_routes_json(output_dir, routes_name, routes, pb)
    write_summary_csv(output_dir, summary_name, routes, pb)

    # Persist metrics per iteration
    metrics_df = pd.DataFrame(metrics)  # cols: iter,best,mean
    metrics_path = os.path.join(output_dir, f"metrics_{algo}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log_info("Metrics saved: %s", metrics_path)

    # Persist run summary (for cross-run comparison)
    run_summary = {
        "algorithm": algo,
        "scorer": scorer_name,
        "final_score": float(score),
        "final_cost": float(total_cost),   # always cost, for reference
        "runtime_s": float(runtime_s),
        "iterations": int(metrics[-1]["iter"]) if metrics else 0,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    log_info("Runtime(s): %.6f | Final score: %.6f | Final cost: %.6f",
             runtime_s, score, total_cost)


if __name__ == "__main__":
    main()
