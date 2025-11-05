#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from Utils.Utils import load_config
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine

CONFIG_PATH = "solver_config.yaml"  # or "config.json"
RUNS_PER_ALGO = 10

def run_one(cfg_base: Dict[str, Any], algo_key: str, run_idx: int) -> Dict[str, Any]:
    """Run a single algorithm variant once and return its run_summary dict."""
    cfg = copy.deepcopy(cfg_base)

    # Base outdir (from config or default)
    base_out = Path(cfg.get("output_dir", "results")).resolve()

    # Per-algo overrides + per-run folder
    if algo_key == "ntnsga2":
        # ntnsga2 = nsga2 + RL
        cfg["algorithm"] = "nsga2"
        cfg["rl_enabled"] = True
        cfg["training_enabled"] = True
        cfg["output_dir"] = str(base_out.parent / f"{base_out.name}_ntnsga2_run{run_idx+1}")
    else:
        cfg["algorithm"] = algo_key
        cfg["rl_enabled"] = False
        cfg["training_enabled"] = False
        cfg["output_dir"] = str(base_out.parent / f"{base_out.name}_{algo_key}_run{run_idx+1}")

    log_info("=== Running %s (run %d/%d) ===", algo_key.upper(), run_idx+1, RUNS_PER_ALGO)
    log_info("Output dir: %s", cfg["output_dir"])

    engine = VRPSolverEngine(cfg)
    # Your class expects prepare(cfg)
    engine.prepare(cfg)

    # Ensure flags are exactly what we want (if Engine sets defaults in __init__)
    engine.rl_enabled = bool(cfg["rl_enabled"])
    engine.training_enabled = bool(cfg["training_enabled"])

    summary = engine.run()
    summary["algo_key"] = algo_key
    summary["output_dir"] = cfg["output_dir"]
    summary["run_index"] = run_idx  # 0-based; use +1 if you prefer 1-based
    return summary


def main() -> int:
    cfg: Dict[str, Any] = load_config(CONFIG_PATH)

    # Set logger early
    set_log_level(str(cfg.get("logger", "INFO")).upper())
    log_info("Using config: %s", CONFIG_PATH)

    algos: List[str] = ["aco", "nsga2", "ntnsga2"]

    all_summaries: List[Dict[str, Any]] = []
    for a in algos:
        for r in range(RUNS_PER_ALGO):
            summary = run_one(cfg, a, r)
            all_summaries.append(summary)

    # Combined reports live next to the base output_dir
    base_out = Path(cfg.get("output_dir", "results")).resolve()
    report_dir = base_out.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    combined_json = report_dir / "multi_run_summary.json"
    combined_json.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    # CSV
    keep = {
        "algo_key", "algorithm", "scorer", "final_distance",
        "final_cost", "final_time", "runtime_s", "iterations",
        "output_dir", "pareto_front_size", "error", "run_index"
    }
    rows = [{k: s.get(k) for k in keep} for s in all_summaries]
    pd.DataFrame(rows).to_csv(report_dir / "multi_run_summary.csv", index=False)

    log_info("Multi-run finished. JSON: %s | CSV: %s", combined_json, report_dir / "multi_run_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
