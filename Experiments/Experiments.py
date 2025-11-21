#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from vrp_core import load_config
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine

CONFIG_PATH = "solver_config.yaml"  # or "config.json"
RUNS_PER_ALGO = 10


class VRPExperiment:
    """
    Simple experiment wrapper around VRPSolverEngine.

    - loads config once
    - runs multiple algorithms for multiple runs
    - writes combined JSON + CSV summary
    """

    def __init__(
        self,
        config_path: str = CONFIG_PATH,
        runs_per_algo: int = RUNS_PER_ALGO,
        algos: Optional[List[str]] = None,
    ) -> None:
        self.config_path = config_path
        self.runs_per_algo = runs_per_algo
        self.algos: List[str] = algos if algos is not None else ["aco", "nsga2", "ntnsga2"]

        # Load base config
        self.cfg: Dict[str, Any] = load_config(self.config_path)

        # Set logger early
        set_log_level(str(self.cfg.get("logger", "INFO")).upper())
        log_info("Using config: %s", self.config_path)

        # Storage for all runs
        self.all_summaries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Single run (direct port of your run_one)
    # ------------------------------------------------------------------
    def run_one(self, algo_key: str, run_idx: int) -> Dict[str, Any]:
        """Run a single algorithm variant once and return its run_summary dict."""
        cfg = copy.deepcopy(self.cfg)

        # Base outdir (from config or default)
        base_out = Path(cfg.get("output_dir", "results")).resolve()

        # Per-algo overrides + per-run folder
        if algo_key == "ntnsga2":
            # ntnsga2 = nsga2 + RL
            cfg["algorithm"] = "nsga2"
            cfg["rl_enabled"] = True
            cfg["training_enabled"] = True
            cfg["output_dir"] = str(
                base_out.parent / f"{base_out.name}_ntnsga2_run{run_idx + 1}"
            )
        else:
            cfg["algorithm"] = algo_key
            cfg["rl_enabled"] = False
            cfg["training_enabled"] = False
            cfg["output_dir"] = str(
                base_out.parent / f"{base_out.name}_{algo_key}_run{run_idx + 1}"
            )

        log_info(
            "=== Running %s (run %d/%d) ===",
            algo_key.upper(),
            run_idx + 1,
            self.runs_per_algo,
        )
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
        summary["run_index"] = run_idx  # 0-based
        return summary

    # ------------------------------------------------------------------
    # Multi-run + reporting (direct port of your main())
    # ------------------------------------------------------------------
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all algorithms for all runs_per_algo and store summaries."""
        self.all_summaries = []

        for a in self.algos:
            for r in range(self.runs_per_algo):
                summary = self.run_one(a, r)
                self.all_summaries.append(summary)

        return self.all_summaries

    def save_reports(self) -> None:
        """Save combined JSON + CSV next to the base output_dir."""
        if not self.all_summaries:
            log_info("No summaries to save; did you call run_all()?")
            return

        # Combined reports live next to the base output_dir
        base_out = Path(self.cfg.get("output_dir", "results")).resolve()
        report_dir = base_out.parent
        report_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        combined_json = report_dir / "multi_run_summary.json"
        combined_json.write_text(
            json.dumps(self.all_summaries, indent=2),
            encoding="utf-8",
        )

        # CSV
        keep = {
            "algo_key",
            "algorithm",
            "scorer",
            "final_distance",
            "final_cost",
            "final_time",
            "runtime_s",
            "iterations",
            "output_dir",
            "pareto_front_size",
            "error",
            "run_index",
        }
        rows = [{k: s.get(k) for k in keep} for s in self.all_summaries]
        pd.DataFrame(rows).to_csv(report_dir / "multi_run_summary.csv", index=False)

        log_info(
            "Multi-run finished. JSON: %s | CSV: %s",
            combined_json,
            report_dir / "multi_run_summary.csv",
        )

