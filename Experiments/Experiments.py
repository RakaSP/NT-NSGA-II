#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from vrp_core import load_config
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine


class VRPExperiment:
    """
    Simple experiment wrapper around VRPSolverEngine.

    - loads config once
    - runs multiple algorithms for multiple runs
    - writes combined JSON + CSV summary
    """

    def __init__(
        self,
        algos: List[str],
        config_path: str,
        runs_per_algo: int,
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
    # Single run
    # ------------------------------------------------------------------
    def run_one(self, algo_key: str, run_idx: int, *, run_seed: int) -> Dict[str, Any]:
        """Run a single algorithm variant once and return its run_summary dict."""
        from vrp_core import load_problem_from_config
        from vrp_core.cluster_azam import make_azam_subproblems

        cfg = copy.deepcopy(self.cfg)

        # Base outdir (from config or default)
        base_out = Path(cfg.get("output_dir", "results")).resolve()

        # Per-algo overrides + per-run folder
        if algo_key == "ntnsga2":
            # ntnsga2 = nsga2 + RL
            cfg["algorithm"] = "nsga2"
            cfg["rl_enabled"] = True
            cfg["training_enabled"] = False
            cfg["output_dir"] = str(base_out / f"ntnsga2_run{run_idx + 1}")
        else:
            cfg["algorithm"] = algo_key
            cfg["rl_enabled"] = False
            cfg["training_enabled"] = False
            cfg["output_dir"] = str(base_out / f"{algo_key}_run{run_idx + 1}")

        log_info(
            "=== Running %s (run %d/%d) ===",
            algo_key.upper(),
            run_idx + 1,
            self.runs_per_algo,
        )
        log_info("Output dir: %s", cfg["output_dir"])

        # ---- build vrps ----
        full_vrp = load_problem_from_config(cfg)
        engine = VRPSolverEngine(cfg)

        # IMPORTANT: seed is random per run, regardless of tsp or not.
        if cfg.get("tsp") is True:
            full_vrp["clustering_seed"] = run_seed
            engine.prepare([full_vrp])
        else:
            sub_vrps: List[Dict[str, Any]] = make_azam_subproblems(
                full_vrp,
                depot_id=0,
                num_depots=1,
                num_vehicles=len(full_vrp["vehicles"]),
                seed=run_seed,
            )
            # ensure every cluster carries the same clustering_seed (fair across algos)
            for sv in sub_vrps:
                sv["clustering_seed"] = run_seed

            log_info("Built %d subproblems", len(sub_vrps))
            engine.prepare(sub_vrps)

        # Ensure flags are exactly what we want
        engine.rl_enabled = bool(cfg["rl_enabled"])
        engine.training_enabled = bool(cfg["training_enabled"])

        # Run (capture errors so CSV/JSON still gets written)
        try:
            summary = engine.run()
        except Exception as e:
            summary = {"error": f"{type(e).__name__}: {e}"}

        # Always attach run metadata (so CSV isn't blank/scuffed)
        summary["algo_key"] = algo_key
        summary["algorithm"] = cfg.get("algorithm")
        summary["scorer"] = cfg.get("scorer")
        summary["output_dir"] = cfg["output_dir"]
        summary["run_index"] = run_idx  # 0-based
        summary["seed"] = run_seed
        return summary

    # ------------------------------------------------------------------
    # Multi-run + reporting
    # ------------------------------------------------------------------
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all algorithms for all runs_per_algo and store summaries."""
        import random

        self.all_summaries = []

        # run1: algo1 -> algo2 -> algo3, then run2: algo1 -> algo2 -> algo3, ...
        for r in range(self.runs_per_algo):
            # One random seed per run, shared across ALL algorithms
            run_seed = random.randint(1, 1_000_000)

            for a in self.algos:
                summary = self.run_one(a, r, run_seed=run_seed)
                self.all_summaries.append(summary)

        return self.all_summaries

    def save_reports(self) -> None:
        """Save combined JSON + CSV inside the base output_dir from config."""
        if not self.all_summaries:
            log_info("No summaries to save; did you call run_all()?")
            return

        # Reports live INSIDE output_dir (not parent)
        report_dir = Path(self.cfg.get("output_dir", "results")).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)

        # Rename to experiment_summary.*
        combined_json = report_dir / "experiment_summary.json"
        combined_csv = report_dir / "experiment_summary.csv"

        # JSON (full payloads)
        combined_json.write_text(
            json.dumps(self.all_summaries, indent=2),
            encoding="utf-8",
        )

        # Clean CSV with only essential metrics
        rows = []
        for summary in self.all_summaries:
            # Get solving time - now uses total_solving_time_s
            solving_time = summary.get("total_solving_time_s")
            
            # Fallback to old field name for backward compatibility
            if solving_time is None:
                solving_time = summary.get("total_runtime_s")
            
            # Get sum_final_time
            sum_final_time = summary.get("sum_final_time")
            
            row = {
                "algorithm": summary.get("algo_key"),
                "run": summary.get("run_index"),
                "seed": summary.get("seed"),
                "distance": summary.get("sum_final_distance"),
                "cost": summary.get("sum_final_cost"),
                "route_time": sum_final_time,  # Total time of routes
                "solving_time": solving_time,  # Time to solve the problem
                "clusters": summary.get("clusters"),
            }
            
            # Add iterations and pareto_front_size if they exist
            iterations = summary.get("iterations")
            if iterations is not None:
                row["iterations"] = iterations
                
            pareto_size = summary.get("pareto_front_size")
            if pareto_size is not None:
                row["pareto_size"] = pareto_size
                
            # Add epochs if it exists (for RL algorithms)
            epochs = summary.get("epochs")
            if epochs is not None:
                row["epochs"] = epochs
                
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(combined_csv, index=False)

        log_info(
            "Experiment finished. JSON: %s | CSV: %s",
            combined_json,
            combined_csv,
        )