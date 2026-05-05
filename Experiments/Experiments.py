#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from random import random
from typing import Any, Dict, List

import pandas as pd

from vrp_core import load_config
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine

# -------------------------
# Experiment options
# -------------------------
import random
STATIC_CLUSTER = True          # True = reuse the same clustering across ALL runs
STATIC_CLUSTER_SEED = 187031     # set to an int to force a specific clustering seed


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
    def run_one(
        self,
        algo_key: str,
        run_idx: int,
        *,
        run_seed: int,
        cluster_seed: int | None = None,
    ) -> Dict[str, Any]:
        """Run a single algorithm variant once and return its run_summary dict."""
        from vrp_core import load_problem_from_config
        from vrp_core.cluster_azam import make_azam_subproblems

        cfg = copy.deepcopy(self.cfg)

        # Seed used for algorithm randomness (if your engine reads cfg["seed"])
        cfg["seed"] = run_seed

        # Seed used ONLY for clustering (defaults to run_seed for backward compatibility)
        effective_cluster_seed = run_seed if cluster_seed is None else cluster_seed

        base_out = Path(cfg.get("output_dir", "results")).resolve()

        if algo_key == "ntnsga2":
            cfg["algorithm"] = "nsga2"
            cfg["rl_enabled"] = True
            cfg["training_enabled"] = False
            cfg["output_dir"] = str(base_out / f"ntnsga2_run{run_idx + 1}")
        elif algo_key == "ntnsga2(learn)":
            cfg["algorithm"] = "nsga2"
            cfg["rl_enabled"] = True
            cfg["training_enabled"] = True
            cfg["output_dir"] = str(base_out / f"ntnsga2_learn_run{run_idx + 1}")
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
        log_info("run_seed=%s | cluster_seed=%s", run_seed, effective_cluster_seed)

        full_vrp = load_problem_from_config(cfg)
        engine = VRPSolverEngine(cfg)

        if cfg.get("tsp") is True:
            # keep metadata consistent
            full_vrp["clustering_seed"] = effective_cluster_seed
            engine.prepare([full_vrp])
        else:
            sub_vrps: List[Dict[str, Any]] = make_azam_subproblems(
                full_vrp,
                depot_id=0,
                num_depots=1,
                num_vehicles=len(full_vrp["vehicles"]),
                seed=effective_cluster_seed,   # <-- CLUSTERING seed (static if enabled)
            )
            for sv in sub_vrps:
                sv["clustering_seed"] = effective_cluster_seed

            log_info("Built %d subproblems", len(sub_vrps))
            engine.prepare(sub_vrps)

        engine.rl_enabled = bool(cfg["rl_enabled"])
        engine.training_enabled = bool(cfg["training_enabled"])

        summary = engine.run()

        summary["algo_key"] = algo_key
        summary["algorithm"] = cfg.get("algorithm")
        summary["scorer"] = cfg.get("scorer")
        summary["output_dir"] = cfg["output_dir"]
        summary["run_index"] = run_idx
        summary["seed"] = run_seed
        summary["clustering_seed"] = effective_cluster_seed
        return summary

    # ------------------------------------------------------------------
    # Multi-run + reporting
    # ------------------------------------------------------------------
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all algorithms for all runs_per_algo and store summaries."""
        import random

        global STATIC_CLUSTER_SEED

        self.all_summaries = []

        # Decide clustering seed once (if static clustering enabled)
        if STATIC_CLUSTER:
            if STATIC_CLUSTER_SEED is None:
                STATIC_CLUSTER_SEED = random.randint(1, 1_000_000)
            fixed_cluster_seed = STATIC_CLUSTER_SEED
            log_info("STATIC_CLUSTER enabled. Using fixed clustering_seed=%s", fixed_cluster_seed)
        else:
            fixed_cluster_seed = None

        for r in range(self.runs_per_algo):
            # algo randomness seed (changes each run; shared across algos for fairness)
            run_seed = random.randint(1, 1_000_000)

            for a in self.algos:
                summary = self.run_one(a, r, run_seed=run_seed, cluster_seed=fixed_cluster_seed)
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