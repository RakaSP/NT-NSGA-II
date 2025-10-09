#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from Utils.Logger import set_log_level, log_info
from Utils.Utils import load_problem, decode_routes, validate_capacity, eval_routes_cost, write_routes_json, write_summary_csv, ensure_dir


class VRPSolverEngine:
    def __init__(self, config: Dict[str, Any]):
        # logging
        set_log_level(str(config.get("logger", "INFO")).upper())

        # io
        self.nodes_path = config.get("nodes", "vrp_nodes.csv")
        self.vehicles_path = config.get("vehicles", "vrp_vehicles.csv")
        self.output_dir = config.get("output_dir", "results")
        ensure_dir(self.output_dir)
        self.routes_name = config.get("routes_out", "solution_routes.json")
        self.summary_name = config.get("summary_out", "solution_summary.csv")

        # choices
        self.algo_name = str(config.get("algorithm")).lower()
        self.scorer_name = str(config.get("scorer")).lower()
        self.params: Dict[str, Any] = config["algo_params"][self.algo_name]
        self.iters: int = int(self.params.pop("iters"))

        # late
        self.vrp: Dict[str, Any] | None = None
        self.score_fn = None
        self.algo = None

    # --- factories ---
    def _make_scorer(self):
        if self.scorer_name == "cost":
            from Scorer.Cost import score_from_perm
        elif self.scorer_name == "distance":
            from Scorer.Distance import score_from_perm
        else:
            raise ValueError("Unknown scorer: use 'cost' or 'distance'")
        return score_from_perm

    def _make_algorithm(self):
        if self.algo_name == "ga":
            from Algorithm.GA import GA as Algo
        elif self.algo_name == "pso":
            from Algorithm.PSO import PSO as Algo
        elif self.algo_name == "aco":
            from Algorithm.ACO import ACO as Algo
        else:
            raise ValueError("Unknown algorithm: use 'ga', 'pso', or 'aco'")
        return Algo

    # --- lifecycle ---
    def prepare(self) -> None:
        self.vrp = load_problem(self.nodes_path, self.vehicles_path)
        Algo = self._make_algorithm()
        self.algo = Algo(vrp=self.vrp, scorer=self.scorer_name,
                         params=self.params)

    def run(self) -> Dict[str, Any]:
        assert self.vrp is not None and self.algo is not None

        log_info("Algorithm: %s | Scorer: %s",
                 self.algo_name.upper(), self.scorer_name.upper())
        log_info("Output dir: %s", self.output_dir)

        perm, score, metrics, runtime_s = self.algo.solve(iters=self.iters)

        routes = decode_routes(perm, self.vrp)
        validate_capacity(routes, self.vrp)
        total_cost, _ = eval_routes_cost(routes, self.vrp)

        write_routes_json(self.output_dir, self.routes_name, routes, self.vrp)
        write_summary_csv(self.output_dir, self.summary_name, routes, self.vrp)

        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(
            self.output_dir, f"metrics_{self.algo_name}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        log_info("Metrics saved: %s", metrics_path)

        run_summary = {
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "final_score": float(score),
            "final_cost": float(total_cost),
            "runtime_s": float(runtime_s),
            "iterations": int(metrics[-1]["iter"]) if metrics else 0,
        }
        with open(os.path.join(self.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        log_info("Runtime(s): %.6f | Final score: %.6f | Final cost: %.6f",
                 runtime_s, score, total_cost)
        return run_summary
