# Algorithm/BaseAlgorithm.py
from __future__ import annotations

import math
import time
import random
from typing import Any, Dict, List, Optional, Iterable, Tuple

import numpy as np
from Utils.Logger import log_trace
from Utils.Utils import decode_routes  # keeps logic centralized
from Utils.Logger import log_info


class BaseAlgorithm:
    def __init__(self, vrp: Dict[str, Any], scorer: str = "cost") -> None:
        self.vrp = vrp
        self.seed = random.randint(0, 1000000)
        log_info("Seed: %d", self.seed)
        self.rng = random.Random(self.seed)
        self.scorer = str(scorer).lower()

        nodes = self.vrp.get("nodes")
        if not nodes or nodes[0].id != 0:
            raise ValueError("nodes must exist and depot id must be 0")
        if any(nodes[i].id != i for i in range(len(nodes))):
            raise ValueError(
                "node ids must be contiguous 0..N-1 (id == index)")

        self.customers: List[int] = [node.id for node in nodes if node.id != 0]
        if not self.customers:
            raise ValueError("No customers found")

        self.metrics: List[Dict[str, Any]] = []
        self.best_perm: Optional[List[int]] = None
        self.best_score: float = float("inf")
        self.best_info: Dict[str, Any] = {}

        self._t0: Optional[float] = None
        self.runtime_s: float = 0.0

    # ---------- lifecycle ----------
    def start_run(self) -> None:
        self.metrics.clear()
        self.best_perm = None
        self.best_score = float("inf")
        self.best_info = {}
        self._t0 = time.perf_counter()

    def finalize(self) -> float:
        if self._t0 is None:
            return 0.0
        self.runtime_s = time.perf_counter() - self._t0
        return self.runtime_s

    def record_iteration(self, it: int, scores: Iterable[float]) -> Tuple[float, float]:
        arr = np.asarray(list(scores), dtype=float)
        b = float(np.min(arr)) if arr.size else float("inf")
        m = float(np.mean(arr)) if arr.size else float("inf")
        self.metrics.append({"iter": int(it), "best": b, "mean": m})
        log_trace("[%s] iter=%d best=%.6f mean=%.6f",
                  self.__class__.__name__.upper(), int(it), b, m)
        return b, m

    def update_global_best(self, perm: List[int], eff_score: float) -> None:
        """Update best permutation/score and always compute readable details."""
        if eff_score < self.best_score:
            self.best_score = float(eff_score)
            self.best_perm = list(perm)
            try:
                self.best_info = self._compute_details(self.best_perm)
            except Exception:
                # Keep robustness: details shouldn't break the run
                self.best_info = {"error": "failed to compute details"}

    # ---------- constraints (raise → +inf) ----------
    def check_constraints(self, perm: List[int]) -> None:
        if len(perm) != len(set(perm)):
            raise ValueError("duplicate customer in permutation")
        if not perm or min(perm) < 1 or max(perm) >= len(self.vrp["nodes"]):
            raise ValueError("permutation contains out-of-range customer id")
        nodes = self.vrp["nodes"]
        vehicles = self.vrp["vehicles"]
        total_demand = sum((nodes[i].demand or 0.0) for i in perm)
        total_capacity = sum(v.max_capacity for v in vehicles)
        if total_demand > total_capacity + 1e-9:
            raise ValueError("total demand exceeds total vehicle capacity")

    # ---------- perm -> solution ----------
    def _route_distance(self, route: List[int], D: np.ndarray) -> float:
        if len(route) < 2:
            return 0.0
        dist = 0.0
        for i in range(len(route) - 1):
            dist += float(D[route[i], route[i + 1]])
        return dist

    def _route_used_capacity(self, route: List[int], nodes) -> float:
        if len(route) <= 2:
            return 0.0
        return float(sum((nodes[i].demand or 0.0) for i in route[1:-1]))

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        routes = decode_routes(perm, self.vrp)  # may raise on infeasible split
        nodes = self.vrp["nodes"]
        vehicles = self.vrp["vehicles"]
        D = self.vrp["D"]

        sol: List[Dict[str, Any]] = []
        for r, veh in zip(routes, vehicles):
            total_distance = self._route_distance(r, D)
            used_capacity = self._route_used_capacity(r, nodes)
            sol.append({
                "vehicle_id": veh.id,
                "vehicle_name": veh.vehicle_name,
                "vehicle_distance_cost": veh.distance_cost,
                "vehicle_initial_cost": veh.initial_cost,
                "total_distance": total_distance,
                "max_capacity": veh.max_capacity,
                "used_capacity": used_capacity,
                "route": [int(x) for x in r],
            })
        return sol

    # ---------- evaluation ----------
    def evaluate_perm(self, perm: List[int]) -> float:
        try:
            self.check_constraints(perm)
            solution = self._solution_from_perm(perm)

            if self.scorer == "distance":
                from Scorer.Distance import score_solution as sdist
                score = sdist(solution)
            else:  # "cost" default
                from Scorer.Cost import score_solution as scost
                score = scost(
                    solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])

            if not math.isfinite(score) or score <= 0.0:
                return float("inf")
            return float(score)
        except Exception:
            return float("inf")

    # ---------- details builder (always on) ----------
    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        """Build a readable dictionary describing the current best solution."""
        solution = self._solution_from_perm(perm)
        total_distance = float(sum(s["total_distance"] for s in solution))
        total_used_capacity = float(sum(s["used_capacity"] for s in solution))
        total_capacity = float(sum(s["max_capacity"] for s in solution))

        # Optional: include the active scorer’s scalar value for transparency
        try:
            if self.scorer == "distance":
                from Scorer.Distance import score_solution as sdist
                score_val = float(sdist(solution))
            else:
                from Scorer.Cost import score_solution as scost
                score_val = float(
                    scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"]))
        except Exception:
            score_val = float("nan")

        return {
            "summary": {
                "vehicles_used": len(solution),
                "total_distance": total_distance,
                "total_used_capacity": total_used_capacity,
                "total_capacity": total_capacity,
                "capacity_utilization": (total_used_capacity / total_capacity) if total_capacity > 0 else 0.0,
                "scorer": self.scorer,
                "score": score_val,
            },
            "per_vehicle": solution,   # list of dicts (one per vehicle/route)
            "permutation": list(perm),  # best permutation that produced this
        }

    # to be implemented by subclasses
    def solve(self, iters: int):
        raise NotImplementedError