from __future__ import annotations

import math
import time
import random
from typing import Any, Dict, List, Optional, Iterable, Tuple, Mapping

from Utils.Logger import log_trace, log_info
from vrp_core.decoding import decode_minimize, decode_split_equal
from vrp_core.scorer.distance import score_solution as sdist
from vrp_core.scorer.cost import score_solution as scost
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle

class BaseAlgorithm:
    def __init__(self, vrp: Dict[str, Any], scorer: str = "cost") -> None:
        self.vrp = vrp
        self.seed = random.randint(0, 1_000_000)
        log_info("Seed: %d", self.seed)
        self.rng = random.Random(self.seed)
        self.scorer = str(scorer).lower()

        nodes: List[node] = self.vrp.get("nodes")
        if not nodes:
            raise ValueError("VRP must provide a non-empty 'nodes' list")
        # Depot is the first entry; its ID can be anything (often 0)
        self.depot_id: int = int(nodes[0].id)

        # Customers = list of GLOBAL node IDs excluding the depot's position (first in list)
        self.customers: List[int] = [int(n.id) for n in nodes[1:]]
        if not self.customers:
            raise ValueError("No customers found (only depot present)")

        self.metrics: List[Dict[str, Any]] = []
        self.best_perm: Optional[List[int]] = None
        self.best_score: float = float("inf")
        self.best_info: Dict[str, Any] = {}
        self.decode_mode = "split_equal"  # or "minimize"

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
        import numpy as np
        arr = np.asarray(list(scores), dtype=float)
        b = float(np.min(arr)) if arr.size else float("inf")
        m = float(np.mean(arr)) if arr.size else float("inf")
        self.metrics.append({"iter": int(it), "best": b, "mean": m})
        log_trace("[%s] iter=%d best=%.6f mean=%.6f",
                  self.__class__.__name__.upper(), int(it), b, m)
        return b, m

    def update_global_best(self, perm: List[int], eff_score: float) -> None:
        if eff_score < self.best_score:
            self.best_score = float(eff_score)
            self.best_perm = list(perm)
            try:
                self.best_info = self._compute_details(self.best_perm)
            except Exception:
                self.best_info = {"error": "failed to compute details"}

    # ---------- constraints (raise → +inf) ----------
    def check_constraints(self, perm: List[int]) -> None:
        # 'perm' must be a permutation of customer IDs (not indices)
        if len(perm) != len(set(perm)):
            raise ValueError("duplicate customer in permutation")

        nodes: List[node] = self.vrp["nodes"]
        by_id = {int(n.id): n for n in nodes}
        if not perm:
            raise ValueError("empty permutation")
        unknown = [i for i in perm if int(i) not in by_id or int(i) == self.depot_id]
        if unknown:
            raise ValueError(f"permutation contains unknown / depot IDs: {unknown}")

        vehicles: List[vehicle] = self.vrp["vehicles"]
        total_demand = sum((by_id[i].demand or 0.0) for i in perm)
        total_capacity = sum(v.max_capacity for v in vehicles)
        if total_demand > total_capacity + 1e-9:
            raise ValueError("total demand exceeds total vehicle capacity")

    # ---------- perm -> solution ----------
    def _route_distance(self, route: List[int], D: Mapping[int, Mapping[int, float]]) -> float:
        if len(route) < 2:
            return 0.0
        dist = 0.0
        for a, b in zip(route[:-1], route[1:]):
            dist += float(D[a][b])
        return dist

    def _route_used_capacity(self, route: List[int], nodes: List[node]) -> float:
        if len(route) <= 2:
            return 0.0
        by_id = {int(n.id): n for n in nodes}
        return float(sum((by_id[i].demand or 0.0) for i in route[1:-1]))

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        # perm is in ID space; decoders expect ID space as well.
        if self.decode_mode == "minimize":
            routes = decode_minimize(perm, self.vrp)
        else:
            routes = decode_split_equal(perm, self.vrp)

        nodes: List[node] = self.vrp["nodes"]
        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

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
                "route": [int(x) for x in r],  # ID route
            })
        return sol

    # ---------- evaluation ----------
    def evaluate_perm(self, perm: List[int]) -> float:
        try:
            self.check_constraints(perm)
            solution = self._solution_from_perm(perm)

            if self.scorer == "distance":
                score = sdist(solution)
            else:  # "cost" default
                score = scost(
                    solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"]
                )

            if not math.isfinite(score) or score <= 0.0:
                return float("inf")
            return float(score)
        except Exception:
            return float("inf")

    # ---------- details builder ----------
    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        solution = self._solution_from_perm(perm)
        total_distance = float(sum(s["total_distance"] for s in solution))
        total_used_capacity = float(sum(s["used_capacity"] for s in solution))
        total_capacity = float(sum(s["max_capacity"] for s in solution))

        try:
            if self.scorer == "distance":
                score_val = float(sdist(solution))
            else:
                score_val = float(scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"]))
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
            "per_vehicle": solution,
            "permutation": list(perm),    # best permutation (ID space)
            "depot_id": self.depot_id,
        }

    def solve(self, iters: int):
        raise NotImplementedError
