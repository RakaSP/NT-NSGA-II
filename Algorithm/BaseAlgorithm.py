from __future__ import annotations

import math
import time
import random
from typing import Any, Dict, List, Optional, Iterable, Tuple, Mapping
import numpy as np  

from Utils.Logger import log_trace, log_info
from vrp_core.decoding import decode_route
from vrp_core.scorer.distance import score_solution as sdist
from vrp_core.scorer.cost import score_solution as scost
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle


class BaseAlgorithm:
    def __init__(self,seed, vrp: Dict[str, Any], scorer: str = "cost") -> None:
        self.vrp = vrp
        self.seed = int(seed)
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

    def record_iteration(self, it: int, scores: Iterable[float], perms: Optional[Iterable[List[int]]] = None) -> Tuple[float, float]:
        arr = np.asarray(list(scores), dtype=float)
        b = float(np.min(arr)) if arr.size else float("inf")
        m = float(np.mean(arr)) if arr.size else float("inf")
        
        # Calculate route times from permutations ONLY if provided
        best_route_time = float("inf")
        mean_route_time = float("inf")
        
        if perms is not None:
            perm_list = list(perms)
            route_times = []
            for perm in perm_list:
                details = self._compute_details(perm)
                total_time = details.get("summary", {}).get("total_time", float("inf"))
                route_times.append(total_time)
            
            if route_times:
                route_arr = np.asarray(route_times, dtype=float)
                best_route_time = float(np.min(route_arr)) if route_arr.size else float("inf")
                mean_route_time = float(np.mean(route_arr)) if route_arr.size else float("inf")
        
        # Only append metrics if this iteration hasn't been recorded yet
        # (Check if the last metric has the same iteration number)
        if not self.metrics or self.metrics[-1]["iter"] != it:
            self.metrics.append({
                "iter": int(it), 
                "best": b, 
                "mean": m,
                "best_route_time": best_route_time,
                "mean_route_time": mean_route_time
            })
            
        log_trace("[%s] iter=%d best=%.6f mean=%.6f best_route_time=%.6f mean_route_time=%.6f",
                self.__class__.__name__.upper(), int(it), b, m, best_route_time, mean_route_time)
        return b, m

    def update_global_best(self, perm: List[int], eff_score: float) -> None:
        if eff_score < self.best_score:
            self.best_score = float(eff_score)
            self.best_perm = list(perm)
            try:
                self.best_info = self._compute_details(self.best_perm)
            except Exception:
                self.best_info = {"error": "failed to compute details"}

    # ---------- constraints ----------
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

        # NOTE: capacity constraints intentionally removed (per your request)

    # ---------- route helpers ----------
    def _route_distance(self, route: List[int], D: Mapping[int, Mapping[int, float]]) -> float:
        if len(route) < 2:
            return 0.0
        dist = 0.0
        for a, b in zip(route[:-1], route[1:]):
            dist += float(D[a][b])
        return dist

    def _route_time(self, route: List[int], Tm: Mapping[int, Mapping[int, float]]) -> float:
        if len(route) < 2:
            return 0.0
        tt = 0.0
        for a, b in zip(route[:-1], route[1:]):
            tt += float(Tm[a][b])
        return tt

    # ---------- perm -> solution ----------
    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        # perm is in ID space; decoders expect ID space as well.

        routes = decode_route(perm, self.vrp)


        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        sol: List[Dict[str, Any]] = []
        for r, veh in zip(routes, vehicles):
            total_distance = self._route_distance(r, D)
            sol.append({
                "vehicle_id": veh.id,
                "vehicle_name": veh.vehicle_name,
                "vehicle_distance_cost": veh.distance_cost,
                "vehicle_initial_cost": veh.initial_cost,
                "total_distance": total_distance,
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
                score = scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])

            if not math.isfinite(score) or score <= 0.0:
                return float("inf")
            return float(score)
        except Exception:
            return float("inf")

    # ---------- details builder ----------
    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        solution = self._solution_from_perm(perm)

        total_distance = float(sum(s["total_distance"] for s in solution))

        # total_time computed from the route + vrp["T"]
        Tm: Mapping[int, Mapping[int, float]] = self.vrp["T"]
        total_time = 0.0
        for s in solution:
            route = s["route"]
            total_time += self._route_time(route, Tm)

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
                "total_time": float(total_time),
                "scorer": self.scorer,
                "score": score_val,
            },
            "per_vehicle": solution,
            "permutation": list(perm),  # best permutation (ID space)
            "depot_id": self.depot_id,
        }

    def solve(self, iters: int):
        raise NotImplementedError
