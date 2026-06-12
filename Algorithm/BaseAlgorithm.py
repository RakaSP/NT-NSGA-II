from __future__ import annotations

import math
import time
import random
from typing import Any, Dict, List, Optional, Iterable, Tuple, Mapping

from Utils.Logger import log_trace, log_info
from vrp_core.decoding import decode_route
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle


class BaseAlgorithm:
    def __init__(self, seed, vrp: Dict[str, Any]) -> None:
        self.vrp = vrp
        self.seed = int(seed)
        log_info("Seed: %d", self.seed)
        self.rng = random.Random(self.seed)

        nodes: List[node] = self.vrp.get("nodes")
        if not nodes:
            raise ValueError("VRP must provide a non-empty 'nodes' list")

        self.depot_id: int = int(nodes[0].id)
        self.customers: List[int] = [int(n.id) for n in nodes[1:]]
        if not self.customers:
            raise ValueError("No customers found (only depot present)")

        self.metrics: List[Dict[str, Any]] = []
        self.best_perm: Optional[List[int]] = None
        self.best_score: float = float("inf")
        self.best_info: Dict[str, Any] = {}

        self._t0: Optional[float] = None
        self.runtime_s: float = 0.0

        # termination
        self._stop_event = None
        self._deadline: Optional[float] = None

    # ---------------- termination helpers ----------------
    def arm_time_limit(self, time_limit_s: Optional[float]) -> None:
        if time_limit_s is None:
            self._deadline = None
            return
        try:
            tl = float(time_limit_s)
        except Exception:
            self._deadline = None
            return
        self._deadline = time.perf_counter() + tl if (math.isfinite(tl) and tl > 0) else None

    def set_stop_event(self, stop_event) -> None:
        self._stop_event = stop_event

    def time_up(self) -> bool:
        return (self._deadline is not None) and (time.perf_counter() >= self._deadline)

    def stop_requested(self) -> bool:
        if self._stop_event is not None:
            try:
                if hasattr(self._stop_event, "is_set") and self._stop_event.is_set():
                    return True
            except Exception:
                pass
        return self.time_up()

    # ---------------- lifecycle ----------------
    def start_run(self) -> None:
        self.metrics.clear()
        self.best_perm = None
        self.best_score = float("inf")
        self.best_info = {}

    def record_iteration(
        self,
        it: int,
        scores: Iterable[float],
        perms: Optional[Iterable[List[int]]] = None,
    ) -> Tuple[float, float]:
        score_list = list(scores)
        b = min(score_list)
        m = (sum(score_list) / len(score_list))

        best_route_time = float("inf")
        mean_route_time = float("inf")

        if perms is not None and not self.stop_requested():
            route_times: List[float] = []
            for perm in perms:
                if self.stop_requested():
                    break
                details = self._compute_details(perm)
                tt = details.get("summary", {}).get("total_time")
                if tt is None:
                    continue
                try:
                    route_times.append(float(tt))
                except Exception:
                    continue
            if route_times:
                best_route_time = min(route_times)
                mean_route_time = sum(route_times) / len(route_times)

        self.metrics.append(
            {
                "iter": int(it),
                "best": float(b),
                "mean": float(m),
                "best_route_time": float(best_route_time),
                "mean_route_time": float(mean_route_time),
            }
        )

        log_trace(
            "[%s] iter=%d best=%.6f mean=%.6f best_route_time=%.6f mean_route_time=%.6f",
            self.__class__.__name__.upper(),
            int(it),
            float(b),
            float(m),
            float(best_route_time),
            float(mean_route_time),
        )
        return float(b), float(m)

    def update_global_best(self, perm: List[int], eff_score: float) -> None:
        if eff_score < self.best_score:
            self.best_score = float(eff_score)
            self.best_perm = list(perm)
            try:
                self.best_info = self._compute_details(self.best_perm)
            except Exception:
                self.best_info = {"error": "failed to compute details"}

    # ---------------- constraints ----------------
    def check_constraints(self, perm: List[int]) -> None:
        if len(perm) != len(set(perm)):
            raise ValueError("duplicate customer in permutation")

        nodes: List[node] = self.vrp["nodes"]
        by_id = {int(n.id): n for n in nodes}
        if not perm:
            raise ValueError("empty permutation")

        unknown = [i for i in perm if int(i) not in by_id or int(i) == self.depot_id]
        if unknown:
            raise ValueError(f"permutation contains unknown / depot IDs: {unknown}")

    # ---------------- route helpers ----------------
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

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        routes = decode_route(perm, self.vrp)

        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        T: Mapping[int, Mapping[int, float]] = self.vrp["T"]

        sol: List[Dict[str, Any]] = []
        for r, veh in zip(routes, vehicles):
            total_distance = self._route_distance(r, D)
            total_time = self._route_time(r, T)
            sol.append(
                {
                    "vehicle_id": veh.id,
                    "vehicle_name": veh.vehicle_name,
                    "vehicle_distance_cost": veh.distance_cost,
                    "vehicle_initial_cost": veh.initial_cost,
                    "total_distance": total_distance,
                    "total_time": total_time,
                    "route": [int(x) for x in r],
                }
            )
        return sol

    def evaluate_perm(self, perm: List[int]) -> float:
        if self.stop_requested():
            return float("inf")
        try:
            self.check_constraints(perm)
            solution = self._solution_from_perm(perm)
            score = sdist(solution)

            if not math.isfinite(score) or score <= 0.0:
                return float("inf")
            return float(score)
        except Exception:
            return float("inf")

    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        solution = self._solution_from_perm(perm)
        total_distance = float(sum(s["total_distance"] for s in solution))
        total_time = float(sum(s["total_time"] for s in solution))

        try:
            score_val = float(sdist(solution))
        except Exception:
            score_val = float("nan")

        return {
            "summary": {
                "vehicles_used": len(solution),
                "total_distance": total_distance,
                "total_time": total_time,
                "score": score_val,
            },
            "per_vehicle": solution,
            "permutation": list(perm),
            "depot_id": self.depot_id,
        }

    def solve(self, iters: int, time_limit_s: Optional[float] = None):
        raise NotImplementedError