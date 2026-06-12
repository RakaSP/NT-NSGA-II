from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from Algorithm.BaseAlgorithm import BaseAlgorithm
from vrp_core.decoding import decode_route
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle
from vrp_core.scorer.distance import score_solution as sdist


class BaseBugReplicated(BaseAlgorithm):
    def __init__(self, seed, vrp: Dict[str, Any]) -> None:
        vrp_copy = dict(vrp)
        vrp_copy["D"], vrp_copy["T"] = self._build_haversine_matrices(vrp_copy["nodes"])
        super().__init__(seed, vrp_copy)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000.0

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return R * c

    def _build_haversine_matrices(
        self,
        nodes: List[node],
    ) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        D: Dict[int, Dict[int, float]] = {}
        T: Dict[int, Dict[int, float]] = {}

        for n1 in nodes:
            id1 = int(n1.id)
            D[id1] = {}
            T[id1] = {}

            for n2 in nodes:
                id2 = int(n2.id)

                if id1 == id2:
                    D[id1][id2] = 0.0
                    T[id1][id2] = 0.0
                    continue

                dist = self._haversine_distance(
                    float(n1.lat),
                    float(n1.lon),
                    float(n2.lat),
                    float(n2.lon),
                )

                D[id1][id2] = float(dist)
                T[id1][id2] = float(dist)

        return D, T

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        routes = decode_route(perm, self.vrp)

        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        T: Mapping[int, Mapping[int, float]] = self.vrp["T"]

        sol: List[Dict[str, Any]] = []

        for r, veh in zip(routes, vehicles):
            customer_route: List[int] = [
                int(x) for x in r if int(x) != self.depot_id
            ]

            total_distance = 0.0
            total_time = 0.0

            if len(customer_route) >= 2:
                for a, b in zip(customer_route[:-1], customer_route[1:]):
                    total_distance += float(D[a][b])
                    total_time += float(T[a][b])

            sol.append(
                {
                    "vehicle_id": veh.id,
                    "vehicle_name": veh.vehicle_name,
                    "vehicle_distance_cost": veh.distance_cost,
                    "vehicle_initial_cost": veh.initial_cost,
                    "total_distance": float(total_distance),
                    "total_time": float(total_time),
                    "route": customer_route,
                }
            )

        return sol

    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        solution = self._solution_from_perm(perm)

        total_distance = float(sum(s["total_distance"] for s in solution))
        total_time = float(sum(s["total_time"] for s in solution))

        score_val = float(sdist(solution))
       

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