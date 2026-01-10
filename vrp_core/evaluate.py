from __future__ import annotations

from typing import Any, Dict, List, Tuple, Mapping

from .models.vehicle import vehicle
from .metrics import route_distance, route_time


def eval_routes_cost(routes: List[List[int]], vrp: Dict[str, Any]) -> Tuple[float, float, float]:
    D: Mapping[int, Mapping[int, float]] = vrp["D"]
    T: Mapping[int, Mapping[int, float]] = vrp["T"]
    vehicles: List[vehicle] = vrp["vehicles"]

    total_cost = 0.0
    total_distance = 0.0
    total_time = 0.0

    for r, veh in zip(routes, vehicles):
        dist_m = route_distance(r, D)
        time_s = route_time(r, T)
        cost = veh.initial_cost + veh.distance_cost * dist_m

        total_cost += cost
        total_distance += dist_m
        total_time += time_s

    return total_cost, total_distance, total_time
