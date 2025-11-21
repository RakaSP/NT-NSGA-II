from __future__ import annotations
from typing import Any, Dict, List, Tuple, Mapping
from .models.node import node
from .models.vehicle import vehicle
from .metrics import route_distance, route_time, route_load, nodes_by_id

def validate_capacity(routes: List[List[int]], vrp: Dict[str, Any]) -> None:
    nodes: List[node] = vrp["nodes"]
    vehicles: List[vehicle] = vrp["vehicles"]
    nodes_map = nodes_by_id(nodes)
    for r, veh in zip(routes, vehicles):
        used = route_load(r, nodes_map)
        if used > veh.max_capacity + 1e-9:
            raise ValueError(f"Capacity violated for vehicle {veh.id}: load={used} > cap={veh.max_capacity}")

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
