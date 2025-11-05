# vrp_core/evaluate.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from .models.node import Node
from .models.vehicle import Vehicle
from .metrics import route_distance, route_time, route_load

def validate_capacity(routes: List[List[int]], vrp: Dict[str, Any]) -> None:
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
    for r, veh in zip(routes, vehicles):
        used = route_load(r, nodes)
        if used > veh.max_capacity + 1e-9:
            raise ValueError(f"Capacity violated for vehicle {veh.id}: load={used} > cap={veh.max_capacity}")

def eval_routes_cost(routes: List[List[int]], vrp: Dict[str, Any]) -> Tuple[float, float, float]:
    D: np.ndarray = vrp["D"]
    T: np.ndarray = vrp["T"]
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
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
