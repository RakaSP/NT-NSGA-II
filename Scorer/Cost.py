#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from Utils.Node import Node
from Utils.Vehicle import Vehicle


def _route_distance(route: List[int], D: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(route) - 1):
        dist += float(D[int(route[i]), int(route[i + 1])])
    return dist


def _route_used_capacity(route: List[int], nodes: List[Node]) -> float:
    if len(route) <= 2:
        return 0.0
    return float(sum((nodes[i].demand or 0.0) for i in route[1:-1]))


def _assert_route_valid(route: List[int], depot_id: int, n_nodes: int) -> None:
    if not route:
        raise ValueError("route is empty")
    if route[0] != depot_id or route[-1] != depot_id:
        raise ValueError("route must start and end with depot")
    if any((i < 0 or i >= n_nodes) for i in route):
        raise ValueError("route contains out-of-range node id")


def score_solution(
    solution: List[Dict[str, Any]],
    nodes: List[Node],
    vehicles: List[Vehicle],
    D: np.ndarray,
    depot_id: int = 0,
) -> float:
    if not nodes or nodes[0].id != depot_id:
        raise ValueError("nodes must exist and depot id must be 0")
    if any(nodes[i].id != i for i in range(len(nodes))):
        raise ValueError("node ids must be contiguous 0..N-1 (id == index)")

    veh_by_id = {v.id: v for v in vehicles}
    n_nodes = len(nodes)
    total_cost = 0.0

    for idx, item in enumerate(solution):
        try:
            vid = int(item["vehicle_id"])
            v_dist_cost = float(item["vehicle_distance_cost"])
            v_init_cost = float(item["vehicle_initial_cost"])
            vmax_cap = float(item["max_capacity"])
            route = [int(x) for x in item["route"]]
        except Exception as e:
            raise ValueError(f"solution[{idx}] invalid/missing fields") from e

        if v_dist_cost < 0 or v_init_cost < 0 or vmax_cap < 0:
            raise ValueError(f"solution[{idx}] negative cost/capacity")

        _assert_route_valid(route, depot_id, n_nodes)

        # recompute, don't trust inputs
        dist_m = _route_distance(route, D)
        used_cap = _route_used_capacity(route, nodes)
        if used_cap > vmax_cap + 1e-9:
            raise ValueError(
                f"solution[{idx}] capacity exceeded: used={used_cap} > max={vmax_cap}"
            )

        if vid in veh_by_id:
            vref = veh_by_id[vid]
            if abs(vref.distance_cost - v_dist_cost) > 1e-9 or abs(vref.initial_cost - v_init_cost) > 1e-9:
                raise ValueError(
                    f"solution[{idx}] vehicle cost mismatch for id={vid}")

        total_cost += v_init_cost + v_dist_cost * dist_m

    return float(total_cost)
