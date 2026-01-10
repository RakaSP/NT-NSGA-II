from __future__ import annotations

from typing import Any, Dict, List, Mapping
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle


def _route_distance(route_ids: List[int], D: Mapping[int, Mapping[int, float]]) -> float:
    if len(route_ids) < 2:
        return 0.0
    dist = 0.0
    for a, b in zip(route_ids[:-1], route_ids[1:]):
        dist += float(D[a][b])
    return float(dist)


def score_solution(
    solution: List[Dict[str, Any]],
    nodes: List[node],
    vehicles: List[vehicle],
    D: Mapping[int, Mapping[int, float]],
    depot_id: int | None = None,
) -> float:
    """
    Capacity-free scoring.

    Expected per-entry fields:
      - vehicle_id
      - vehicle_distance_cost
      - vehicle_initial_cost
      - route (list of GLOBAL node IDs)
    """
    if not nodes:
        raise ValueError("nodes must be non-empty")

    nodes_map: Dict[int, node] = {int(n.id): n for n in nodes}
    valid_ids = set(nodes_map.keys())

    veh_by_id = {int(v.id): v for v in vehicles}
    total_cost = 0.0

    for idx, item in enumerate(solution):
        try:
            vid = int(item["vehicle_id"])
            v_dist_cost = float(item["vehicle_distance_cost"])
            v_init_cost = float(item["vehicle_initial_cost"])
            route_ids = [int(x) for x in item["route"]]
        except Exception as e:
            raise ValueError(f"solution[{idx}] invalid/missing fields") from e

        if v_dist_cost < 0 or v_init_cost < 0:
            raise ValueError(f"solution[{idx}] negative cost")

        if not route_ids:
            raise ValueError("route is empty")
        if route_ids[0] != route_ids[-1]:
            raise ValueError("route must start and end at the same depot ID")
        unknown = [rid for rid in route_ids if int(rid) not in valid_ids]
        if unknown:
            raise ValueError(f"route contains unknown node IDs: {unknown}")

        dist_m = _route_distance(route_ids, D)

        # verify vehicle costs if vehicle exists
        if vid in veh_by_id:
            vref = veh_by_id[vid]
            if abs(vref.distance_cost - v_dist_cost) > 1e-9 or abs(vref.initial_cost - v_init_cost) > 1e-9:
                raise ValueError(f"solution[{idx}] vehicle cost mismatch for id={vid}")

        total_cost += v_init_cost + v_dist_cost * dist_m

    return float(total_cost)
