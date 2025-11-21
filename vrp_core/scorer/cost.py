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

def _route_used_capacity(route_ids: List[int], nodes_map: Mapping[int, node]) -> float:
    if len(route_ids) <= 2:
        return 0.0
    total = 0.0
    for nid in route_ids[1:-1]:  # exclude depot endpoints
        total += float(nodes_map[int(nid)].demand or 0.0)
    return float(total)

def score_solution(
    solution: List[Dict[str, Any]],
    nodes: List[node],
    vehicles: List[vehicle],
    D: Mapping[int, Mapping[int, float]],
    depot_id: int | None = None,
) -> float:
    """
    Score a solution where:
    - Each route in 'solution' is a list of GLOBAL node IDs (not positions).
    - Distances come from dict D[id_a][id_b].
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
            vmax_cap = float(item["max_capacity"])
            route_ids = [int(x) for x in item["route"]]
        except Exception as e:
            raise ValueError(f"solution[{idx}] invalid/missing fields") from e

        if v_dist_cost < 0 or v_init_cost < 0 or vmax_cap < 0:
            raise ValueError(f"solution[{idx}] negative cost/capacity")

        # basic route validation
        if not route_ids:
            raise ValueError("route is empty")
        if route_ids[0] != route_ids[-1]:
            raise ValueError("route must start and end at the same depot ID")
        unknown = [rid for rid in route_ids if int(rid) not in valid_ids]
        if unknown:
            raise ValueError(f"route contains unknown node IDs: {unknown}")

        dist_m = _route_distance(route_ids, D)
        used_cap = _route_used_capacity(route_ids, nodes_map)
        if used_cap > vmax_cap + 1e-9:
            raise ValueError(
                f"solution[{idx}] capacity exceeded: used={used_cap} > max={vmax_cap}"
            )

        # verify vehicle costs if vehicle exists
        if vid in veh_by_id:
            vref = veh_by_id[vid]
            if abs(vref.distance_cost - v_dist_cost) > 1e-9 or abs(vref.initial_cost - v_init_cost) > 1e-9:
                raise ValueError(
                    f"solution[{idx}] vehicle cost mismatch for id={vid}"
                )

        total_cost += v_init_cost + v_dist_cost * dist_m

    return float(total_cost)
