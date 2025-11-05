# vrp_core/decoding.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Literal
import numpy as np
from .models.node import Node
from .models.vehicle import Vehicle
from .metrics import route_load
from .evaluate import eval_routes_cost

def _compositions_nonneg(n: int, k: int):
    def rec(remaining, parts_left, prefix):
        if parts_left == 1:
            yield tuple(prefix + [remaining]); return
        for x in range(remaining + 1):
            yield from rec(remaining - x, parts_left - 1, prefix + [x])
    return rec(n, k, [])

def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    k = min(k, n-k)
    num = 1; den = 1
    for i in range(1, k+1):
        num *= n - (k - i)
        den *= i
    return num // den

def _route_feasible_capacity(route: List[int], nodes: List[Node], cap: float) -> bool:
    load = 0.0 if len(route) <= 2 else float(sum((nodes[i].demand or 0.0) for i in route[1:-1]))
    return load <= cap + 1e-9

def _decode_split_equal(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
    depot = 0
    N, V = len(perm), len(vehicles)

    total_demand = float(sum((nodes[i].demand or 0.0) for i in perm))
    total_capacity = float(sum(v.max_capacity for v in vehicles))
    if total_capacity + 1e-9 < total_demand:
        raise ValueError("Infeasible: total vehicle capacity < total demand.")

    num_splits = _n_choose_k(N + V - 1, V - 1)
    if num_splits > 1_000_000:
        raise ValueError(f"Split explosion: evaluating {num_splits:,} splits; reduce N or V.")

    best = None
    for lengths in _compositions_nonneg(N, V):
        routes: List[List[int]] = []
        cursor = 0
        feasible = True
        for veh, L in zip(vehicles, lengths):
            chunk = perm[cursor: cursor + L]; cursor += L
            route = [depot] + chunk + [depot]
            if not _route_feasible_capacity(route, nodes, veh.max_capacity):
                feasible = False; break
            routes.append(route)
        if not feasible: continue
        tot_cost, tot_dist, tot_time = eval_routes_cost(routes, vrp)
        sig = lengths
        if best is None:
            best = (tot_cost, tot_dist, tot_time, routes, sig)
        else:
            bc, bd, bt, _, bs = best
            if (tot_cost < bc - 1e-9) or \
               (abs(tot_cost - bc) <= 1e-9 and (tot_dist < bd - 1e-6 or
                                               (abs(tot_dist - bd) <= 1e-6 and sig < bs))):
                best = (tot_cost, tot_dist, tot_time, routes, sig)
    if best is None:
        raise ValueError("No feasible split found that respects per-vehicle capacities.")
    return best[3]

def decode_routes(
    perm: List[int],
    vrp: Dict[str, Any],
    mode: Literal["minimize_vehicles", "split_equal"] = "minimize_vehicles",
) -> List[List[int]]:
    if mode == "split_equal":
        return _decode_split_equal(perm, vrp)

    # minimize_vehicles (original)
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
    depot = 0
    routes: List[List[int]] = []
    cursor = 0
    customers = len(perm)

    total_demand = float(sum((nodes[i].demand or 0.0) for i in perm))
    total_capacity = float(sum(v.max_capacity for v in vehicles))
    if total_capacity + 1e-9 < total_demand:
        raise ValueError("Infeasible: total vehicle capacity < total demand.")

    for veh in vehicles:
        cap_left = float(veh.max_capacity)
        route = [depot]
        while cursor < customers:
            node_id = int(perm[cursor])
            dem = float(nodes[node_id].demand or 0.0)
            if dem <= cap_left + 1e-9:
                route.append(node_id); cap_left -= dem; cursor += 1
            else:
                break
        route.append(depot)
        if len(route) > 2:
            routes.append(route)
        if cursor >= customers:
            break

    if cursor < customers:
        raise ValueError("Infeasible with given vehicle count/capacities.")
    return routes
