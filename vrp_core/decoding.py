from __future__ import annotations
from typing import Any, Dict, List, Tuple, Mapping
import math

from .models.node import node
from .models.vehicle import vehicle

INF = float("inf")

def _nodes_by_id(nodes: List[node]) -> Dict[int, node]:
    return {int(n.id): n for n in nodes}

def decode_minimize(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:
    """
    ID-based decode: 'perm' is a list of customer IDs (excluding depot ID).
    Returns list of routes as lists of IDs starting/ending with depot ID.
    """
    nodes: List[node] = vrp["nodes"]
    vehicles: List[vehicle] = vrp["vehicles"]
    depot_id = int(nodes[0].id)

    routes: List[List[int]] = []
    cursor = 0
    customers = len(perm)

    by_id = _nodes_by_id(nodes)
    total_demand = sum((by_id[i].demand or 0.0) for i in perm)
    total_cap = sum(v.max_capacity for v in vehicles)
    if total_cap + 1e-9 < total_demand:
        raise ValueError("Infeasible: total vehicle capacity < total demand.")

    for veh in vehicles:
        cap_left = float(veh.max_capacity)
        route = [depot_id]
        while cursor < customers:
            nid = int(perm[cursor])
            dem = float(by_id[nid].demand or 0.0)
            if dem <= cap_left + 1e-9:
                route.append(nid); cap_left -= dem; cursor += 1
            else:
                break
        route.append(depot_id)
        if len(route) > 2:
            routes.append(route)
        if cursor >= customers:
            break

    if cursor < customers:
        raise ValueError("Infeasible with given vehicle count/capacities.")
    return routes


def decode_split_equal(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:
    """
    Optimal V-way contiguous partition via DP (O(V * N^2)), ID-based.
    Empty chunk -> [depot,depot] (still pays vehicle.initial_cost to match previous behavior).
    """
    nodes: List[node] = vrp["nodes"]
    vehicles: List[vehicle] = vrp["vehicles"]
    D: Mapping[int, Mapping[int, float]] = vrp["D"]  # ID->ID->distance (meters)
    depot_id = int(nodes[0].id)

    N = len(perm)
    V = len(vehicles)

    by_id = _nodes_by_id(nodes)

    # --- quick infeasibility check ---
    total_demand = float(sum((by_id[i].demand or 0.0) for i in perm))
    total_capacity = float(sum(v.max_capacity for v in vehicles))
    if total_capacity + 1e-9 < total_demand:
        raise ValueError("Infeasible: total vehicle capacity < total demand.")

    # --- prefix sums for demand ---
    demand_of = [float(by_id[i].demand or 0.0) for i in perm]
    pref_dem = [0.0]
    for d in demand_of:
        pref_dem.append(pref_dem[-1] + d)
    def seg_demand(i: int, j: int) -> float:
        return pref_dem[j] - pref_dem[i]

    # --- prefix for path edges along perm to get chain distance in O(1) ---
    prefix_edge = [0.0] * (N)
    for t in range(1, N):
        a, b = perm[t-1], perm[t]
        prefix_edge[t] = prefix_edge[t-1] + float(D[a][b])

    def chain_distance(i: int, j: int) -> float:
        """Distance of perm[i] -> ... -> perm[j-1] (no depot edges)."""
        if j - i <= 1:
            return 0.0
        return prefix_edge[j-1] - prefix_edge[i]

    def route_distance(i: int, j: int) -> float:
        """Full route distance [depot, perm[i..j-1], depot]; if i==j (empty) -> 0."""
        if i == j:
            return 0.0
        return float(D[depot_id][perm[i]]) + chain_distance(i, j) + float(D[perm[j-1]][depot_id])

    # --- precompute distances ---
    seg_dist = [[0.0]*(N+1) for _ in range(N+1)]
    for i in range(N+1):
        for j in range(i, N+1):
            seg_dist[i][j] = route_distance(i, j)

    # --- per-vehicle segment costs & capacity feasibility ---
    seg_cost = [[[INF]*(N+1) for _ in range(N+1)] for _ in range(V)]
    for v in range(V):
        init_c = float(vehicles[v].initial_cost)
        dist_c = float(vehicles[v].distance_cost)
        cap = float(vehicles[v].max_capacity)
        for i in range(N+1):
            for j in range(i, N+1):
                dem = seg_demand(i, j)
                if dem <= cap + 1e-9:
                    seg_cost[v][i][j] = init_c + dist_c * seg_dist[i][j]

    # --- DP over (v, j) ---
    dp_cost = [[INF]*(N+1) for _ in range(V+1)]
    dp_dist = [[INF]*(N+1) for _ in range(V+1)]
    prev_cut = [[-1]*(N+1) for _ in range(V+1)]

    dp_cost[0][0] = 0.0
    dp_dist[0][0] = 0.0

    for v in range(1, V+1):
        for j in range(0, N+1):
            best_c, best_d, best_i = INF, INF, -1
            for i in range(0, j+1):
                sc = seg_cost[v-1][i][j]
                if sc == INF:
                    continue
                pc = dp_cost[v-1][i]
                if pc == INF:
                    continue
                cand_c = pc + sc
                cand_d = dp_dist[v-1][i] + seg_dist[i][j]
                if (cand_c < best_c - 1e-9) or (abs(cand_c - best_c) <= 1e-9 and cand_d < best_d - 1e-6):
                    best_c, best_d, best_i = cand_c, cand_d, i
            dp_cost[v][j] = best_c
            dp_dist[v][j] = best_d
            prev_cut[v][j] = best_i

    if not math.isfinite(dp_cost[V][N]):
        raise ValueError("No feasible split found for given capacities.")

    # --- reconstruct cuts ---
    cuts: List[Tuple[int,int]] = []
    j = N
    for v in range(V, 0, -1):
        i = prev_cut[v][j]
        if i < 0:
            raise RuntimeError("Backtrack failed; inconsistent DP state.")
        cuts.append((i, j))
        j = i
    cuts.reverse()

    # --- build routes in vehicle order ---
    routes: List[List[int]] = []
    for (i, j), veh in zip(cuts, vehicles):
        if i == j:
            routes.append([depot_id, depot_id])
        else:
            chunk = [perm[t] for t in range(i, j)]
            routes.append([depot_id] + chunk + [depot_id])
    return routes
