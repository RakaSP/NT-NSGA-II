from __future__ import annotations

from typing import Any, Dict, List, Tuple, Mapping
import math

from .models.node import node
from .models.vehicle import vehicle

INF = float("inf")


def _nodes_by_id(nodes: List[node]) -> Dict[int, node]:
    return {int(n.id): n for n in nodes}


def decode_minimize(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:

    nodes: List[node] = vrp["nodes"]
    vehicles: List[vehicle] = vrp["vehicles"]
    depot_id = int(nodes[0].id)

    routes: List[List[int]] = []
    cursor = 0
    customers = len(perm)

    for _veh in vehicles:
        route = [depot_id]
        while cursor < customers:
            nid = int(perm[cursor])
            route.append(nid)
            cursor += 1
            # keep filling this vehicle until we decide to stop —
            # without capacities, the only stop condition is "all customers assigned".
            # So this first vehicle will take everything, unless you later add a rule.

        route.append(depot_id)
        if len(route) > 2:
            routes.append(route)

        if cursor >= customers:
            break

    if cursor < customers:
        raise ValueError("Infeasible: not enough vehicles to assign all customers.")

    return routes


def decode_split_equal(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:

    nodes: List[node] = vrp["nodes"]
    vehicles: List[vehicle] = vrp["vehicles"]
    D: Mapping[int, Mapping[int, float]] = vrp["D"]  # ID->ID->distance (meters)
    depot_id = int(nodes[0].id)

    N = len(perm)
    V = len(vehicles)

    # --- prefix for path edges along perm to get chain distance in O(1) ---
    prefix_edge = [0.0] * N
    for t in range(1, N):
        a, b = perm[t - 1], perm[t]
        prefix_edge[t] = prefix_edge[t - 1] + float(D[a][b])

    def chain_distance(i: int, j: int) -> float:
        """Distance of perm[i] -> ... -> perm[j-1] (no depot edges)."""
        if j - i <= 1:
            return 0.0
        return prefix_edge[j - 1] - prefix_edge[i]

    def route_distance(i: int, j: int) -> float:
        """Full route distance [depot, perm[i..j-1], depot]; if i==j (empty) -> 0."""
        if i == j:
            return 0.0
        return float(D[depot_id][perm[i]]) + chain_distance(i, j) + float(D[perm[j - 1]][depot_id])

    # --- precompute distances ---
    seg_dist = [[0.0] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(i, N + 1):
            seg_dist[i][j] = route_distance(i, j)

    # --- per-vehicle segment costs (all feasible) ---
    seg_cost = [[[INF] * (N + 1) for _ in range(N + 1)] for _ in range(V)]
    for v in range(V):
        init_c = float(vehicles[v].initial_cost)
        dist_c = float(vehicles[v].distance_cost)
        for i in range(N + 1):
            for j in range(i, N + 1):
                seg_cost[v][i][j] = init_c + dist_c * seg_dist[i][j]

    # --- DP over (v, j) ---
    dp_cost = [[INF] * (N + 1) for _ in range(V + 1)]
    dp_dist = [[INF] * (N + 1) for _ in range(V + 1)]
    prev_cut = [[-1] * (N + 1) for _ in range(V + 1)]

    dp_cost[0][0] = 0.0
    dp_dist[0][0] = 0.0

    for v in range(1, V + 1):
        for j in range(0, N + 1):
            best_c, best_d, best_i = INF, INF, -1
            for i in range(0, j + 1):
                sc = seg_cost[v - 1][i][j]
                pc = dp_cost[v - 1][i]
                if pc == INF:
                    continue
                cand_c = pc + sc
                cand_d = dp_dist[v - 1][i] + seg_dist[i][j]
                if (cand_c < best_c - 1e-9) or (abs(cand_c - best_c) <= 1e-9 and cand_d < best_d - 1e-6):
                    best_c, best_d, best_i = cand_c, cand_d, i
            dp_cost[v][j] = best_c
            dp_dist[v][j] = best_d
            prev_cut[v][j] = best_i

    if not math.isfinite(dp_cost[V][N]):
        raise ValueError("No feasible split found.")

    # --- reconstruct cuts ---
    cuts: List[Tuple[int, int]] = []
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
    for (i, j), _veh in zip(cuts, vehicles):
        if i == j:
            routes.append([depot_id, depot_id])
        else:
            chunk = [perm[t] for t in range(i, j)]
            routes.append([depot_id] + chunk + [depot_id])
    return routes
