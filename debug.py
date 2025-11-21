# debug_route_bruteforce.py
# -*- coding: utf-8 -*-
"""
Brute-force debugger for route distances.

Idea
----
Given:
- A set of node IDs (e.g. all customers in one cluster / route),
- Their coordinates (lat, lon) from the nodes CSV,
- One or more target distances (in km),

we:

1. Build a full distance matrix between these nodes using haversine_km.
2. Try *every* possible path:
   - Any ordering of any subset with length >= 2
   - (i.e. we allow paths that do NOT visit all nodes)
3. For each target distance, either:
   - report all paths whose length is within TOLERANCE_KM of the target, or
   - if none exist, report the path(s) whose length is closest to the target.

WARNING: search space grows very fast with #nodes.
Use MAX_SUBSET_SIZE_NODES to cap the length of paths being brute-forced.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------
# CONFIG (adjust these for your case)
# ---------------------------------------------------------------------

# CSV with columns: id, lat, lon[, demand]
NODES_CSV = "Problems/GIB.csv"

# --- Option A: take nodes from one cluster_*/solution_routes.json route
USE_SOLUTION_ROUTE = True                # set False to use MANUAL_NODE_IDS
SOLUTION_ROOT = "results/GIB_NSGA2_base_nsga2_run1"   # base dir that contains cluster_*/
CLUSTER_ID = 1                           # e.g. 1 -> results/.../cluster_1/...
# 1-based index of route in solution_routes.json
ROUTE_INDEX = 1

# --- Option B: explicit node IDs (if USE_SOLUTION_ROUTE = False)
# Example: [0, 5, 7, 10, 13]
MANUAL_NODE_IDS: List[int] = []

# Whether to include depot (id=0) in the search automatically
INCLUDE_DEPOT_IN_SEARCH = False

# Target distances to investigate (km)
# e.g. [12.15, 11.51, 8.1, 14.75, 14.31, 9.09, 54.11]
TARGET_DISTANCES_KM: List[float] = [11.51]

# Numerical tolerance for "exact" match (km)
TOLERANCE_KM = 1e-2

# Maximum path length (number of nodes) to brute force.
# Paths will have length k = 2.. MAX_SUBSET_SIZE_NODES (or up to n_nodes).
MAX_SUBSET_SIZE_NODES = 7

# --- NEW: sequence similarity options --------------------------------
# If True, only keep "exact" matches (within tolerance) whose sequence
# is similar to the original route:
# - subsequence of original customer order, or (optionally)
# - subsequence of reversed original customer order.
USE_ORIGINAL_SEQUENCE_FILTER = True
ALLOW_REVERSED_SEQUENCE = False
PRINT_ORIGINAL_PATH_DISTANCE = True
# ---------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------

EARTH_R = 6_371_000  # meters


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * \
        math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return (2.0 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))) / 1000.0


def path_distance_km(stops: Sequence[int], coords: Dict[int, Tuple[float, float]]) -> float:
    """Sum great-circle distance along consecutive stop pairs."""
    if not stops or len(stops) < 2:
        return 0.0
    dist = 0.0
    for a, b in zip(stops[:-1], stops[1:]):
        if a not in coords or b not in coords:
            continue
        (lat1, lon1) = coords[a]
        (lat2, lon2) = coords[b]
        dist += haversine_km(lat1, lon1, lat2, lon2)
    return float(dist)

# ---------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------


def load_nodes(nodes_csv: str) -> Tuple[pd.DataFrame, Dict[int, Tuple[float, float]]]:
    df = pd.read_csv(nodes_csv)
    need = {"id", "lat", "lon"}
    if not need.issubset(df.columns):
        raise ValueError("nodes CSV must have columns: id, lat, lon[, demand]")
    df = df.sort_values("id").reset_index(drop=True)
    coords = {int(r.id): (float(r.lat), float(r.lon))
              for r in df.itertuples(index=False)}
    return df, coords


def load_solution(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "routes" not in data:
        raise ValueError(f"{path} must contain top-level 'routes'.")
    return data


def get_route_node_ids_from_solution(
    root_dir: str,
    cluster_id: int,
    route_index: int,
) -> List[int]:
    """Load one cluster_*/solution_routes.json and return node IDs for a route."""
    sol_path = os.path.join(
        root_dir, f"cluster_{cluster_id}", "solution_routes.json")
    if not os.path.isfile(sol_path):
        raise FileNotFoundError(f"Solution file not found: {sol_path}")

    sol = load_solution(sol_path)
    routes = sol.get("routes", [])
    if not routes:
        raise ValueError(f"No routes in {sol_path}")
    if route_index < 1 or route_index > len(routes):
        raise IndexError(
            f"ROUTE_INDEX={route_index} is out of range. "
            f"File has {len(routes)} routes."
        )

    route = routes[route_index - 1]
    stops = route.get("stops_by_id") or []
    ids = [int(x) for x in stops]
    return ids

# ---------------------------------------------------------------------
# Distance matrix + brute force
# ---------------------------------------------------------------------


def build_distance_matrix(
    node_ids: Sequence[int],
    coords: Dict[int, Tuple[float, float]],
) -> Tuple[List[int], Dict[int, int], List[List[float]]]:
    """Return (node_ids_list, id_to_index, dist_matrix) where dist_matrix[i][j] is km."""
    ids = list(dict.fromkeys(int(n)
               for n in node_ids))  # unique, preserve order
    n = len(ids)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    dist = [[0.0] * n for _ in range(n)]

    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                continue
            (lat1, lon1) = coords[a]
            (lat2, lon2) = coords[b]
            dist[i][j] = haversine_km(lat1, lon1, lat2, lon2)

    return ids, id_to_idx, dist


def path_length_from_ids(
    node_order: Sequence[int],
    id_to_idx: Dict[int, int],
    dist: List[List[float]],
) -> float:
    """Compute length of a path using a precomputed distance matrix."""
    if len(node_order) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(node_order[:-1], node_order[1:]):
        ia = id_to_idx[a]
        ib = id_to_idx[b]
        total += dist[ia][ib]
    return total


def count_total_paths(n_nodes: int, max_subset: int | None = None) -> int:
    """Return sum_{k=2}^{max_subset} P(n,k)."""
    if n_nodes < 2:
        return 0
    if max_subset is None or max_subset > n_nodes:
        max_subset = n_nodes
    total = 0
    fact = math.factorial
    for k in range(2, max_subset + 1):
        total += fact(n_nodes) // fact(n_nodes - k)
    return total

# --- NEW: sequence similarity helpers --------------------------------


def _is_subsequence(path: Sequence[int], seq: Sequence[int]) -> bool:
    """Return True if 'path' is a subsequence of 'seq' (relative order preserved)."""
    it = iter(seq)
    for x in path:
        for y in it:
            if y == x:
                break
        else:
            # ran out of seq without finding x
            return False
    return True


def respects_original_sequence(
    path: Sequence[int],
    original_order: Sequence[int],
    allow_reversed: bool = True,
) -> bool:
    """Check if path follows original order (or its reverse, if allowed)."""
    if _is_subsequence(path, original_order):
        return True
    if allow_reversed and _is_subsequence(path, list(reversed(original_order))):
        return True
    return False


def brute_force_paths(
    node_ids: Sequence[int],
    id_to_idx: Dict[int, int],
    dist: List[List[float]],
    target_distances: Sequence[float],
    tol: float,
    max_subset_size: int | None = None,
    original_order: Sequence[int] | None = None,
    use_original_sequence_filter: bool = False,
    allow_reversed_sequence: bool = True,
):
    """
    Try all permutations of all subsets (size >=2) of node_ids.
    For each target in target_distances, track:
    - all "exact" paths with |len - target| <= tol
      (optionally filtered to those similar to original_order)
    - closest path(s) otherwise
    """
    ids = list(node_ids)
    n = len(ids)
    if n < 2:
        raise ValueError("Need at least 2 nodes to build a path.")

    if max_subset_size is None or max_subset_size > n:
        max_subset_size = n

    targets = list(target_distances)
    exact_paths = {t: [] for t in targets}
    best_paths = {t: [] for t in targets}
    best_diff = {t: float("inf") for t in targets}

    total_search_space = count_total_paths(n, max_subset_size)
    print(f"[bruteforce] n_nodes = {n}, max_subset_size = {max_subset_size}")
    print(f"[bruteforce] total candidate paths = {total_search_space:,}")
    if use_original_sequence_filter and original_order is not None:
        print(f"[bruteforce] using original sequence filter with order: {list(original_order)} "
              f"(allow_reversed={allow_reversed_sequence})")

    for k in range(2, max_subset_size + 1):
        print(f"[bruteforce] exploring permutations of length k={k} ...")
        for perm in itertools.permutations(ids, k):
            length = path_length_from_ids(perm, id_to_idx, dist)

            for t in targets:
                diff = abs(length - t)

                # exact-ish matches
                if diff <= tol:
                    if (
                        not use_original_sequence_filter
                        or original_order is None
                        or respects_original_sequence(
                            perm,
                            original_order,
                            allow_reversed=allow_reversed_sequence,
                        )
                    ):
                        exact_paths[t].append((perm, length))

                # best-so-far tracking (unchanged, always consider all perms)
                if diff + 1e-12 < best_diff[t]:
                    best_diff[t] = diff
                    best_paths[t] = [(perm, length)]
                elif abs(diff - best_diff[t]) <= 1e-12:
                    best_paths[t].append((perm, length))

    return exact_paths, best_paths, best_diff

# ---------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------


def print_distance_matrix(node_ids: Sequence[int], dist: List[List[float]]):
    """Print distance matrix rounded to 3 decimals."""
    df = pd.DataFrame(dist, index=node_ids, columns=node_ids)
    df = df.round(3)
    print("\n[debug] distance matrix between candidate nodes (km):")
    print(df.to_string())


def print_paths_report(
    node_ids: Sequence[int],
    exact_paths,
    best_paths,
    best_diff,
):
    for t in sorted(best_diff.keys()):
        print("\n" + "=" * 72)
        print(f"Target distance: {t:.4f} km")
        print("-" * 72)

        exact = exact_paths[t]
        if exact:
            print(f"Found {len(exact)} path(s) within tolerance:")
            for perm, length in exact:
                path_str = " -> ".join(str(n) for n in perm)
                print(
                    f"  {path_str}  |  {length:.4f} km  (error = {abs(length - t):.6f} km)")
        else:
            print("No path within tolerance.")
            print(f"Closest distance error: {best_diff[t]:.6f} km")
            bp = best_paths[t]
            print(f"Number of closest path(s) with that error: {len(bp)}")
            for perm, length in bp:
                path_str = " -> ".join(str(n) for n in perm)
                print(
                    f"  {path_str}  |  {length:.4f} km  (error = {abs(length - t):.6f} km)")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    # 1) load coordinates
    nodes_df, coords = load_nodes(NODES_CSV)

    # 2) decide which node IDs to use
    if USE_SOLUTION_ROUTE:
        route_node_ids = get_route_node_ids_from_solution(
            SOLUTION_ROOT, CLUSTER_ID, ROUTE_INDEX
        )
        print(
            f"[info] route nodes from cluster_{CLUSTER_ID}, route_index={ROUTE_INDEX}:")
        print("       ", route_node_ids)
    else:
        if not MANUAL_NODE_IDS:
            raise ValueError(
                "MANUAL_NODE_IDS is empty and USE_SOLUTION_ROUTE=False.")
        route_node_ids = list(MANUAL_NODE_IDS)
        print("[info] using MANUAL_NODE_IDS:")
        print("       ", route_node_ids)

    # Optionally strip depot from route set for search
    node_ids = list(route_node_ids)
    if not INCLUDE_DEPOT_IN_SEARCH:
        node_ids = [n for n in node_ids if n != 0]

    # Optionally add depot back as candidate node
    if INCLUDE_DEPOT_IN_SEARCH and 0 in coords and 0 not in node_ids:
        node_ids = [0] + node_ids

    # Unique & sanity check
    node_ids = list(dict.fromkeys(node_ids))
    if len(node_ids) < 2:
        raise ValueError(
            "After preprocessing, need at least 2 distinct nodes.")

    print(f"[info] candidate nodes for brute-force search: {node_ids}")
    print(f"[info] number of candidate nodes: {len(node_ids)}")
    
        # --- NEW: compute original path distance (respecting INCLUDE_DEPOT_IN_SEARCH)
    if PRINT_ORIGINAL_PATH_DISTANCE:
        # Start from the raw route (with possible repeated depot 0 at start/end)
        original_path_for_distance = [int(n) for n in route_node_ids]

        # If depot is not included in search, also drop it from this path
        if not INCLUDE_DEPOT_IN_SEARCH:
            original_path_for_distance = [n for n in original_path_for_distance if n != 0]

        print("[info] original path used for distance (respecting INCLUDE_DEPOT_IN_SEARCH):")
        print("       ", original_path_for_distance)

        if len(original_path_for_distance) >= 2:
            orig_dist = path_distance_km(original_path_for_distance, coords)
            print(f"[info] original path distance: {orig_dist:.4f} km")
        else:
            print("[info] original path too short to compute distance (len < 2)")


    # Build original_order for sequence similarity (customers-only or with depot)
    original_order: List[int] = []
    if USE_SOLUTION_ROUTE:
        for n in route_node_ids:
            if not INCLUDE_DEPOT_IN_SEARCH and n == 0:
                continue
            if n in node_ids and n not in original_order:
                original_order.append(n)
    else:
        # If manual, we just mirror the node_ids order as original
        original_order = list(node_ids)

    if USE_ORIGINAL_SEQUENCE_FILTER and original_order:
        print("[info] original order used for sequence filter:", original_order)

    # 3) build distance matrix & show it
    ids, id_to_idx, dist = build_distance_matrix(node_ids, coords)
    print_distance_matrix(ids, dist)

    # 4) brute force paths vs targets
    exact_paths, best_paths, best_diff = brute_force_paths(
        ids,
        id_to_idx,
        dist,
        TARGET_DISTANCES_KM,
        tol=TOLERANCE_KM,
        max_subset_size=MAX_SUBSET_SIZE_NODES,
        original_order=original_order if original_order else None,
        use_original_sequence_filter=USE_ORIGINAL_SEQUENCE_FILTER,
        allow_reversed_sequence=ALLOW_REVERSED_SEQUENCE,
    )

    # 5) pretty report
    print_paths_report(ids, exact_paths, best_paths, best_diff)


if __name__ == "__main__":
    main()
