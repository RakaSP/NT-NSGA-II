from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Mapping
import numpy as np
import math
from Utils.Logger import log_info

# -------------------------
# helpers
# -------------------------

def _log_cluster_ids(clusters: List[List[int]]) -> None:
    for idx, cl in enumerate(clusters):
        log_info("[cluster %d] ids=%s", idx, cl)


def _get_D(vrp: Dict[str, Any]) -> Mapping[int, Mapping[int, float]]:
    if "D" in vrp: return vrp["D"]
    if "d" in vrp: return vrp["d"]
    raise KeyError("VRP needs distance dict 'D' or 'd' keyed by node ID")


def _get_T(vrp: Dict[str, Any]) -> Optional[Mapping[int, Mapping[int, float]]]:
    if "T" in vrp: return vrp["T"]
    return vrp.get("t", None)


def _coords(vrp: Dict[str, Any]) -> np.ndarray:
    nodes = vrp["nodes"]
    # order is exactly the list order; we keep a pos<->id map nearby
    return np.array([[float(n.lat), float(n.lon)] for n in nodes], dtype=float)


def _pos_maps(vrp: Dict[str, Any]) -> Tuple[Dict[int, int], List[int]]:
    """id2pos, pos2id (pos is index inside vrp['nodes'])."""
    pos2id = [int(n.id) for n in vrp["nodes"]]
    id2pos = {int(nid): i for i, nid in enumerate(pos2id)}
    return id2pos, pos2id


def _demand_of_ids(ids: List[int], vrp: Dict[str, Any]) -> float:
    by_id = {int(n.id): n for n in vrp["nodes"]}
    return float(sum((by_id[i].demand or 0.0) for i in ids if i in by_id))


def _complete_graph_weight_iddict(D: Mapping[int, Mapping[int, float]], ids: List[int]) -> float:
    if len(ids) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(ids)):
        a = ids[i]
        for j in range(i + 1, len(ids)):
            b = ids[j]
            total += float(D[a][b])
    return float(total)


def _kmeans_pp_init(
    X: np.ndarray,
    K: int,
    fixed_first_index: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    k-means++ init.

    Note: if K == 1 and fixed_first_index is not None, this is effectively
    deterministic: the only centroid is fixed at the depot.
    """
    N = len(X)
    if rng is None:
        rng = np.random.default_rng()
    C = np.empty((K, 2), float)
    if fixed_first_index is None:
        C[0] = X[rng.integers(N)]
        start = 1
    else:
        C[0] = X[int(fixed_first_index)]
        start = 1
    for k in range(start, K):
        d2 = np.min(np.linalg.norm(X[:, None, :] - C[None, :k, :], axis=2) ** 2, axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        C[k] = X[rng.choice(N, p=probs)]
    return C


def _assign_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def _recompute_centroids(X: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    C = np.zeros((K, 2), float)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            C[k] = np.nan
        else:
            C[k] = X[idx].mean(axis=0)
    return C


def _kmeans(
    X: np.ndarray,
    K: int,
    initC: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic Lloyd k-means.

    All randomness (for empty-cluster reseeding) goes through `rng` so that
    behavior is fully controlled by the outer `seed`.
    """
    labels = _assign_labels(X, initC)
    C = initC
    for _ in range(max_iter):
        Cnew = _recompute_centroids(X, labels, K)
        # re-seed empties to farthest point from existing centers
        for k in range(K):
            if np.any(np.isnan(Cnew[k])):
                valid = np.array(
                    [i for i in range(K) if not np.any(np.isnan(Cnew[i]))], int
                )
                if valid.size == 0:
                    # controlled by rng, not global np.random
                    Cnew[k] = X[rng.integers(len(X))]
                else:
                    d2 = np.min(
                        np.linalg.norm(
                            X[:, None, :] - Cnew[None, valid, :],
                            axis=2,
                        )
                        ** 2,
                        axis=1,
                    )
                    Cnew[k] = X[int(np.argmax(d2))]
        new_labels = _assign_labels(X, Cnew)
        if np.array_equal(new_labels, labels):
            C = Cnew
            break
        C, labels = Cnew, new_labels
    return labels, C


def _closest_to_point_by_ids(
    X: np.ndarray,
    id2pos: Dict[int, int],
    ids: List[int],
    p: np.ndarray,
) -> int:
    idx = [id2pos[i] for i in ids]
    d = np.linalg.norm(X[idx] - p[None, :], axis=1)
    return ids[int(np.argmin(d))]

# -------------------------
# main clustering (Azam-style)
# -------------------------

def cluster_nodes_azam(
    vrp: Dict[str, Any],
    *,
    depot_id: int = 0,
    num_depots: int = 1,
    num_vehicles: Optional[int] = None,
    seed: Optional[int] = None,
    capacity_per_cluster: Optional[float] = None,  # usually vehicles[0].max_capacity
) -> Dict[str, Any]:
    """
    Azam-style clustering (ID-based):
      1) k-means with depot fixed as a centroid (K=num_depots).
      2) depot per cluster = node closest to centroid.
      3) if vehicles > clusters: iteratively split clusters
         via 2-means with **random** seeding (controlled by `seed`);
         depot stays shared.
      4) optional capacity repair: reassign boundary nodes to neighbor clusters with slack.

    `seed`:
      - Controls all randomness inside this function via numpy Generator.
      - If `seed` is None, a fresh RNG is created.

    Returns:
      {"clusters": List[List[int]], "depots": List[int]}
      (each cluster is a list of GLOBAL node IDs and includes depot_id)
    """
    rng = np.random.default_rng(seed)
    X = _coords(vrp)
    D = _get_D(vrp)
    id2pos, pos2id = _pos_maps(vrp)
    N = len(X)  # kept for readability; not directly used

    if depot_id not in id2pos:
        raise ValueError(f"depot_id {depot_id} not found among node IDs")

    # 1) initial clustering (K=num_depots), fix depot centroid (by INDEX position)
    K = max(1, int(num_depots))
    initC = _kmeans_pp_init(X, K, fixed_first_index=id2pos[depot_id], rng=rng)
    labels, C = _kmeans(X, K, initC, rng=rng)

    clusters: List[List[int]] = []
    depots: List[int] = []
    for k in range(K):
        pos_members = np.where(labels == k)[0].tolist()
        if not pos_members:
            continue
        ids = [pos2id[p] for p in pos_members]
        dep_k = _closest_to_point_by_ids(X, id2pos, ids, C[k])
        depots.append(dep_k)
        # ensure depot_id included and first
        if depot_id not in ids:
            ids = [depot_id] + ids
        else:
            ids = [depot_id] + [i for i in ids if i != depot_id]
        clusters.append(ids)

    # 2) split while vehicles > clusters
    if num_vehicles is not None and num_vehicles > len(clusters):
        need = num_vehicles - len(clusters)
        for _ in range(need):
            # compute weights
            weights = np.array(
                [_complete_graph_weight_iddict(D, cl) for cl in clusters],
                dtype=float,
            )

            # choose which cluster to split USING rng (prob ∝ weight)
            total_w = weights.sum()
            if total_w <= 0 or not np.isfinite(total_w):
                # fall back: uniform random choice
                idx_big = int(rng.integers(len(clusters)))
            else:
                probs = weights / total_w
                idx_big = int(rng.choice(len(clusters), p=probs))

            big = clusters[idx_big]

            # remove depot for splitting; we'll add it to both parts after
            core_ids = [i for i in big if i != depot_id]
            if len(core_ids) <= 1:
                a = big
                b = [depot_id] + ([core_ids[0]] if core_ids else [])
                clusters.pop(idx_big)
                depots.pop(idx_big)
                clusters.extend([a, b])
                depots.extend([depot_id, depot_id])
                continue

            core_pos = [id2pos[i] for i in core_ids]
            coreX = X[core_pos]

            # **** RANDOM 2-means seeding (THIS depends on seed) ****
            # pick two distinct indices in this cluster
            pair_idx = rng.choice(len(core_ids), size=2, replace=False)
            init2 = np.vstack([coreX[pair_idx[0]], coreX[pair_idx[1]]])

            lab2, _ = _kmeans(coreX, 2, init2, rng=rng)

            A = [core_ids[i] for i in np.where(lab2 == 0)[0]]
            B = [core_ids[i] for i in np.where(lab2 == 1)[0]]
            newA = [depot_id] + A
            newB = [depot_id] + B

            clusters.pop(idx_big)
            depots.pop(idx_big)
            clusters.extend([newA, newB])
            depots.extend([depot_id, depot_id])

    # 3) optional capacity repair (simple greedy) using IDs (deterministic)
    if capacity_per_cluster is not None and capacity_per_cluster > 0:
        by_id = {int(n.id): n for n in vrp["nodes"]}

        def cluster_demand(c: List[int]) -> float:
            return float(sum((by_id[i].demand or 0.0) for i in c if i != depot_id))

        changed = True
        while changed:
            changed = False
            for ci, cl in enumerate(clusters):
                dem = cluster_demand(cl)
                if dem <= capacity_per_cluster + 1e-9:
                    continue
                # candidates to move: non-depot nodes, farthest from depot first
                cand = sorted(
                    [i for i in cl if i != depot_id],
                    key=lambda n: float(D[depot_id][n]),
                    reverse=True,
                )
                for n in cand:
                    best_j, best_increase = None, math.inf
                    for cj, other in enumerate(clusters):
                        if cj == ci:
                            continue
                        dem_other = cluster_demand(other)
                        add = float(by_id[n].demand or 0.0)
                        if dem_other + add > capacity_per_cluster + 1e-9:
                            continue
                        inc = float(D[depot_id][n])  # simple proxy
                        if inc < best_increase:
                            best_increase = inc
                            best_j = cj
                    if best_j is not None:
                        cl.remove(n)
                        clusters[best_j].append(n)
                        changed = True
                        break  # restart scanning
            clusters = [c if c else [depot_id] for c in clusters]

    _log_cluster_ids(clusters)
    return {"clusters": clusters, "depots": [depot_id] * len(clusters)}

# -------------------------
# build subproblems (IDs preserved)
# -------------------------

def build_cluster_subproblems(
    vrp: Dict[str, Any],
    clusters: List[List[int]],
    depots: List[int],
) -> List[Dict[str, Any]]:
    """
    Build subproblems with IDs kept as GLOBAL ids (no reindexing).
    We also pass through the FULL D/T dicts so any D[id_i][id_j] lookups remain valid.
    """
    nodes = vrp["nodes"]
    D_full = _get_D(vrp)
    T_full = _get_T(vrp)

    # Fast lookup from global id -> original node object
    by_id = {int(n.id): n for n in nodes}

    subs = []
    for k, (cl, dep) in enumerate(zip(clusters, depots)):
        # ensure depot present and placed first in the list we store (IDs still global)
        if dep not in cl:
            ordered = [dep] + cl[:]
        else:
            ordered = [dep] + [i for i in cl if i != dep]

        # copy the node objects (keeping the SAME global ids)
        sub_nodes = []
        for gid in ordered:
            n = by_id[gid]
            sub_nodes.append(
                type(n)(
                    id=n.id,          # keep GLOBAL id
                    lat=n.lat,
                    lon=n.lon,
                    demand=getattr(n, "demand", None),
                )
            )

        # vehicles: simple example: clone from base[0], one per cluster
        base = vrp["vehicles"][0]
        sub_vehicles = [
            type(base)(
                id=0,
                vehicle_name=f"{base.vehicle_name}-c{k}",
                max_capacity=base.max_capacity,
                initial_cost=base.initial_cost,
                distance_cost=base.distance_cost,
                used_capacity=None,
                distance=None,
                route=[],
            )
        ]

        sub = {
            "nodes": sub_nodes,
            "vehicles": sub_vehicles,
            "D": D_full,
            "T": T_full,
            "cluster_global_ids": ordered,
        }

        log_info("[sub %d] ids=%s", k, ordered)
        subs.append(sub)

    return subs


def make_azam_subproblems(
    vrp: Dict[str, Any],
    *,
    depot_id: int = 0,
    num_depots: int = 1,
    num_vehicles: Optional[int] = None,
    seed: Optional[int] = None,
    capacity_per_cluster: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cl = cluster_nodes_azam(
        vrp,
        depot_id=depot_id,
        num_depots=num_depots,
        num_vehicles=num_vehicles,
        seed=seed,
        capacity_per_cluster=capacity_per_cluster,
    )
    subs = build_cluster_subproblems(vrp, cl["clusters"], cl["depots"])
    return subs
