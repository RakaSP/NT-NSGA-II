# vrp_core/metrics.py
from __future__ import annotations
import math
import random
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Optional
import numpy as np
from .models.node import node

EARTH_R = 6_371_008.8  # meters

# ---------------- Geo ----------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------------- Helpers ----------------
def nodes_by_id(nodes: Iterable[node]) -> Dict[int, node]:
    out: Dict[int, node] = {}
    for n in nodes:
        if n.id in out:
            raise ValueError(f"Duplicate node id detected: {n.id}")
        out[n.id] = n
    return out

def _ensure_edge(D: Mapping[int, Mapping[int, float]], a: int, b: int) -> float:
    try:
        return float(D[a][b])
    except KeyError as e:
        # Fail loud with context
        missing = f"{a}->{b}" if a in D else f"{a} (row missing)"
        raise KeyError(f"Distance/Time entry missing for edge {missing}") from e

# ---------------- Builders (ID-KEYED DICTS) ----------------
def build_distance_dict_from_nodes(nodes: Iterable[node]) -> Dict[int, Dict[int, float]]:
    """
    Build symmetric distance map in meters: D[id1][id2].
    """
    lst = list(nodes)
    ids = [n.id for n in lst]
    lats = [n.lat for n in lst]
    lons = [n.lon for n in lst]

    D: Dict[int, Dict[int, float]] = {i: {} for i in ids}
    for i, ia in enumerate(ids):
        D[ia][ia] = 0.0
        la, oa = lats[i], lons[i]
        for j in range(i + 1, len(ids)):
            ib = ids[j]
            d = _haversine_m(la, oa, lats[j], lons[j])
            D[ia][ib] = d
            D[ib][ia] = d
    return D

def build_time_dict_from_distance(
    D: Mapping[int, Mapping[int, float]],
    *,
    speed_kmh: float = 50.0,
    time_noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Dict[int, Dict[int, float]]:
    """
    Build symmetric time map in seconds: T[id1][id2] from distance dict.
    Adds symmetric multiplicative Gaussian noise if time_noise_std > 0.
    """
    # stable ID order for noise symmetrization
    ids = sorted(D.keys())
    n = len(ids)
    speed_ms = speed_kmh * 1000.0 / 3600.0

    # base times
    Tm = np.zeros((n, n), dtype=float)
    for i, ia in enumerate(ids):
        row = D.get(ia)
        if row is None:
            raise KeyError(f"Row missing in distance dict for id {ia}")
        for j, ib in enumerate(ids):
            if i == j:
                continue
            dij = _ensure_edge(D, ia, ib)
            Tm[i, j] = max(dij / speed_ms, 0.0)

    # symmetric noise
    if time_noise_std and time_noise_std > 0.0:
        rng = np.random.RandomState(seed if seed is not None else random.randint(1, 100000))
        noise = rng.normal(1.0, time_noise_std, Tm.shape)
        noise = (noise + noise.T) / 2.0
        Tm = Tm * noise

    np.fill_diagonal(Tm, 0.0)
    Tm = np.maximum(Tm, 0.0)

    # back to dict
    T: Dict[int, Dict[int, float]] = {ia: {} for ia in ids}
    for i, ia in enumerate(ids):
        for j, ib in enumerate(ids):
            T[ia][ib] = float(Tm[i, j])
    return T

# ---------------- Route metrics (ID-BASED) ----------------
def route_distance(route: Sequence[int], D: Mapping[int, Mapping[int, float]]) -> float:
    """
    Sum of D[a][b] along consecutive node ID pairs in route.
    """
    if len(route) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        total += _ensure_edge(D, a, b)
    return float(total)

def route_time(route: Sequence[int], T: Mapping[int, Mapping[int, float]]) -> float:
    """
    Sum of T[a][b] along consecutive node ID pairs in route.
    """
    if len(route) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        total += _ensure_edge(T, a, b)
    return float(total)

def route_load(route: Sequence[int], nodes_map: Mapping[int, node]) -> float:
    """
    Load is sum of demands on internal stops (excluding endpoints).
    Raises if an ID is not present in nodes_map.
    """
    if len(route) <= 2:
        return 0.0
    total = 0.0
    for nid in route[1:-1]:
        try:
            d = nodes_map[nid].demand
        except KeyError as e:
            raise KeyError(f"Unknown node id in route: {nid}") from e
        total += float(d or 0.0)
    return float(total)

# ---------------- Optional adapters (for any legacy APIs) ----------------
def dict_to_matrix(D: Mapping[int, Mapping[int, float]], id_order: Sequence[int]) -> np.ndarray:
    """
    Produce a numpy matrix in the specified ID order (for any solver that still demands arrays).
    """
    n = len(id_order)
    M = np.zeros((n, n), dtype=float)
    for i, a in enumerate(id_order):
        row = D.get(a)
        if row is None:
            raise KeyError(f"Row missing for id {a}")
        for j, b in enumerate(id_order):
            M[i, j] = _ensure_edge(D, a, b) if a != b else 0.0
    return M
