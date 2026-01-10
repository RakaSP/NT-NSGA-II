# vrp_core/metrics.py
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Mapping, Sequence, Optional, Tuple

import numpy as np
import requests

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


# ---------------- Optional adapters (for any legacy APIs) ----------------
def dict_to_matrix(D: Mapping[int, Mapping[int, float]], id_order: Sequence[int]) -> np.ndarray:

    n = len(id_order)
    M = np.zeros((n, n), dtype=float)
    for i, a in enumerate(id_order):
        row = D.get(a)
        if row is None:
            raise KeyError(f"Row missing for id {a}")
        for j, b in enumerate(id_order):
            M[i, j] = _ensure_edge(D, a, b) if a != b else 0.0
    return M


def build_distance_time_dict_from_osrm(
    nodes: Iterable[node],
    *,
    base_url: str = "https://router.project-osrm.org",
    profile: str = "driving",
    batch_size: int = 50,
    timeout_s: int = 30,
    unreachable: str = "raise",  # "raise" | "inf"
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    """
    Distance+duration matrix using OSRM 'table' service.
    Returns:
      D[id_a][id_b] in meters
      T[id_a][id_b] in seconds
    """
    lst = list(nodes)
    if not lst:
        return {}, {}

    ids = [int(n.id) for n in lst]

    # OSRM expects lon,lat (NOT lat,lon)
    coords = [(float(n.lon), float(n.lat)) for n in lst]
    coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in coords])

    n = len(ids)
    dist_m = np.full((n, n), np.nan, dtype=float)
    dur_s = np.full((n, n), np.nan, dtype=float)

    headers = {
        "User-Agent": "vrp-research/1.0 (keyless-osrm-table)",
        "Accept": "application/json",
    }

    all_idx = list(range(n))
    for start in range(0, n, batch_size):
        src_idx = all_idx[start : start + batch_size]

        params = {
            "annotations": "distance,duration",
            "sources": ";".join(map(str, src_idx)),
            "destinations": ";".join(map(str, all_idx)),
        }

        url = f"{base_url.rstrip('/')}/table/v1/{profile}/{coords_str}"
        r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"OSRM error {r.status_code}: {r.text[:300]}")

        data = r.json()
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM response not ok: {data}")

        distances = data.get("distances")  # meters
        durations = data.get("durations")  # seconds
        if distances is None or durations is None:
            raise RuntimeError(f"OSRM missing distances/durations: keys={list(data.keys())}")

        for local_i, global_i in enumerate(src_idx):
            dist_m[global_i, :] = np.array(distances[local_i], dtype=float)
            dur_s[global_i, :] = np.array(durations[local_i], dtype=float)

    np.fill_diagonal(dist_m, 0.0)
    np.fill_diagonal(dur_s, 0.0)

    # Handle unreachable routes (None -> nan)
    if np.isnan(dist_m).any() or np.isnan(dur_s).any():
        if unreachable == "inf":
            dist_m = np.where(np.isnan(dist_m), float("inf"), dist_m)
            dur_s = np.where(np.isnan(dur_s), float("inf"), dur_s)
        else:
            bad = np.argwhere(np.isnan(dur_s) | np.isnan(dist_m))
            samples = []
            for k in range(min(8, bad.shape[0])):
                i, j = int(bad[k, 0]), int(bad[k, 1])
                samples.append((ids[i], ids[j]))
            raise RuntimeError(
                f"Unreachable pairs returned by OSRM (showing up to 8): {samples}. "
                f"Use unreachable='inf' if you want to keep running."
            )

    D: Dict[int, Dict[int, float]] = {ida: {} for ida in ids}
    T: Dict[int, Dict[int, float]] = {ida: {} for ida in ids}
    for i, ida in enumerate(ids):
        for j, idb in enumerate(ids):
            D[ida][idb] = float(dist_m[i, j])
            T[ida][idb] = float(dur_s[i, j])

    return D, T
