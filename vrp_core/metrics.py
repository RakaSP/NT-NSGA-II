# vrp_core/metrics.py
from __future__ import annotations
import math, random
from typing import List, Optional
import numpy as np
from .models.node import Node

EARTH_R = 6_371_008.8  # meters

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_distance_matrix_from_nodes(nodes: List[Node]) -> np.ndarray:
    n = len(nodes)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        li, oi = nodes[i].lat, nodes[i].lon
        for j in range(i+1, n):
            d = _haversine_m(li, oi, nodes[j].lat, nodes[j].lon)
            D[i, j] = D[j, i] = d
    return D

def build_time_matrix_from_distance(D: np.ndarray) -> np.ndarray:
    speed_kmh = 50.0
    time_noise_std = 0.1
    time_seed = random.randint(1, 100000)
    speed_ms = speed_kmh * 1000 / 3600
    T_base = D / speed_ms
    rng = np.random.RandomState(time_seed)
    noise = rng.normal(1.0, time_noise_std, D.shape)
    noise = (noise + noise.T) / 2
    T = T_base * noise
    T = np.maximum(T, 0)
    np.fill_diagonal(T, 0)
    return T

def route_distance(route: List[int], D: np.ndarray) -> float:
    return float(sum(float(D[route[i], route[i+1]]) for i in range(len(route)-1)))

def route_time(route: List[int], T: np.ndarray) -> float:
    return float(sum(float(T[route[i], route[i+1]]) for i in range(len(route)-1)))

def route_load(route: List[int], nodes: List[Node]) -> float:
    if len(route) <= 2:
        return 0.0
    return float(sum((nodes[i].demand or 0.0) for i in route[1:-1]))
