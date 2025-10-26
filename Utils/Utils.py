#!/usr/bin/env python3
# Utils.py — minimal
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import os
import json
import math
import random
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# adjust this import to wherever you defined them
from Utils.Node import Node
from Utils.Vehicle import Vehicle


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yml", ".yaml"):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Config must be .yml/.yaml or .json")


# -------- distance helpers (meters) --------
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
        for j in range(i + 1, n):
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

# -------- CSV -> objects + D --------


def _float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return float(x)


def load_problem(nodes_csv: str, vehicles_csv: str) -> Dict[str, Any]:
    # nodes
    df_nodes = pd.read_csv(nodes_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "lat", "lon"):
        if col not in df_nodes.columns:
            raise ValueError(
                "nodes.csv must have columns: id,lat,lon[,demand]")
    df_nodes["id"] = df_nodes["id"].astype(int)
    df_nodes["lat"] = df_nodes["lat"].astype(float)
    df_nodes["lon"] = df_nodes["lon"].astype(float)
    if "demand" not in df_nodes.columns:
        df_nodes["demand"] = 0.0
    df_nodes["demand"] = df_nodes["demand"].map(_float_or_none)

    # enforce: depot id 0 has demand 0
    if len(df_nodes) and int(df_nodes.loc[0, "id"]) == 0:
        df_nodes.loc[0, "demand"] = 0.0

    # no negative demand
    if (df_nodes["demand"].fillna(0.0) < 0).any():
        raise ValueError("nodes.demand must be >= 0")

    nodes: List[Node] = [
        Node(
            id=int(r.id),
            lat=float(r.lat),
            lon=float(r.lon),
            demand=None if pd.isna(r.demand) else float(r.demand),
        )
        for r in df_nodes.itertuples(index=False)
    ]

    # enforce: IDs are contiguous 0..N-1 so id == index (keeps everything simple)
    ids = np.array([n.id for n in nodes], dtype=int)
    N = len(ids)
    if not (ids.min() == 0 and ids.max() == N - 1 and np.all(ids == np.arange(N))):
        raise ValueError(
            "Node ids must be contiguous 0..N-1 (depot id==0). Reindex your CSV if needed.")

    # vehicles
    df_veh = pd.read_csv(vehicles_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "vehicle_name", "initial_cost", "distance_cost"):
        if col not in df_veh.columns:
            raise ValueError(
                "vehicles.csv must have columns: id,vehicle_name,initial_cost,distance_cost[,max_capacity|capacity]")
    if "max_capacity" not in df_veh.columns:
        if "capacity" not in df_veh.columns:
            raise ValueError("vehicles.csv needs max_capacity or capacity")
        df_veh = df_veh.rename(columns={"capacity": "max_capacity"})

    df_veh["id"] = df_veh["id"].astype(int)
    df_veh["vehicle_name"] = df_veh["vehicle_name"].astype(str)
    df_veh["max_capacity"] = df_veh["max_capacity"].astype(float)
    df_veh["initial_cost"] = df_veh["initial_cost"].astype(float)
    df_veh["distance_cost"] = df_veh["distance_cost"].astype(float)
    if (df_veh["max_capacity"] < 0).any() or (df_veh["initial_cost"] < 0).any() or (df_veh["distance_cost"] < 0).any():
        raise ValueError("vehicle capacities/costs must be >= 0")

    vehicles: List[Vehicle] = [
        Vehicle(
            id=int(r.id),
            vehicle_name=str(r.vehicle_name),
            max_capacity=float(r.max_capacity),
            initial_cost=float(r.initial_cost),
            distance_cost=float(r.distance_cost),
            used_capacity=None,
            distance=None,
            route=[],
        )
        for r in df_veh.itertuples(index=False)
    ]

    # distance matrix
    D = build_distance_matrix_from_nodes(nodes)

    # time matrix
    T = build_time_matrix_from_distance(D)

    return {"nodes": nodes, "vehicles": vehicles, "D": D, "T": T}

# --- small helpers kept in Utils to keep Core tiny ---


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _route_distance(route: List[int], D: np.ndarray) -> float:
    dist = 0.0
    for i in range(len(route) - 1):
        dist += float(D[route[i], route[i + 1]])
    return dist


def _route_time(route: List[int], T: np.ndarray) -> float:
    time = 0.0
    for i in range(len(route) - 1):
        time += float(T[route[i], route[i + 1]])
    return time


def _route_load(route: List[int], nodes: List[Node]) -> float:
    if len(route) <= 2:
        return 0.0
    return float(sum((nodes[i].demand or 0.0) for i in route[1:-1]))


def decode_routes(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:
    """
    Greedy split: vehicles in file order; pack by capacity.
    Assumes IDs==indices; depot id==0. Returns node indices including depot at both ends.
    """
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
    depot = 0
    routes: List[List[int]] = []
    cursor = 0
    customers = len(perm)

    total_demand = float(sum((n.demand or 0.0) for n in nodes))
    total_capacity = float(sum(v.max_capacity for v in vehicles))
    if total_capacity + 1e-9 < total_demand:
        raise ValueError("Infeasible: total vehicle capacity < total demand.")

    for veh in vehicles:
        cap_left = float(veh.max_capacity)
        route = [depot]
        while cursor < customers:
            node_id = int(perm[cursor])  # id == index
            dem = float(nodes[node_id].demand or 0.0)
            if dem <= cap_left + 1e-9:
                route.append(node_id)
                cap_left -= dem
                cursor += 1
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


def validate_capacity(routes: List[List[int]], vrp: Dict[str, Any]) -> None:
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
    for r, veh in zip(routes, vehicles):
        used = _route_load(r, nodes)
        if used > veh.max_capacity + 1e-9:
            raise ValueError(
                f"Capacity violated for vehicle {veh.id}: load={used} > cap={veh.max_capacity}"
            )


def eval_routes_cost(routes: List[List[int]], vrp: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Returns: (total_cost, total_distance, total_time)
    route_cost = distance_m * vehicle.distance_cost + vehicle.initial_cost
    solution_cost = sum(route_cost)
    """
    D: np.ndarray = vrp["D"]
    T: np.ndarray = vrp["T"]  # Added time matrix
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]

    total_cost = 0.0
    total_distance = 0.0  # Added total distance
    total_time = 0.0      # Added total time
    
    breakdown = []
    for r, veh in zip(routes, vehicles):
        dist_m = _route_distance(r, D)
        time_s = _route_time(r, T)  # Calculate time for this route
        used = _route_load(r, nodes)
        cost = veh.initial_cost + veh.distance_cost * dist_m
        
        total_cost += cost
        total_distance += dist_m  # Accumulate total distance
        total_time += time_s      # Accumulate total time
        
        breakdown.append({
            "vehicle_id": veh.id,
            "vehicle_name": veh.vehicle_name,
            "capacity_used": used,
            "capacity_max": veh.max_capacity,
            "capacity_utilization": used / veh.max_capacity if veh.max_capacity > 0 else 0.0,
            "stops_by_id": [nodes[i].id for i in r],
            "distance_m": dist_m,
            "time_s": time_s,  # Added time to breakdown
            "initial_cost": veh.initial_cost,
            "distance_cost": veh.distance_cost,
            "route_cost": cost,
        })
    
    return total_cost, total_distance, total_time


def routes_to_json(routes: List[List[int]], vrp: Dict[str, Any]) -> Dict[str, Any]:
    D: np.ndarray = vrp["D"]
    T: np.ndarray = vrp["T"]  # Added time matrix
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]
    out = {"routes": []}
    for r, veh in zip(routes, vehicles):
        used = _route_load(r, nodes)
        dist_m = _route_distance(r, D)
        time_s = _route_time(r, T)
        out["routes"].append({
            "vehicle_id": veh.id,
            "vehicle_name": veh.vehicle_name,
            "capacity_used": used,
            "capacity_max": veh.max_capacity,
            "capacity_utilization": used / veh.max_capacity if veh.max_capacity > 0 else 0.0,
            "distance_m": dist_m,
            "time_s": time_s,  # Added time to JSON output
            "stops_by_id": [nodes[i].id for i in r],
        })
    return out


def write_routes_json(output_dir: str, filename: str, routes: List[List[int]], vrp: Dict[str, Any]) -> str:
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(routes_to_json(routes, vrp), f, indent=2)
    return out_path


def write_summary_csv(output_dir: str, filename: str, routes: List[List[int]], vrp: Dict[str, Any]) -> str:
    D: np.ndarray = vrp["D"]
    T: np.ndarray = vrp["T"]  # Added time matrix
    nodes: List[Node] = vrp["nodes"]
    vehicles: List[Vehicle] = vrp["vehicles"]

    rows = []
    for r, veh in zip(routes, vehicles):
        dist_m = _route_distance(r, D)
        time_s = _route_time(r, T)  # Calculate time for this route
        used = _route_load(r, nodes)
        cost = veh.initial_cost + veh.distance_cost * dist_m
        rows.append({
            "vehicle_id": veh.id,
            "vehicle_name": veh.vehicle_name,
            "num_stops_including_depot": len(r),
            "capacity_used": used,
            "capacity_max": veh.max_capacity,
            "capacity_utilization": used / veh.max_capacity if veh.max_capacity > 0 else 0.0,
            "distance_m": dist_m,
            "time_s": time_s,  # Added time to summary
            "initial_cost": veh.initial_cost,
            "distance_cost": veh.distance_cost,
            "route_cost": cost,
            "stops_by_id": [nodes[i].id for i in r],
        })
    df = pd.DataFrame(rows)
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path