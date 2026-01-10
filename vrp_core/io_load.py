# vrp_core/io_load.py
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from .models.node import node
from .models.vehicle import vehicle
from .metrics import (
    build_distance_dict_from_nodes,
    build_time_dict_from_distance,
    build_distance_time_dict_from_osrm,
)


def load_problem(nodes_csv: str, vehicles_csv: str) -> Dict[str, Any]:
    # ---------- nodes ----------
    df_nodes = pd.read_csv(nodes_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "lat", "lon"):
        if col not in df_nodes.columns:
            raise ValueError("nodes.csv must have columns: id,lat,lon")

    # enforce types
    df_nodes["id"] = df_nodes["id"].astype(int)
    df_nodes["lat"] = df_nodes["lat"].astype(float)
    df_nodes["lon"] = df_nodes["lon"].astype(float)

    # uniqueness of IDs (no index/contiguity assumptions)
    if df_nodes["id"].duplicated().any():
        dupes = df_nodes[df_nodes["id"].duplicated()]["id"].tolist()
        raise ValueError(f"Duplicate node ids found: {dupes}")

    nodes: List[node] = [
        node(
            id=int(r.id),
            lat=float(r.lat),
            lon=float(r.lon),
        )
        for r in df_nodes.itertuples(index=False)
    ]

    # ---------- vehicles ----------
    df_veh = pd.read_csv(vehicles_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "vehicle_name", "initial_cost", "distance_cost"):
        if col not in df_veh.columns:
            raise ValueError("vehicles.csv must have columns: id,vehicle_name,initial_cost,distance_cost")

    df_veh["id"] = df_veh["id"].astype(int)
    df_veh["vehicle_name"] = df_veh["vehicle_name"].astype(str)
    df_veh["initial_cost"] = df_veh["initial_cost"].astype(float)
    df_veh["distance_cost"] = df_veh["distance_cost"].astype(float)

    if (df_veh["initial_cost"] < 0).any() or (df_veh["distance_cost"] < 0).any():
        raise ValueError("vehicle costs must be >= 0")

    vehicles: List[vehicle] = [
        vehicle(
            id=int(r.id),
            vehicle_name=str(r.vehicle_name),
            initial_cost=float(r.initial_cost),
            distance_cost=float(r.distance_cost),
            distance=None,
            route=[],  # routes hold node IDs
        )
        for r in df_veh.itertuples(index=False)
    ]

    # ---------- distance/time as ID-keyed dicts ----------
    D = build_distance_dict_from_nodes(nodes)  # D[id_a][id_b] in meters
    T = build_time_dict_from_distance(D)       # T[id_a][id_b] in seconds

    return {
        "nodes": nodes,
        "nodes_map": {n.id: n for n in nodes},
        "vehicles": vehicles,
        "D": D,
        "T": T,
    }


def load_problem_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("tsp"):
        from .tsplib_io import load_problem_tsplib
        return load_problem_tsplib(
            str(cfg["tsp_file"]),
            str(cfg["vehicles_csv"]),
            int(cfg.get("tsp_depot_id", 1)),
        )
    return load_problem_google(str(cfg["nodes"]), str(cfg["vehicles"]))


def load_problem_google(nodes_csv: str, vehicles_csv: str) -> Dict[str, Any]:
    # ---------- nodes ----------
    df_nodes = pd.read_csv(nodes_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "lat", "lon"):
        if col not in df_nodes.columns:
            raise ValueError("nodes.csv must have columns: id,lat,lon")

    df_nodes["id"] = df_nodes["id"].astype(int)
    df_nodes["lat"] = df_nodes["lat"].astype(float)
    df_nodes["lon"] = df_nodes["lon"].astype(float)

    if df_nodes["id"].duplicated().any():
        dupes = df_nodes[df_nodes["id"].duplicated()]["id"].tolist()
        raise ValueError(f"Duplicate node ids found: {dupes}")

    nodes: List[node] = [
        node(
            id=int(r.id),
            lat=float(r.lat),
            lon=float(r.lon),
        )
        for r in df_nodes.itertuples(index=False)
    ]

    # ---------- vehicles ----------
    df_veh = pd.read_csv(vehicles_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "vehicle_name", "initial_cost", "distance_cost"):
        if col not in df_veh.columns:
            raise ValueError("vehicles.csv must have columns: id,vehicle_name,initial_cost,distance_cost")

    df_veh["id"] = df_veh["id"].astype(int)
    df_veh["vehicle_name"] = df_veh["vehicle_name"].astype(str)
    df_veh["initial_cost"] = df_veh["initial_cost"].astype(float)
    df_veh["distance_cost"] = df_veh["distance_cost"].astype(float)

    if (df_veh["initial_cost"] < 0).any() or (df_veh["distance_cost"] < 0).any():
        raise ValueError("vehicle costs must be >= 0")

    vehicles: List[vehicle] = [
        vehicle(
            id=int(r.id),
            vehicle_name=str(r.vehicle_name),
            initial_cost=float(r.initial_cost),
            distance_cost=float(r.distance_cost),
            distance=None,
            route=[],
        )
        for r in df_veh.itertuples(index=False)
    ]

    # ---------- distance/time from OSRM (keyless) ----------
    D, T = build_distance_time_dict_from_osrm(nodes)

    return {
        "nodes": nodes,
        "nodes_map": {n.id: n for n in nodes},
        "vehicles": vehicles,
        "D": D,
        "T": T,
    }
