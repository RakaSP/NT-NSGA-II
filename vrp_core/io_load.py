# vrp_core/io_load.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .models.node import Node
from .models.vehicle import Vehicle
from .metrics import build_distance_matrix_from_nodes, build_time_matrix_from_distance

def _float_or_none(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, str) and x.strip() == "": return None
    return float(x)

def load_problem(nodes_csv: str, vehicles_csv: str) -> Dict[str, Any]:
    # nodes
    df_nodes = pd.read_csv(nodes_csv).sort_values("id").reset_index(drop=True)
    for col in ("id","lat","lon"):
        if col not in df_nodes.columns:
            raise ValueError("nodes.csv must have columns: id,lat,lon[,demand]")
    df_nodes["id"] = df_nodes["id"].astype(int)
    df_nodes["lat"] = df_nodes["lat"].astype(float)
    df_nodes["lon"] = df_nodes["lon"].astype(float)
    if "demand" not in df_nodes.columns:
        df_nodes["demand"] = 0.0
    df_nodes["demand"] = df_nodes["demand"].map(_float_or_none)
    if len(df_nodes) and int(df_nodes.loc[0,"id"]) == 0:
        df_nodes.loc[0,"demand"] = 0.0
    if (df_nodes["demand"].fillna(0.0) < 0).any():
        raise ValueError("nodes.demand must be >= 0")

    nodes: List[Node] = [
        Node(id=int(r.id), lat=float(r.lat), lon=float(r.lon),
             demand=None if pd.isna(r.demand) else float(r.demand))
        for r in df_nodes.itertuples(index=False)
    ]

    ids = np.array([n.id for n in nodes], dtype=int)
    N = len(ids)
    if not (ids.min() == 0 and ids.max() == N-1 and np.all(ids == np.arange(N))):
        raise ValueError("Node ids must be contiguous 0..N-1 (depot id==0).")

    # vehicles
    df_veh = pd.read_csv(vehicles_csv).sort_values("id").reset_index(drop=True)
    for col in ("id","vehicle_name","initial_cost","distance_cost"):
        if col not in df_veh.columns:
            raise ValueError("vehicles.csv must have columns: id,vehicle_name,initial_cost,distance_cost[,max_capacity|capacity]")
    if "max_capacity" not in df_veh.columns:
        if "capacity" not in df_veh.columns:
            raise ValueError("vehicles.csv needs max_capacity or capacity")
        df_veh = df_veh.rename(columns={"capacity": "max_capacity"})
    df_veh["id"] = df_veh["id"].astype(int)
    df_veh["vehicle_name"] = df_veh["vehicle_name"].astype(str)
    df_veh["max_capacity"] = df_veh["max_capacity"].astype(float)
    df_veh["initial_cost"] = df_veh["initial_cost"].astype(float)
    df_veh["distance_cost"] = df_veh["distance_cost"].astype(float)
    if (df_veh["max_capacity"]<0).any() or (df_veh["initial_cost"]<0).any() or (df_veh["distance_cost"]<0).any():
        raise ValueError("vehicle capacities/costs must be >= 0")

    vehicles: List[Vehicle] = [
        Vehicle(
            id=int(r.id),
            vehicle_name=str(r.vehicle_name),
            max_capacity=float(r.max_capacity),
            initial_cost=float(r.initial_cost),
            distance_cost=float(r.distance_cost),
            used_capacity=None, distance=None, route=[],
        )
        for r in df_veh.itertuples(index=False)
    ]

    D = build_distance_matrix_from_nodes(nodes)
    T = build_time_matrix_from_distance(D)
    return {"nodes": nodes, "vehicles": vehicles, "D": D, "T": T}

def load_problem_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("tsp"):
        from .tsplib_io import load_problem_tsplib
        return load_problem_tsplib(str(cfg["tsp_file"]),
                                   str(cfg["vehicles_csv"]),
                                   int(cfg.get("tsp_depot_id", 1)))
    return load_problem(str(cfg["nodes"]), str(cfg["vehicles"]))
