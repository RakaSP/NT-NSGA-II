# vrp_core/tsplib_io.py
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import tsplib95
from .models.node import Node
from .models.vehicle import Vehicle
from .metrics import build_time_matrix_from_distance

def _build_tsplib_distance_matrix(prob) -> np.ndarray:
    nodes = list(prob.get_nodes())
    N = len(nodes)
    D = np.zeros((N, N), dtype=float)
    for i, ni in enumerate(nodes):
        for j in range(i+1, N):
            nj = nodes[j]
            w = prob.get_weight(ni, nj)
            D[i, j] = D[j, i] = float(w)
    return D

def _nodes_from_tsplib(prob, depot_id: int) -> List[Node]:
    ts_ids = sorted(prob.get_nodes())
    if depot_id not in ts_ids:
        raise ValueError(f"tsp_depot_id {depot_id} not in instance nodes")
    ordered = [depot_id] + [n for n in ts_ids if n != depot_id]
    id_map = {ts: i for i, ts in enumerate(ordered)}
    nodes: List[Node] = []
    for ts in ordered:
        x, y = prob.node_coords[ts]
        nodes.append(Node(id=id_map[ts], lat=float(y), lon=float(x), demand=0.0))
    return nodes

def load_problem_tsplib(tsp_path: str, vehicles_csv: str, depot_id: int = 1) -> Dict[str, Any]:
    prob = tsplib95.load(tsp_path)
    if not hasattr(prob, "node_coords"):
        raise ValueError("TSPLIB file has no NODE_COORD_SECTION")
    nodes = _nodes_from_tsplib(prob, depot_id)
    df_veh = pd.read_csv(vehicles_csv).sort_values("id").reset_index(drop=True)
    for col in ("id","vehicle_name","initial_cost","distance_cost"):
        if col not in df_veh.columns:
            raise ValueError("vehicles.csv must have columns: id,vehicle_name,initial_cost,distance_cost[,max_capacity|capacity]")
    if "max_capacity" not in df_veh.columns:
        if "capacity" not in df_veh.columns:
            raise ValueError("vehicles.csv needs max_capacity or capacity")
        df_veh = df_veh.rename(columns={"capacity": "max_capacity"})
    vehicles: List[Vehicle] = [
        Vehicle(
            id=int(r.id), vehicle_name=str(r.vehicle_name),
            max_capacity=float(r.max_capacity),
            initial_cost=float(r.initial_cost),
            distance_cost=float(r.distance_cost),
            used_capacity=None, distance=None, route=[],
        )
        for r in df_veh.itertuples(index=False)
    ]
    D = _build_tsplib_distance_matrix(prob)
    T = build_time_matrix_from_distance(D)
    return {"nodes": nodes, "vehicles": vehicles, "D": D, "T": T}
