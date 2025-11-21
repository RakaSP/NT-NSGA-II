# vrp_core/tsplib_io.py
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import tsplib95
from .models.node import node
from .models.vehicle import vehicle
from .metrics import build_time_dict_from_distance  # <-- dict-based time builder

def _build_tsplib_distance_dict(prob) -> Dict[int, Dict[int, float]]:
    """
    Build symmetric distance dict D[id1][id2] from a TSPLIB problem.
    Uses TSPLIB node IDs directly (no remapping).
    """
    ids = sorted(prob.get_nodes())  # TSPLIB node IDs (usually 1..N)
    D: Dict[int, Dict[int, float]] = {i: {} for i in ids}
    for i_idx, ia in enumerate(ids):
        D[ia][ia] = 0.0
        for j_idx in range(i_idx + 1, len(ids)):
            ib = ids[j_idx]
            w = prob.get_weight(ia, ib)
            dij = float(w)
            D[ia][ib] = dij
            D[ib][ia] = dij
    return D

def _nodes_from_tsplib(prob, depot_id: int) -> List[node]:
    """
    Create node objects with IDs equal to TSPLIB node IDs.
    Depot appears first in the returned list, but IDs are unchanged.
    """
    ts_ids = sorted(prob.get_nodes())
    if depot_id not in ts_ids:
        raise ValueError(f"tsp_depot_id {depot_id} not in instance nodes")
    ordered = [depot_id] + [n for n in ts_ids if n != depot_id]

    nodes: List[node] = []
    for ts in ordered:
        # TSPLIB coords are typically (x, y); map to lon=x, lat=y
        x, y = prob.node_coords[ts]
        nodes.append(node(id=int(ts), lat=float(y), lon=float(x), demand=0.0))
    return nodes

def load_problem_tsplib(tsp_path: str, vehicles_csv: str, depot_id: int = 1) -> Dict[str, Any]:
    prob = tsplib95.load(tsp_path)
    if not hasattr(prob, "node_coords"):
        raise ValueError("TSLIB file has no NODE_COORD_SECTION")

    # Nodes (IDs preserved)
    nodes = _nodes_from_tsplib(prob, depot_id)

    # Vehicles
    df_veh = pd.read_csv(vehicles_csv).sort_values("id").reset_index(drop=True)
    for col in ("id", "vehicle_name", "initial_cost", "distance_cost"):
        if col not in df_veh.columns:
            raise ValueError("vehicles.csv must have columns: id,vehicle_name,initial_cost,distance_cost[,max_capacity|capacity]")
    if "max_capacity" not in df_veh.columns:
        if "capacity" not in df_veh.columns:
            raise ValueError("vehicles.csv needs max_capacity or capacity")
        df_veh = df_veh.rename(columns={"capacity": "max_capacity"})

    vehicles: List[vehicle] = [
        vehicle(
            id=int(r.id),
            vehicle_name=str(r.vehicle_name),
            max_capacity=float(r.max_capacity),
            initial_cost=float(r.initial_cost),
            distance_cost=float(r.distance_cost),
            used_capacity=None,
            distance=None,
            route=[],  # routes will be lists of node IDs
        )
        for r in df_veh.itertuples(index=False)
    ]

    # Distances/Times as ID-keyed dicts
    D = _build_tsplib_distance_dict(prob)
    T = build_time_dict_from_distance(D)  # seconds; symmetric noise per metrics.py defaults

    return {"nodes": nodes, "vehicles": vehicles, "D": D, "T": T}
