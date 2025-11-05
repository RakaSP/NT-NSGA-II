# vrp_core/export.py
from __future__ import annotations
import json, os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from .metrics import route_distance, route_time, route_load

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def routes_to_json(routes: List[List[int]], vrp: Dict[str, Any]) -> Dict[str, Any]:
    D: np.ndarray = vrp["D"]; T: np.ndarray = vrp["T"]
    nodes = vrp["nodes"]; vehicles = vrp["vehicles"]
    out = {"routes": []}
    for r, veh in zip(routes, vehicles):
        used = route_load(r, nodes)
        dist_m = route_distance(r, D)
        time_s = route_time(r, T)
        out["routes"].append({
            "vehicle_id": veh.id,
            "vehicle_name": veh.vehicle_name,
            "capacity_used": used,
            "capacity_max": veh.max_capacity,
            "capacity_utilization": used / veh.max_capacity if veh.max_capacity > 0 else 0.0,
            "distance_m": dist_m,
            "time_s": time_s,
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
    D = vrp["D"]; T = vrp["T"]
    nodes = vrp["nodes"]; vehicles = vrp["vehicles"]
    rows = []
    for r, veh in zip(routes, vehicles):
        dist_m = route_distance(r, D)
        time_s = route_time(r, T)
        used = route_load(r, nodes)
        cost = veh.initial_cost + veh.distance_cost * dist_m
        rows.append({
            "vehicle_id": veh.id,
            "vehicle_name": veh.vehicle_name,
            "num_stops_including_depot": len(r),
            "capacity_used": used,
            "capacity_max": veh.max_capacity,
            "capacity_utilization": used / veh.max_capacity if veh.max_capacity > 0 else 0.0,
            "distance_m": dist_m,
            "time_s": time_s,
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
