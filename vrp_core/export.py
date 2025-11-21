from __future__ import annotations
import json, os
from typing import Any, Dict, List, Mapping
import pandas as pd
from .metrics import route_distance, route_time, route_load, nodes_by_id

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def routes_to_json(routes: List[List[int]], vrp: Dict[str, Any]) -> Dict[str, Any]:
    D: Mapping[int, Mapping[int, float]] = vrp["D"]
    T: Mapping[int, Mapping[int, float]] = vrp["T"]
    nodes = vrp["nodes"]; vehicles = vrp["vehicles"]
    nodes_map = nodes_by_id(nodes)
    out = {"routes": []}
    for r, veh in zip(routes, vehicles):
        used = route_load(r, nodes_map)
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
            "stops_by_id": list(r),  # route is already a list of IDs
        })
    return out

def write_routes_json(output_dir: str, filename: str, routes: List[List[int]], vrp: Dict[str, Any]) -> str:
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(routes_to_json(routes, vrp), f, indent=2)
    return out_path

def write_summary_csv(output_dir: str, filename: str, routes: List[List[int]], vrp: Dict[str, Any]) -> str:
    D: Mapping[int, Mapping[int, float]] = vrp["D"]
    T: Mapping[int, Mapping[int, float]] = vrp["T"]
    nodes = vrp["nodes"]; vehicles = vrp["vehicles"]
    nodes_map = nodes_by_id(nodes)
    rows = []
    for r, veh in zip(routes, vehicles):
        dist_m = route_distance(r, D)
        time_s = route_time(r, T)
        used = route_load(r, nodes_map)
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
            "stops_by_id": list(r),
        })
    df = pd.DataFrame(rows)
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path
