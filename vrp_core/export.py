# vrp_core/export.py
from __future__ import annotations

import json
import os
import time
import math
import platform
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from .metrics import route_distance, route_time


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _leg_dt(
    a: int,
    b: int,
    D: Mapping[int, Mapping[int, float]],
    T: Mapping[int, Mapping[int, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (distance_m, time_s) for edge a->b from matrices, None if missing."""
    d: Optional[float]
    t: Optional[float]
    try:
        d = float(D[a][b])
    except Exception:
        d = None
    try:
        t = float(T[a][b])
    except Exception:
        t = None
    return d, t


def compute_total_cost_time(routes: List[List[int]], vrp: Dict[str, Any]) -> Tuple[float, float]:
    """
    Always computes:
      - total_cost = sum(vehicle.initial_cost + vehicle.distance_cost * route_distance)
      - total_time = sum(route_time)
    This does NOT depend on the optimization scorer.
    """
    D: Mapping[int, Mapping[int, float]] = vrp["D"]
    T: Mapping[int, Mapping[int, float]] = vrp["T"]
    vehicles = vrp["vehicles"]

    total_cost = 0.0
    total_time = 0.0

    for r, veh in zip(routes, vehicles):
        dist_m = float(route_distance(r, D))
        time_s = float(route_time(r, T))
        total_time += time_s
        total_cost += float(veh.initial_cost) + float(veh.distance_cost) * dist_m

    return float(total_cost), float(total_time)


def routes_to_json(routes: List[List[int]], vrp: Dict[str, Any]) -> Dict[str, Any]:
    D: Mapping[int, Mapping[int, float]] = vrp["D"]
    T: Mapping[int, Mapping[int, float]] = vrp["T"]
    vehicles = vrp["vehicles"]

    total_cost, total_time = compute_total_cost_time(routes, vrp)

    out: Dict[str, Any] = {
        "total_cost": total_cost,
        "total_time": total_time,
        "routes": [],
    }

    for r, veh in zip(routes, vehicles):
        dist_m = float(route_distance(r, D))
        time_s = float(route_time(r, T))
        route_cost = float(veh.initial_cost) + float(veh.distance_cost) * dist_m

        # ---- per-leg details from D/T ----
        legs: List[Dict[str, Any]] = []
        dist_sum = 0.0
        time_sum = 0.0
        have_d = False
        have_t = False

        for a, b in zip(r[:-1], r[1:]):
            d, t = _leg_dt(int(a), int(b), D, T)
            legs.append(
                {
                    "from": int(a),
                    "to": int(b),
                    "distance_m": d,
                    "time_s": t,
                }
            )
            if d is not None:
                dist_sum += d
                have_d = True
            if t is not None:
                time_sum += t
                have_t = True

        out["routes"].append(
            {
                "vehicle_id": veh.id,
                "vehicle_name": veh.vehicle_name,
                "distance_m": dist_m,
                "time_s": time_s,
                "route_cost": route_cost,
                "stops_by_id": list(r),

                # per-leg breakdown + consistency checks
                "legs": legs,
                "distance_m_legs_sum": dist_sum if have_d else None,
                "time_s_legs_sum": time_sum if have_t else None,
                "distance_m_diff": (dist_m - dist_sum) if have_d else None,
                "time_s_diff": (time_s - time_sum) if have_t else None,
            }
        )

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
    vehicles = vrp["vehicles"]

    rows = []
    for r, veh in zip(routes, vehicles):
        dist_m = float(route_distance(r, D))
        time_s = float(route_time(r, T))
        cost = float(veh.initial_cost) + float(veh.distance_cost) * dist_m
        rows.append(
            {
                "vehicle_id": veh.id,
                "vehicle_name": veh.vehicle_name,
                "num_stops_including_depot": len(r),
                "distance_m": dist_m,
                "time_s": time_s,
                "initial_cost": float(veh.initial_cost),
                "distance_cost": float(veh.distance_cost),
                "route_cost": float(cost),
                "stops_by_id": list(r),
            }
        )

    df = pd.DataFrame(rows)
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path


# ==========================================================
# METADATA (extra output file to replicate a run)
# ==========================================================

def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return _jsonable(x.item())
        except Exception:
            pass
    try:
        return str(x)
    except Exception:
        return repr(x)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _infer_node_ids(vrp: Dict[str, Any], max_sample: int = 200) -> Dict[str, Any]:
    """
    Prefer vrp['D'] keys. Fallback to vrp['nodes'] list or vrp['customers'].
    We store only a small sample so metadata stays small.
    """
    ids: Optional[List[int]] = None

    D = vrp.get("D")
    if isinstance(D, dict) and D:
        try:
            ids = sorted(int(k) for k in D.keys())
        except Exception:
            try:
                ids = sorted(list(D.keys()))
            except Exception:
                ids = None

    if ids is None:
        nodes = vrp.get("nodes")
        if isinstance(nodes, list):
            try:
                ids = sorted(int(n.get("id")) for n in nodes if isinstance(n, dict) and "id" in n)
            except Exception:
                ids = None

    if ids is None:
        customers = vrp.get("customers")
        if isinstance(customers, list):
            try:
                ids = sorted(int(c) for c in customers)
            except Exception:
                ids = None

    if ids is None:
        return {"num_nodes": None, "node_ids_sample": None, "node_ids_sample_truncated": False}

    truncated = len(ids) > max_sample
    return {
        "num_nodes": int(len(ids)),
        "node_ids_sample": ids[:max_sample],
        "node_ids_sample_truncated": bool(truncated),
    }


def _vehicles_meta(vrp: Dict[str, Any]) -> List[Dict[str, Any]]:
    vehicles = vrp.get("vehicles", [])
    if not isinstance(vehicles, list):
        return []
    out: List[Dict[str, Any]] = []
    for v in vehicles:
        out.append(
            {
                "id": _jsonable(getattr(v, "id", None)),
                "vehicle_name": _jsonable(getattr(v, "vehicle_name", None)),
                "initial_cost": _safe_float(getattr(v, "initial_cost", None)),
                "distance_cost": _safe_float(getattr(v, "distance_cost", None)),
            }
        )
    return out


def write_metadata_json(
    output_dir: str,
    filename: str,
    *,
    config: Dict[str, Any],
    vrp: Dict[str, Any],
    runtime_s: Optional[float] = None,
    iterations: Optional[int] = None,
    algorithm: Optional[str] = None,
    scorer: Optional[str] = None,
    algo_params: Optional[Dict[str, Any]] = None,
    cluster_id: Optional[int] = None,
    run_index: Optional[int] = None,
    epoch: Optional[int] = None,
) -> str:
    """
    Writes an extra JSON file (e.g. metadata.json) to help replicate the run.

    Requirements you asked:
      - clustering_seed == seed (we set seed := vrp['clustering_seed'])
      - store runtime, seed, nodes, vehicles, algorithm, params, etc.
    """
    ensure_dir(output_dir)

    clustering_seed = vrp.get("clustering_seed", None)
    seed = clustering_seed  # MUST match, per your request

    node_info = _infer_node_ids(vrp)
    vehicles_list = vrp.get("vehicles", [])
    vehicles_count = int(len(vehicles_list)) if isinstance(vehicles_list, list) else None

    # best-effort input sources
    nodes_src = config.get("nodes", None)
    vehicles_src = config.get("vehicles", None)
    if vehicles_src is None:
        vehicles_src = config.get("vehicles_csv", None)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "run": {
            "output_dir": str(output_dir),
            "runtime_s": _safe_float(runtime_s),
            "iterations": int(iterations) if iterations is not None else None,
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "run_index": int(run_index) if run_index is not None else None,
            "epoch": int(epoch) if epoch is not None else None,
        },

        # seeds
        "seed": _jsonable(seed),
        "clustering_seed": _jsonable(clustering_seed),

        # problem
        "problem": {
            "nodes_source": _jsonable(nodes_src),
            "vehicles_source": _jsonable(vehicles_src),

            "num_nodes": node_info["num_nodes"],
            "node_ids_sample": node_info["node_ids_sample"],
            "node_ids_sample_truncated": node_info["node_ids_sample_truncated"],

            "num_vehicles": vehicles_count,
            "vehicles": _vehicles_meta(vrp),
        },

        # algorithm
        "algorithm": {
            "name": _jsonable(algorithm if algorithm is not None else config.get("algorithm", None)),
            "scorer": _jsonable(scorer if scorer is not None else config.get("scorer", None)),
            "params": _jsonable(algo_params),
        },

        # minimal config snapshot (no giant matrices)
        "config_snapshot": {
            "nodes": _jsonable(config.get("nodes", None)),
            "vehicles": _jsonable(config.get("vehicles", None)),
            "vehicles_csv": _jsonable(config.get("vehicles_csv", None)),
            "tsp": _jsonable(config.get("tsp", None)),
            "tsp_file": _jsonable(config.get("tsp_file", None)),
            "tsp_depot_id": _jsonable(config.get("tsp_depot_id", None)),
            "output_dir": _jsonable(config.get("output_dir", None)),
            "routes_out": _jsonable(config.get("routes_out", None)),
            "summary_out": _jsonable(config.get("summary_out", None)),
            "logger": _jsonable(config.get("logger", None)),
            "time_limit": _jsonable(config.get("time_limit", None)),
            "epochs": _jsonable(config.get("epochs", None)),
            "batch_size": _jsonable(config.get("batch_size", None)),
            "decode_mode": _jsonable(config.get("decode_mode", None)),
            "rl_enabled": _jsonable(config.get("rl_enabled", None)),
            "training_enabled": _jsonable(config.get("training_enabled", None)),
            "second_nn_path": _jsonable(config.get("second_nn_path", None)),
        },

        # env
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
