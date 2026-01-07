# Runner.py
from __future__ import annotations
import random
from typing import Any, Dict, List
import os

from vrp_core import load_config, load_problem_from_config, ensure_dir
from vrp_core.cluster_azam import make_azam_subproblems
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine

CONFIG_PATH = "solver_config.yaml"


def main() -> int:
    cfg: Dict[str, Any] = load_config(CONFIG_PATH)
    set_log_level(str(cfg.get("logger", "INFO")).upper())
    log_info("Using config: %s", CONFIG_PATH)

    full_vrp = load_problem_from_config(cfg)
    engine = VRPSolverEngine(cfg)
    
    
    if cfg['tsp'] == True:
        full_vrp["clustering_seed"] = random.randint(1,1000000)
        engine.prepare([full_vrp])
    else:
        sub_vrps: List[Dict[str, Any]] = make_azam_subproblems(
            full_vrp,
            depot_id=0,
            num_depots=1,
            num_vehicles=len(full_vrp["vehicles"]),
            seed=42,
            capacity_per_cluster=cfg.get("cluster_capacity", None),
        )
        log_info("Built %d subproblems", len(sub_vrps))
        engine.prepare(sub_vrps)
    summary = engine.run()

    root_out = cfg.get("output_dir", "results")
    ensure_dir(root_out)
    import json
    with open(os.path.join(root_out, "run_summary_aggregated.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log_info("Aggregated summary saved to %s", os.path.join(root_out, "run_summary_aggregated.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())