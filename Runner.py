#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict

from Utils.Utils import load_config
from Utils.Logger import log_info, set_log_level
from Core import VRPSolverEngine

# Just set this to your config file path
CONFIG_PATH = "solver_config.yaml"  # or "config.json"


def main() -> int:
    cfg: Dict[str, Any] = load_config(CONFIG_PATH)

    # set logger early (Core will set it again too, but this ensures early logs)
    set_log_level(str(cfg.get("logger", "INFO")).upper())
    log_info("Using config: %s", CONFIG_PATH)

    engine = VRPSolverEngine(cfg)
    engine.prepare(cfg)
    summary = engine.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
