#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from typing import Any, Dict
from Core import VRPSolverEngine
from Utils.Utils import load_config
from Utils.Logger import log_info, set_log_level
import copy


def main():
    # Load base configuration
    base_config = load_config("solver_config.yaml")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    algorithms = ["ga", "pso", "aco"]

    print("Starting multi-runner experiment:")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Iterations: 30")
    print("-" * 50)

    # Run experiments
    for iteration in range(1, 31):
        print(f"\n--- Iteration {iteration} ---")

        for algorithm in algorithms:
            print(f"Running {algorithm.upper()}")

            # Create config for this run
            config = copy.deepcopy(base_config)

            # Set problem files
            config['nodes'] = f"Problems/vrp_nodes.csv"
            config['vehicles'] = "Problems/vrp_vehicles.csv"

            # Set algorithm and seed
            config['algorithm'] = algorithm

            # Set output directory
            config['output_dir'] = f"results/{algorithm}_{iteration}"

            # Set logger level
            set_log_level(str(config.get("logger", "INFO")).upper())

            # Run the solver directly
            engine = VRPSolverEngine(config)
            engine.prepare()
            summary = engine.run()

            log_info(
                "DONE | algo=%s iter=%d final_score=%.6f runtime_s=%.3f",
                algorithm,
                iteration,
                float(summary.get("final_score", 0.0)),
                float(summary.get("runtime_s", 0.0)),
            )

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
