from __future__ import annotations

from Experiments.Experiments import VRPExperiment

CONFIG_PATH = "solver_config.yaml"
RUNS_PER_ALGO = 30

def main() -> int:
    exp = VRPExperiment(algos=["ntnsga2", "nsga2", "ga", "aco", "pso"], config_path=CONFIG_PATH, runs_per_algo=RUNS_PER_ALGO)
    exp.run_all()
    exp.save_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
