from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from Experiments.Experiments import VRPExperiment

CONFIG_PATH = "solver_config.yaml"
RUNS_PER_ALGO = 1

ENABLE_PROFILING = False
PROFILE_OUTPUT = "profile.prof"
PROFILE_TXT_OUTPUT = "profile_summary.txt"


def run_experiment() -> int:
    exp = VRPExperiment(
        algos=["ntnsga2"],
        config_path=CONFIG_PATH,
        runs_per_algo=RUNS_PER_ALGO,
        pre_cluster_path="results/pre_cluster.csv",
    )

    exp.run_all()
    exp.save_reports()
    return 0


def main() -> int:
    if not ENABLE_PROFILING:
        return run_experiment()

    profiler = cProfile.Profile()

    try:
        profiler.enable()
        result = run_experiment()
        profiler.disable()
    finally:
        profile_path = Path(PROFILE_OUTPUT)
        summary_path = Path(PROFILE_TXT_OUTPUT)

        profiler.dump_stats(profile_path)

        with summary_path.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(80)

        print(f"\nProfiling saved to: {profile_path}")
        print(f"Text summary saved to: {summary_path}")
        print("\nTop runtime functions:")
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(30)

    return result


if __name__ == "__main__":
    raise SystemExit(main())