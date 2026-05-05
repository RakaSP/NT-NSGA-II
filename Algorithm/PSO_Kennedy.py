# CLASSIC PSO, SOURCE: Kennedy, J., & Eberhart, R. C. (1995). Particle Swarm Optimization. Proceedings of the IEEE International Conference on Neural Networks, 4, 1942–1948. DOI: 10.1109/ICNN.1995.488968.

# Algorithm/ClassicPSO.py
from __future__ import annotations

import time
from typing import List, Mapping, Optional

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace


class PSO_Kennedy(BaseAlgorithm):
    def __init__(self, vrp, params, seed):
        super().__init__(vrp=vrp, seed=seed)

        self.population_size = int(params.get("population_size", 60))
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")

        # Classic global-best PSO parameters.
        # Kennedy and Eberhart's original PSO uses cognitive and social acceleration terms.
        self.cognitive_coeff = float(params.get("cognitive_coeff", 2.0))
        self.social_coeff = float(params.get("social_coeff", 2.0))

        if self.cognitive_coeff < 0.0:
            raise ValueError("cognitive_coeff must be >= 0")
        if self.social_coeff < 0.0:
            raise ValueError("social_coeff must be >= 0")

        self.num_customers = len(self.customers)
        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        log_info(
            "ClassicPSO params: pop=%d c1=%.3f c2=%.3f",
            self.population_size,
            self.cognitive_coeff,
            self.social_coeff,
        )

    # ============================================================
    # Random-key encoding
    # ============================================================

    def _decode_position(self, position: List[float]) -> List[int]:
        """
        Converts a continuous PSO position vector into a customer permutation.

        Example:
            position = [0.30, -1.20, 2.10]
            sorted indices = [1, 0, 2]
            permutation = [customers[1], customers[0], customers[2]]
        """
        order = sorted(range(self.num_customers), key=lambda i: position[i])
        return [self.customers[i] for i in order]

    def _initialize_position(self) -> List[float]:
        """
        Initializes a continuous particle position.

        The values themselves are random keys. Their sorted order determines
        the decoded permutation.
        """
        return [self.rng.random() for _ in range(self.num_customers)]

    def _initialize_velocity(self) -> List[float]:
        """
        Initializes particle velocity.

        Small random velocities are enough because the decoded permutation
        depends on the relative ordering of the position values.
        """
        return [self.rng.uniform(-1.0, 1.0) for _ in range(self.num_customers)]

    def _evaluate_position(self, position: List[float]) -> float:
        perm = self._decode_position(position)
        return float(self.evaluate_perm(perm))

    # ============================================================
    # Main solver
    # ============================================================

    def solve(self, iters: int, time_limit_s: Optional[float] = None):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        start_time = time.time()
        runtime_s = 0.0

        self.start_run()
        log_info("ClassicPSO iterations: %d", iters)

        # --------------------------------------------------------
        # Initialize swarm
        # --------------------------------------------------------

        positions: List[List[float]] = []
        velocities: List[List[float]] = []

        personal_best_positions: List[List[float]] = []
        personal_best_scores: List[float] = []

        for _ in range(self.population_size):
            position = self._initialize_position()
            velocity = self._initialize_velocity()

            score = self._evaluate_position(position)

            positions.append(position)
            velocities.append(velocity)

            personal_best_positions.append(position[:])
            personal_best_scores.append(score)

        # --------------------------------------------------------
        # Initialize global best
        # --------------------------------------------------------

        best_idx = min(range(self.population_size), key=lambda i: personal_best_scores[i])

        global_best_position = personal_best_positions[best_idx][:]
        global_best_score = personal_best_scores[best_idx]

        global_best_perm = self._decode_position(global_best_position)
        self.update_global_best(global_best_perm, global_best_score)

        # --------------------------------------------------------
        # Main PSO loop
        # --------------------------------------------------------

        for iteration in range(1, iters + 1):
            if time_limit_s is not None and time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            current_scores: List[float] = []
            current_perms: List[List[int]] = []

            for i in range(self.population_size):
                position = positions[i]
                velocity = velocities[i]

                # Classic PSO velocity and position update:
                #
                # v = v
                #     + c1 * r1 * (personal_best - current_position)
                #     + c2 * r2 * (global_best - current_position)
                #
                # x = x + v
                for d in range(self.num_customers):
                    r1 = self.rng.random()
                    r2 = self.rng.random()

                    cognitive = (
                        self.cognitive_coeff
                        * r1
                        * (personal_best_positions[i][d] - position[d])
                    )

                    social = (
                        self.social_coeff
                        * r2
                        * (global_best_position[d] - position[d])
                    )

                    velocity[d] = velocity[d] + cognitive + social
                    position[d] = position[d] + velocity[d]

                score = self._evaluate_position(position)
                perm = self._decode_position(position)

                current_scores.append(score)
                current_perms.append(perm)

                # Update personal best
                if score < personal_best_scores[i] - 1e-9:
                    personal_best_positions[i] = position[:]
                    personal_best_scores[i] = score

                    # Update global best
                    if score < global_best_score - 1e-9:
                        global_best_position = position[:]
                        global_best_score = score
                        global_best_perm = perm[:]

                        log_info(
                            "Iter %d: NEW BEST %.6f",
                            iteration,
                            global_best_score,
                        )

            self.update_global_best(global_best_perm, global_best_score)
            self.record_iteration(iteration, current_scores, current_perms)

            if iteration % 10 == 0:
                mean_score = sum(current_scores) / max(1, len(current_scores))
                log_trace(
                    "[ClassicPSO] iter=%d best=%.6f mean=%.6f",
                    iteration,
                    global_best_score,
                    mean_score,
                )

        if runtime_s == 0.0:
            runtime_s = time.time() - start_time

        return self.best_perm, float(self.best_score), self.metrics, runtime_s, self.best_info