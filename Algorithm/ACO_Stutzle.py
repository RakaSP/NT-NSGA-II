# MAX-MIN Ant System (MMAS) implementation.
#
# Reference:
# Dorigo, M. (2007).
# Ant colony optimization.
# Scholarpedia, 2(3), 1461.
# URL: http://www.scholarpedia.org/article/Ant_colony_optimization
#
# Main MMAS reference:
# Stützle, T., & Hoos, H. H. (2000).
# MAX-MIN Ant System.
# Future Generation Computer Systems, 16, 889–914.
# DOI: 10.1016/S0167-739X(00)00043-1


from __future__ import annotations

import math
import time

from typing import List, Mapping, Optional, Tuple

import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class ACO_Stutzle(BaseAlgorithm):
    EPS = 1e-9

    def __init__(self, vrp, params, seed: int = 0):
        super().__init__(vrp=vrp, seed=seed)

        self.num_customers = len(self.customers)

        if self.num_customers <= 0:
            raise ValueError("MMASACO requires at least one customer")

        self.number_of_ants = int(params.get("number_of_ants", self.num_customers))
        self.pheromone_exponent = float(params.get("pheromone_exponent", 1.0))  # alpha
        self.heuristic_exponent = float(params.get("heuristic_exponent", 2.0))  # beta
        self.evaporation_rate = float(params.get("evaporation_rate", 0.02))      # rho
        self.trail_persistence = 1.0 - self.evaporation_rate
        self.pheromone_constant = float(params.get("pheromone_constant", 1.0))  # Q

        # p_best is commonly used in MMAS to estimate tau_min from tau_max.
        # Smaller p_best gives more exploration; typical values are around 0.05.
        self.p_best = float(params.get("p_best", 0.05))

        # MMAS may update pheromone using either iteration-best or global-best.
        # Valid values:
        # - "iteration_best"
        # - "global_best"
        self.best_update = str(params.get("best_update", "iteration_best"))

        if self.number_of_ants <= 0:
            raise ValueError("number_of_ants must be > 0")
        if self.pheromone_exponent < 0.0:
            raise ValueError("pheromone_exponent/alpha must be >= 0")
        if self.heuristic_exponent < 0.0:
            raise ValueError("heuristic_exponent/beta must be >= 0")
        if not (0.0 < self.evaporation_rate < 1.0):
            raise ValueError("evaporation_rate/rho must be in (0, 1)")
        if self.pheromone_constant <= 0.0:
            raise ValueError("pheromone_constant/Q must be > 0")
        if not (0.0 < self.p_best < 1.0):
            raise ValueError("p_best must be in (0, 1)")
        if self.best_update not in {"iteration_best", "global_best"}:
            raise ValueError(
                "best_update must be either 'iteration_best' or 'global_best'"
            )

        self.customer_to_index = {
            customer: i for i, customer in enumerate(self.customers)
        }

        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        cost_matrix = np.array(
            [
                [float(D[a][b]) for b in self.customers]
                for a in self.customers
            ],
            dtype=float,
        )

        self.symmetric_tsp = self._is_symmetric_matrix(cost_matrix)

        self.heuristic_matrix = 1.0 / (cost_matrix + self.EPS)
        np.fill_diagonal(self.heuristic_matrix, 0.0)

        # Before the first best solution exists, use a safe initial bound.
        # It will be recomputed after the first iteration-best solution is found.
        initial_tour = self._nearest_neighbor_initial_tour()
        initial_score = float(self.evaluate_perm(initial_tour))

        if not math.isfinite(initial_score) or initial_score <= 0.0:
            initial_score = 1.0

        self.tau_max = self._compute_tau_max(initial_score)
        self.tau_min = self._compute_tau_min(self.tau_max)

        if self.tau_max <= 0.0:
            raise ValueError("tau_max must be > 0")
        if self.tau_min <= 0.0:
            raise ValueError("tau_min must be > 0")
        if self.tau_min > self.tau_max:
            raise ValueError("tau_min must be <= tau_max")

        # MMAS initializes pheromone trails to tau_max.
        self.pheromone_matrix = np.full(
            (self.num_customers, self.num_customers),
            self.tau_max,
            dtype=float,
        )
        np.fill_diagonal(self.pheromone_matrix, 0.0)

        log_info(
            (
                "MMASACO params: ants=%d alpha=%.2f beta=%.2f "
                "rho=%.2f persistence=%.2f Q=%.2f p_best=%.3f update=%s "
                "tau_min=%.6f tau_max=%.6f"
            ),
            self.number_of_ants,
            self.pheromone_exponent,
            self.heuristic_exponent,
            self.evaporation_rate,
            self.trail_persistence,
            self.pheromone_constant,
            self.p_best,
            self.best_update,
            self.tau_min,
            self.tau_max,
        )

    def solve(self, iters: int, time_limit_s: Optional[float] = None):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        self.start_run()

        start_time = time.time()
        runtime_s = 0.0

        log_info("MMASACO iterations: %d", iters)

        for iteration in range(1, iters + 1):
            if time_limit_s is not None and time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            perms: List[List[int]] = []
            scores: List[float] = []

            for _ in range(self.number_of_ants):
                perm = self._construct_permutation()
                score = float(self.evaluate_perm(perm))

                perms.append(perm)
                scores.append(score)

            best_i = int(np.argmin(scores))
            iteration_best_perm = perms[best_i]
            iteration_best_score = scores[best_i]

            self.update_global_best(iteration_best_perm, iteration_best_score)

            self._update_pheromone_bounds()

            deposit_perm, deposit_score = self._choose_deposit_solution(
                iteration_best_perm,
                iteration_best_score,
            )

            self._evaporate_pheromone()
            self._deposit_pheromone(deposit_perm, deposit_score)
            self._enforce_pheromone_bounds()

            self.record_iteration(iteration, scores, perms)

        if runtime_s == 0.0:
            runtime_s = time.time() - start_time

        return (
            self.best_perm,
            float(self.best_score),
            self.metrics,
            runtime_s,
            self.best_info,
        )

    def _construct_permutation(self) -> List[int]:
        unvisited = set(range(self.num_customers))

        current = self.rng.randrange(self.num_customers)
        order = [current]
        unvisited.remove(current)

        while unvisited:
            candidates = list(unvisited)
            chosen = self._select_next_customer(current, candidates)

            order.append(chosen)
            unvisited.remove(chosen)
            current = chosen

        return [self.customers[i] for i in order]

    def _select_next_customer(self, current: int, candidates: List[int]) -> int:
        tau = self.pheromone_matrix[current, candidates] ** self.pheromone_exponent
        eta = self.heuristic_matrix[current, candidates] ** self.heuristic_exponent

        weights = tau * eta
        total = float(np.sum(weights))

        if total <= 0.0 or not math.isfinite(total):
            return self.rng.choice(candidates)

        threshold = self.rng.random() * total
        cumulative = 0.0

        for candidate, weight in zip(candidates, weights):
            cumulative += float(weight)

            if cumulative >= threshold:
                return candidate

        return candidates[-1]

    def _choose_deposit_solution(
        self,
        iteration_best_perm: List[int],
        iteration_best_score: float,
    ) -> Tuple[List[int], float]:
        if self.best_update == "global_best" and self.best_perm is not None:
            return self.best_perm, float(self.best_score)

        return iteration_best_perm, iteration_best_score

    def _update_pheromone_bounds(self) -> None:
        if self.best_score is None:
            return

        best_score = float(self.best_score)

        if not math.isfinite(best_score) or best_score <= 0.0:
            return

        self.tau_max = self._compute_tau_max(best_score)
        self.tau_min = self._compute_tau_min(self.tau_max)

    def _compute_tau_max(self, best_score: float) -> float:
        return self.pheromone_constant / (
            self.evaporation_rate * best_score + self.EPS
        )

    def _compute_tau_min(self, tau_max: float) -> float:
        n = max(2, self.num_customers)
        avg = n / 2.0
        p_dec = self.p_best ** (1.0 / n)

        denominator = (avg - 1.0) * p_dec

        if denominator <= self.EPS:
            return tau_max

        tau_min = tau_max * (1.0 - p_dec) / denominator

        if not math.isfinite(tau_min) or tau_min <= 0.0:
            return tau_max / 10.0

        if tau_min > tau_max:
            return tau_max

        return tau_min

    def _evaporate_pheromone(self) -> None:
        self.pheromone_matrix *= self.trail_persistence
        np.fill_diagonal(self.pheromone_matrix, 0.0)

    def _deposit_pheromone(self, permutation: List[int], score: float) -> None:
        if not permutation:
            return

        if not math.isfinite(score) or score <= 0.0:
            return

        delta = self.pheromone_constant / score
        indices = [self.customer_to_index[customer] for customer in permutation]

        for i in range(len(indices)):
            a = indices[i]
            b = indices[(i + 1) % len(indices)]

            self.pheromone_matrix[a, b] += delta

            if self.symmetric_tsp:
                self.pheromone_matrix[b, a] += delta

    def _enforce_pheromone_bounds(self) -> None:
        np.clip(
            self.pheromone_matrix,
            self.tau_min,
            self.tau_max,
            out=self.pheromone_matrix,
        )

        np.fill_diagonal(self.pheromone_matrix, 0.0)

    def _nearest_neighbor_initial_tour(self) -> List[int]:
        remaining = set(self.customers)

        current = self.rng.choice(self.customers)
        tour = [current]
        remaining.remove(current)

        while remaining:
            next_city = min(
                remaining,
                key=lambda city: float(self.vrp["D"][current][city]),
            )

            tour.append(next_city)
            remaining.remove(next_city)
            current = next_city

        return tour

    def _is_symmetric_matrix(self, matrix: np.ndarray) -> bool:
        if matrix.shape[0] != matrix.shape[1]:
            return False

        return bool(np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-9))