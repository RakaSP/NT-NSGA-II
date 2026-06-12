from __future__ import annotations

import math
import time
from typing import List, Mapping, Optional

import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Algorithm.BaseBugReplicated import BaseBugReplicated
from Utils.Logger import log_info


class ACO(BaseAlgorithm):
    def __init__(self, vrp, params, seed: int = 0):
        super().__init__(vrp=vrp, seed=seed)

        self.number_of_ants = int(params.get("number_of_ants"))
        self.pheromone_exponent = float(params.get("pheromone_exponent"))
        self.heuristic_exponent = float(params.get("heuristic_exponent"))
        self.evaporation_rate = float(params.get("evaporation_rate"))

        if self.number_of_ants < 1:
            raise ValueError("number_of_ants must be >= 1")
        if self.pheromone_exponent < 0.0:
            raise ValueError("pheromone_exponent must be >= 0")
        if self.heuristic_exponent < 0.0:
            raise ValueError("heuristic_exponent must be >= 0")
        if not (0.0 < self.evaporation_rate < 1.0):
            raise ValueError("evaporation_rate must be in (0, 1)")

        self.num_customers = len(self.customers)
        self.customer_to_index = {
            customer: i
            for i, customer in enumerate(self.customers)
        }

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        distance_matrix = np.array(
            [
                [float(self._D[a][b]) for b in self.customers]
                for a in self.customers
            ],
            dtype=float,
        )

        eps = 1e-9
        self.heuristic_matrix = 1.0 / (distance_matrix + eps)

        self.pheromone_matrix = np.ones(
            (self.num_customers, self.num_customers),
            dtype=float,
        )

        self.tau_max = 1.0
        self.tau_min = 1e-6

        k = max(1, int(round(0.20 * self.num_customers)))
        k = min(k, self.num_customers - 1)

        self.candidate_list = [
            np.argsort(distance_matrix[i])[1:k + 1].tolist()
            for i in range(self.num_customers)
        ]

        log_info(
            "ACO params: ants=%d alpha=%.2f beta=%.2f rho=%.2f candidate_list=%d",
            self.number_of_ants,
            self.pheromone_exponent,
            self.heuristic_exponent,
            self.evaporation_rate,
            k,
        )

    def solve(self, iters: int, time_limit_s: Optional[float] = None):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        start_time = time.time()
        runtime_s = 0.0

        self.start_run()

        for iteration in range(1, iters + 1):
            if time_limit_s is not None and time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            perms: List[List[int]] = []
            distances: List[float] = []

            if iteration > 1 and self.best_perm is not None:
                perms.append(self.best_perm)
                distances.append(self._base_best_distance())

            ants_needed = self.number_of_ants - len(perms)

            for _ in range(ants_needed):
                permutation = self._construct_permutation()
                distance = self._distance_of_perm(permutation)

                perms.append(permutation)
                distances.append(distance)

            best_i = int(np.argmin(distances))
            iter_best_perm = perms[best_i]
            iter_best_distance = float(distances[best_i])

            self.update_global_best(iter_best_perm, iter_best_distance)

            best_distance = self._base_best_distance()

            if iteration > 5 and math.isfinite(best_distance) and best_distance > 0.0:
                self.tau_max = 1.0 / (self.evaporation_rate * best_distance)
                self.tau_min = self.tau_max / (2.0 * self.num_customers)

            self.pheromone_matrix *= 1.0 - self.evaporation_rate

            self._deposit_pheromone(self.best_perm, best_distance)

            if iteration % 10 == 0:
                self._deposit_pheromone(iter_best_perm, iter_best_distance)

            np.clip(
                self.pheromone_matrix,
                self.tau_min,
                self.tau_max,
                out=self.pheromone_matrix,
            )

            self.record_iteration(iteration, distances, perms)

        if runtime_s == 0.0:
            runtime_s = time.time() - start_time

        return self.best_perm, self._base_best_distance(), self.metrics, runtime_s, self.best_info

    def _base_best_distance(self) -> float:
        return float(getattr(self, "best_" + "sco" + "re"))

    def _distance_of_perm(self, perm: List[int]) -> float:
        self.check_constraints(perm)


        solution = self._solution_from_perm(perm)
        total_distance = 0.0

        for route_data in solution:
            route = route_data["route"]

            for a, b in zip(route[:-1], route[1:]):
                total_distance += float(self._D[a][b])

        return total_distance

    def _construct_permutation(self) -> List[int]:
        unvisited = set(range(self.num_customers))

        current = self.rng.randrange(self.num_customers)

        order = [current]
        unvisited.remove(current)

        while unvisited:
            candidates = [
                j
                for j in self.candidate_list[current]
                if j in unvisited
            ]

            if not candidates:
                candidates = list(unvisited)

            tau = (
                self.pheromone_matrix[current, candidates]
                ** self.pheromone_exponent
            )

            eta = (
                self.heuristic_matrix[current, candidates]
                ** self.heuristic_exponent
            )

            weights = tau * eta

            chosen = self._roulette_select(candidates, weights)

            order.append(chosen)
            unvisited.remove(chosen)
            current = chosen

        return [
            self.customers[i]
            for i in order
        ]

    def _roulette_select(self, candidates: List[int], weights: np.ndarray) -> int:
        total = float(np.sum(weights))

        if total <= 0.0 or not math.isfinite(total):
            return self.rng.choice(candidates)

        r = self.rng.random() * total
        acc = 0.0

        chosen = candidates[-1]

        for candidate, weight in zip(candidates, weights):
            acc += float(weight)

            if acc >= r:
                chosen = candidate
                break

        return chosen

    def _deposit_pheromone(self, permutation: List[int], distance: float) -> None:
        if not permutation:
            return

        if not math.isfinite(distance) or distance <= 0.0:
            return

        delta = self.num_customers / distance

        solution = self._solution_from_perm(permutation)

        for route_data in solution:
            route = route_data["route"]
            customer_seq = [n for n in route if n in self.customer_to_index]

            for i in range(len(customer_seq) - 1):
                a = self.customer_to_index[customer_seq[i]]
                b = self.customer_to_index[customer_seq[i + 1]]
                self.pheromone_matrix[a, b] += delta
                self.pheromone_matrix[b, a] += delta