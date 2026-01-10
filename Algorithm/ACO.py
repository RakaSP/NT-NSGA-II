# Algorithm/ACO.py
from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple, Mapping, Optional
import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class ACO(BaseAlgorithm):
    def __init__(self, vrp, scorer, params, seed: int = 0):
        super().__init__(vrp=vrp, scorer=scorer, seed=seed)

        number_of_ants = int(params.get("number_of_ants"))
        pheromone_exponent = float(params.get("pheromone_exponent"))
        heuristic_exponent = float(params.get("heuristic_exponent"))
        evaporation_rate = float(params.get("evaporation_rate"))

        self.number_of_ants = number_of_ants
        self.pheromone_exponent = pheromone_exponent
        self.heuristic_exponent = heuristic_exponent
        self.evaporation_rate = evaporation_rate

        if self.number_of_ants <= 0:
            raise ValueError("number_of_ants must be > 0")
        if self.pheromone_exponent < 0:
            raise ValueError("pheromone_exponent must be >= 0")
        if self.heuristic_exponent < 0:
            raise ValueError("heuristic_exponent must be >= 0")
        if not (0.0 < self.evaporation_rate < 1.0):
            raise ValueError("evaporation_rate must be in (0, 1)")

        log_info(
            "ACO params: number_of_ants=%d, pheromone_exponent=%.3f, "
            "heuristic_exponent=%.3f, evaporation_rate=%.3f",
            self.number_of_ants, self.pheromone_exponent,
            self.heuristic_exponent, self.evaporation_rate
        )

        self.num_customers = len(self.customers)

        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        customers = self.customers
        C = np.array([[float(D[a][b]) for b in customers] for a in customers], dtype=float)
        epsilon = 1e-9
        self.heuristic_matrix = 1.0 / (C + epsilon)

        self.pheromone_matrix = np.ones((self.num_customers, self.num_customers), dtype=float)

    def solve(self, iters: int, stop_event: Optional[object] = None) -> Tuple[List[int], float, List[dict], float]:
        def _stop() -> bool:
            return stop_event is not None and getattr(stop_event, "is_set", lambda: False)()

        if iters <= 0:
            raise ValueError("iters must be > 0")
        log_info("Iterations: %d", iters)
        self.start_run()

        last_permutations: List[List[int]] = []

        for iteration_index in range(1, iters + 1):
            if _stop():
                break

            permutations: List[List[int]] = []
            for _ in range(self.number_of_ants):
                if _stop():
                    break
                permutations.append(self._construct_permutation())

            if not permutations:
                break

            last_permutations = permutations
            scores = [self.evaluate_perm(p) for p in permutations]

            self.pheromone_matrix *= (1.0 - self.evaporation_rate)

            best_index = int(np.argmin(scores))
            best_perm = permutations[best_index]
            best_score = float(scores[best_index])
            if math.isfinite(best_score) and best_score > 0.0:
                self._deposit_pheromone(best_perm, best_score)

            self.record_iteration(iteration_index, scores, permutations)
            self.update_global_best(best_perm, best_score)

        runtime_seconds = self.finalize()

        if self.best_perm is None:
            if last_permutations:
                self.best_perm = min(last_permutations, key=lambda p: self.evaluate_perm(p))
                self.best_score = float(self.evaluate_perm(self.best_perm))
            else:
                self.best_perm = self._initialize_individual()
                self.best_score = float(self.evaluate_perm(self.best_perm))

        return self.best_perm, float(self.best_score), self.metrics, runtime_seconds

    def _initialize_individual(self) -> List[int]:
        perm = self.customers[:]
        self.rng.shuffle(perm)
        return perm

    def _construct_permutation(self) -> List[int]:
        unvisited_indices = list(range(self.num_customers))
        current_index = self.rng.randrange(self.num_customers)
        visit_order = [current_index]
        unvisited_indices.remove(current_index)

        while unvisited_indices:
            weights = []
            for j in unvisited_indices:
                tau = self.pheromone_matrix[current_index, j] ** self.pheromone_exponent
                eta = self.heuristic_matrix[current_index, j] ** self.heuristic_exponent
                weights.append(tau * eta)

            total_weight = sum(weights)
            if not math.isfinite(total_weight) or total_weight <= 0.0:
                chosen_pos = self.rng.randrange(len(unvisited_indices))
            else:
                r = self.rng.random() * total_weight
                accumulator = 0.0
                chosen_pos = 0
                for k, w in enumerate(weights):
                    accumulator += w
                    if accumulator >= r:
                        chosen_pos = k
                        break

            next_index = unvisited_indices.pop(chosen_pos)
            visit_order.append(next_index)
            current_index = next_index

        return [self.customers[i] for i in visit_order]

    def _deposit_pheromone(self, permutation: List[int], score: float) -> None:
        amount = 1.0 / score
        local_indices = [self.customers.index(cust_id) for cust_id in permutation]
        for i in range(self.num_customers - 1):
            a = local_indices[i]
            b = local_indices[i + 1]
            self.pheromone_matrix[a, b] += amount
            self.pheromone_matrix[b, a] += amount
