# Algorithm/ACO.py
from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple
import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class ACO(BaseAlgorithm):
    def __init__(self, vrp, scorer, params):
        # Do not pass seed/detail_fn; match GA style and avoid zero-arg super fragility
        BaseAlgorithm.__init__(self, vrp=vrp, scorer=scorer)

        number_of_ants = int(params.get(
            "number_of_ants"))
        pheromone_exponent = float(params.get(
            "pheromone_exponent"))
        heuristic_exponent = float(params.get(
            "heuristic_exponent"))
        evaporation_rate = float(params.get(
            "evaporation_rate"))

        # Assign + validate
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

        log_info("ACO params: number_of_ants=%d, pheromone_exponent=%.3f, heuristic_exponent=%.3f, evaporation_rate=%.3f",
                 self.number_of_ants, self.pheromone_exponent, self.heuristic_exponent, self.evaporation_rate)
        # Precompute sizes and tables
        self.num_customers = len(self.customers)

        # Heuristic matrix: pairwise customer-to-customer inverse distances
        distance_matrix = self.vrp["D"]
        customers = self.customers
        C = distance_matrix[np.ix_(customers, customers)].astype(float)
        epsilon = 1e-9
        self.heuristic_matrix = 1.0 / (C + epsilon)

        # Pheromone matrix
        self.pheromone_matrix = np.ones(
            (self.num_customers, self.num_customers), dtype=float
        )

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")
        log_info("Iterations: %d", iters)
        self.start_run()

        for iteration_index in range(1, iters + 1):
            permutations = [self._construct_permutation()
                            for _ in range(self.number_of_ants)]
            scores = [self.evaluate_perm(p) for p in permutations]

            # Evaporation
            self.pheromone_matrix *= (1.0 - self.evaporation_rate)

            # Deposit pheromone only for the iteration-best (reduces noise)
            best_index = int(np.argmin(scores))
            best_perm = permutations[best_index]
            best_score = float(scores[best_index])
            if math.isfinite(best_score) and best_score > 0.0:
                self._deposit_pheromone(best_perm, best_score)

            self.record_iteration(iteration_index, scores)
            self.update_global_best(best_perm, best_score)

        runtime_seconds = self.finalize()
        best_perm = self.best_perm if self.best_perm is not None else permutations[0]
        return best_perm, float(self.best_score), self.metrics, runtime_seconds

    # --- helpers ---
    def _construct_permutation(self) -> List[int]:
        """Construct a permutation of customer IDs using a roulette-wheel policy."""
        unvisited_indices = list(range(self.num_customers))
        current_index = self.rng.randrange(self.num_customers)
        visit_order = [current_index]
        unvisited_indices.remove(current_index)

        while unvisited_indices:
            weights = []
            for j in unvisited_indices:
                tau = self.pheromone_matrix[current_index,
                                            j] ** self.pheromone_exponent
                eta = self.heuristic_matrix[current_index,
                                            j] ** self.heuristic_exponent
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

        # Map local indices -> global customer IDs
        return [self.customers[i] for i in visit_order]

    def _deposit_pheromone(self, permutation: List[int], score: float) -> None:
        amount = 1.0 / score
        # Map customer IDs to local indices (O(n^2), fine for small n)
        local_indices = [self.customers.index(
            cust_id) for cust_id in permutation]
        for i in range(self.num_customers - 1):
            a = local_indices[i]
            b = local_indices[i + 1]
            self.pheromone_matrix[a, b] += amount
            self.pheromone_matrix[b, a] += amount
