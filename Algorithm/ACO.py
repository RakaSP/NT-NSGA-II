# Algorithm/ACO.py
from __future__ import annotations

import math
from typing import List, Tuple, Mapping, Optional
import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class ACO(BaseAlgorithm):
    def __init__(self, vrp, scorer, params, seed: int = 0):
        super().__init__(vrp=vrp, scorer=scorer, seed=seed)

        # ===== Parameters (FROM CONFIG) =====
        self.number_of_ants = int(params.get("number_of_ants"))
        self.pheromone_exponent = float(params.get("pheromone_exponent"))
        self.heuristic_exponent = float(params.get("heuristic_exponent"))
        self.evaporation_rate = float(params.get("evaporation_rate"))

        self.q0 = float(params.get("q0", 0.3))
        self.candidate_k = int(params.get("candidate_k", 20))

        log_info(
            "ACO params: ants=%d alpha=%.2f beta=%.2f rho=%.2f",
            self.number_of_ants,
            self.pheromone_exponent,
            self.heuristic_exponent,
            self.evaporation_rate,
        )

        # ===== Problem data =====
        self.num_customers = len(self.customers)
        self.customer_to_index = {c: i for i, c in enumerate(self.customers)}

        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        C = np.array(
            [[float(D[a][b]) for b in self.customers] for a in self.customers],
            dtype=float,
        )

        eps = 1e-9
        self.heuristic_matrix = 1.0 / (C + eps)

        # ===== Pheromone =====
        self.pheromone_matrix = np.ones(
            (self.num_customers, self.num_customers), dtype=float
        )
        self.tau_max = 1.0
        self.tau_min = 1e-6

        # ===== Candidate list =====
        k = min(self.candidate_k, self.num_customers - 1)
        self.candidate_list = [
            np.argsort(C[i])[1 : k + 1].tolist()
            for i in range(self.num_customers)
        ]

    # ============================================================
    # Solve
    # ============================================================
    def solve(
        self,
        iters: int,
        stop_event: Optional[object] = None,
    ) -> Tuple[List[int], float, List[dict], float]:

        def _stop():
            return stop_event is not None and getattr(
                stop_event, "is_set", lambda: False
            )()

        self.start_run()

        for iteration in range(1, iters + 1):
            if _stop():
                break

            perms: List[List[int]] = []
            scores: List[float] = []

            for _ in range(self.number_of_ants):
                p = self._construct_permutation()
                s = self.evaluate_perm(p)
                perms.append(p)
                scores.append(s)

            # ===== Iteration best =====
            best_i = int(np.argmin(scores))
            iter_best_perm = perms[best_i]
            iter_best_score = scores[best_i]

            # ===== Global best =====
            self.update_global_best(iter_best_perm, iter_best_score)

            # ===== Dynamic MMAS bounds (delayed) =====
            if iteration > 5 and self.best_score and math.isfinite(self.best_score):
                self.tau_max = 1.0 / (self.evaporation_rate * self.best_score)
                self.tau_min = self.tau_max / (2.0 * self.num_customers)

            # ===== Evaporation =====
            self.pheromone_matrix *= (1.0 - self.evaporation_rate)

            # ===== Rank-1 global-best deposit =====
            self._deposit_pheromone(self.best_perm, self.best_score)

            # ===== Occasional iteration-best reinforcement =====
            if iteration % 10 == 0:
                self._deposit_pheromone(iter_best_perm, iter_best_score)

            # ===== Clamp =====
            np.clip(
                self.pheromone_matrix,
                self.tau_min,
                self.tau_max,
                out=self.pheromone_matrix,
            )

            self.record_iteration(iteration, scores, perms)

        runtime_seconds = self.finalize()
        return self.best_perm, float(self.best_score), self.metrics, runtime_seconds

    # ============================================================
    # Construction
    # ============================================================
    def _construct_permutation(self) -> List[int]:
        unvisited = set(range(self.num_customers))
        current = self.rng.randrange(self.num_customers)

        order = [current]
        unvisited.remove(current)

        while unvisited:
            candidates = [j for j in self.candidate_list[current] if j in unvisited]
            if not candidates:
                candidates = list(unvisited)

            tau = self.pheromone_matrix[current, candidates] ** self.pheromone_exponent
            eta = self.heuristic_matrix[current, candidates] ** self.heuristic_exponent
            weights = tau * eta

            if self.rng.random() < self.q0:
                chosen = candidates[int(np.argmax(weights))]
            else:
                total = float(np.sum(weights))
                r = self.rng.random() * total if total > 0 else 0.0
                acc = 0.0
                chosen = candidates[-1]
                for j, w in zip(candidates, weights):
                    acc += float(w)
                    if acc >= r:
                        chosen = j
                        break

            order.append(chosen)
            unvisited.remove(chosen)
            current = chosen

        return [self.customers[i] for i in order]

    # ============================================================
    # Pheromone update
    # ============================================================
    def _deposit_pheromone(self, permutation: List[int], score: float) -> None:
        if not permutation or not math.isfinite(score) or score <= 0.0:
            return

        Q = self.num_customers
        delta = Q / score

        idx = [self.customer_to_index[c] for c in permutation]
        for i in range(len(idx) - 1):
            a, b = idx[i], idx[i + 1]
            self.pheromone_matrix[a, b] += delta
            self.pheromone_matrix[b, a] += delta
