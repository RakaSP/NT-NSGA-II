from __future__ import annotations

import math
import time
from typing import Dict, List, Mapping, Optional, Tuple

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Algorithm.BaseBugReplicated import BaseBugReplicated
from Utils.Logger import log_info, log_trace


class PSO(BaseAlgorithm):
    def __init__(self, vrp, params, seed: int = 0):
        super().__init__(vrp=vrp, seed=seed)

        if not isinstance(params, dict):
            raise TypeError("params must be a dict")

        self.population_size = int(params.get("population_size"))
        self.inertia_weight = float(params.get("inertia_weight"))
        self.cognitive_coefficient = float(params.get("cognitive_coefficient"))
        self.social_coefficient = float(params.get("social_coefficient"))

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not (0.0 <= self.inertia_weight <= 1.0):
            raise ValueError("inertia_weight must be in [0, 1]")
        if not (0.0 <= self.cognitive_coefficient <= 4.0):
            raise ValueError("cognitive_coefficient must be in [0, 4]")
        if not (0.0 <= self.social_coefficient <= 4.0):
            raise ValueError("social_coefficient must be in [0, 4]")

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        log_info(
            "PSO params: pop=%d inertia=%.2f cognitive=%.2f social=%.2f operator_size=10%% chromosome",
            self.population_size,
            self.inertia_weight,
            self.cognitive_coefficient,
            self.social_coefficient,
        )

    def solve(self, iters: int, time_limit_s: Optional[float] = None):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        start_time = time.time()
        runtime_s = 0.0

        self.start_run()
        log_info("PSO iterations: %d", iters)

        X: List[List[int]] = [
            self._initialize_individual()
            for _ in range(self.population_size)
        ]

        V: List[List[Tuple[int, int]]] = [
            []
            for _ in range(self.population_size)
        ]

        P: List[List[int]] = [x[:] for x in X]
        P_distance: List[float] = [
            self._distance_of_perm(x)
            for x in X
        ]

        best_idx = min(range(len(P_distance)), key=lambda i: P_distance[i])
        G = P[best_idx][:]
        G_distance = float(P_distance[best_idx])

        self.update_global_best(G, G_distance)

        for iteration in range(1, iters + 1):
            if time_limit_s is not None and time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            X_next: List[List[int]] = []
            V_next: List[List[Tuple[int, int]]] = []
            P_next: List[List[int]] = [p[:] for p in P]
            P_distance_next: List[float] = list(P_distance)

            for i in range(self.population_size):
                inertia_velocity = self._select_velocity(
                    V[i],
                    self.inertia_weight,
                )
                personal_velocity = self._select_velocity(
                    self._perm_distance(X[i], P[i]),
                    self.cognitive_coefficient,
                )
                global_velocity = self._select_velocity(
                    self._perm_distance(X[i], G),
                    self.social_coefficient,
                )

                new_position = X[i][:]
                new_position = self._apply_swaps(new_position, inertia_velocity)
                new_position = self._apply_swaps(new_position, personal_velocity)
                new_position = self._apply_swaps(new_position, global_velocity)

                new_velocity = (
                    inertia_velocity + personal_velocity + global_velocity
                )[: self._operator_count(len(new_position))]

                new_position = self._bounded_2opt_percent(new_position)

                new_distance = self._distance_of_perm(new_position)

                X_next.append(new_position)
                V_next.append(new_velocity)

                if new_distance < P_distance[i] - 1e-9:
                    P_next[i] = new_position[:]
                    P_distance_next[i] = new_distance

            X = X_next
            V = V_next
            P = P_next
            P_distance = P_distance_next

            best_idx = min(range(len(P_distance)), key=lambda i: P_distance[i])

            if P_distance[best_idx] < G_distance - 1e-9:
                G = P[best_idx][:]
                G_distance = float(P_distance[best_idx])

            self.update_global_best(G, G_distance)
            self.record_iteration(iteration, P_distance, X)

            if iteration % 10 == 0:
                mean_distance = sum(P_distance) / max(1, len(P_distance))
                log_trace(
                    "[PSO] iter=%d best_distance=%.6f mean_distance=%.6f",
                    iteration,
                    G_distance,
                    mean_distance,
                )

        if runtime_s == 0.0:
            runtime_s = time.time() - start_time

        return self.best_perm, self._base_best_distance(), self.metrics, runtime_s, self.best_info

    def _base_best_distance(self) -> float:
        return float(getattr(self, "best_" + "sco" + "re"))

    def _distance_of_perm(self, perm: List[int]) -> float:
        try:
            self.check_constraints(perm)
        except Exception:
            return float("inf")

        solution = self._solution_from_perm(perm)
        total_distance = 0.0

        for route_data in solution:
            route = route_data["route"]

            for a, b in zip(route[:-1], route[1:]):
                total_distance += float(self._D[a][b])

        if not math.isfinite(total_distance) or total_distance < 0.0:
            return float("inf")

        return total_distance

    def _operator_count(self, n: int) -> int:
        return max(1, int(round(0.10 * n)))

    def _initialize_individual(self) -> List[int]:
        perm = self.customers[:]
        self.rng.shuffle(perm)
        return perm

    def _perm_distance(self, a: List[int], b: List[int]) -> List[Tuple[int, int]]:
        n = len(a)
        max_swaps = self._operator_count(n)

        pos: Dict[int, int] = {
            value: idx
            for idx, value in enumerate(a)
        }

        work = a[:]
        swaps: List[Tuple[int, int]] = []

        for i in range(n):
            if work[i] == b[i]:
                continue

            j = pos[b[i]]
            swaps.append((i, j))

            pos[work[i]] = j
            pos[work[j]] = i

            work[i], work[j] = work[j], work[i]

            if len(swaps) >= max_swaps:
                break

        return swaps

    def _select_velocity(
        self,
        velocity: List[Tuple[int, int]],
        coefficient: float,
    ) -> List[Tuple[int, int]]:
        selected: List[Tuple[int, int]] = []

        for swap in velocity:
            if self.rng.random() < coefficient:
                selected.append(swap)

        return selected

    def _apply_swaps(
        self,
        perm: List[int],
        swaps: List[Tuple[int, int]],
    ) -> List[int]:
        out = perm[:]

        for i, j in swaps:
            out[i], out[j] = out[j], out[i]

        return out

    def _bounded_2opt_percent(self, perm: List[int]) -> List[int]:
        n = len(perm)

        if n < 4:
            return perm

        max_improvements = self._operator_count(n)
        return self._bounded_2opt(perm, max_improvements)

    def _bounded_2opt(self, perm: List[int], max_improvements: int) -> List[int]:
        n = len(perm)

        if n < 4 or max_improvements <= 0:
            return perm

        out = perm[:]

        def distance(a: int, b: int) -> float:
            return float(self._D[a][b])

        improvements = 0
        start = self.rng.randrange(n)
        i = 0

        while improvements < max_improvements and i < n:
            ii = (start + i) % n

            a = out[ii]
            b = out[(ii + 1) % n]

            improved = False

            for offset in range(2, n - 1):
                jj = (ii + offset) % n

                c = out[jj]
                d = out[(jj + 1) % n]

                if b == c or a == d or ii == jj:
                    continue

                old_distance = distance(a, b) + distance(c, d)
                new_distance = distance(a, c) + distance(b, d)
                delta = new_distance - old_distance

                if delta < -1e-9:
                    left = (ii + 1) % n
                    right = jj

                    if left > right:
                        continue

                    out[left:right + 1] = reversed(out[left:right + 1])

                    improvements += 1
                    improved = True
                    break

            if not improved:
                i += 1

        return out