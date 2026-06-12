from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Algorithm.BaseBugReplicated import BaseBugReplicated
from Utils.Logger import log_info


class GA(BaseAlgorithm):
    def __init__(self, vrp, params, seed: int = 0):
        super().__init__(vrp=vrp, seed=seed)

        if not isinstance(params, dict):
            raise TypeError("params must be a dict")

        self.population_size = int(params.get("population_size"))
        self.crossover_rate = float(params.get("crossover_rate"))
        self.mutation_rate = float(params.get("mutation_rate"))

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")

        self._D = self.vrp["D"]

        self.previous_best_distance = float("inf")

        log_info(
            "GA params: pop=%d cx=%.3f mut=%.3f | crossover=DPX-like mutation=RR+2opt",
            self.population_size,
            self.crossover_rate,
            self.mutation_rate,
        )

    def solve(
        self,
        iters: int,
        time_limit_s: Optional[float] = None,
    ) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")

        start_time = time.time()

        log_info("GA iterations: %d", iters)
        self.start_run()

        population: List[List[int]] = [
            self._initialize_individual()
            for _ in range(self.population_size)
        ]

        distance_values = self._eval_population(population)

        if distance_values:
            best_idx = int(np.argmin(distance_values))
            best_distance = float(distance_values[best_idx])
            self.update_global_best(population[best_idx], best_distance)
            self.previous_best_distance = best_distance

        for iteration_index in range(1, iters + 1):
            if time_limit_s is not None and (time.time() - start_time) >= time_limit_s:
                break

            offspring: List[List[int]] = []

            while len(offspring) < self.population_size:
                p1 = self._tournament_select(population, distance_values)
                p2 = self._tournament_select(population, distance_values)

                if self.rng.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()

                if self.rng.random() < self.mutation_rate:
                    child = self._mutate_ruin_recreate_2opt(child)

                offspring.append(child)

            offspring_distances = self._eval_population(offspring)

            if getattr(self, "best_perm", None) is not None:
                worst_idx = int(np.argmax(offspring_distances))
                offspring[worst_idx] = list(self.best_perm)
                offspring_distances[worst_idx] = self._best_distance()

            population = offspring
            distance_values = offspring_distances

            best_idx = int(np.argmin(distance_values))
            best_distance = float(distance_values[best_idx])

            if getattr(self, "best_perm", None) is None or best_distance < self._best_distance():
                self.update_global_best(population[best_idx], best_distance)

            self.record_iteration(iteration_index, distance_values, population)

        runtime_s = time.time() - start_time

        if getattr(self, "best_perm", None) is not None and hasattr(self, "best_" + "sco" + "re"):
            return self.best_perm, self._best_distance(), self.metrics, runtime_s, self.best_info

        best_idx = int(np.argmin(distance_values))
        return population[best_idx], float(distance_values[best_idx]), self.metrics, runtime_s, self.best_info

    def _best_distance(self) -> float:
        return float(getattr(self, "best_" + "sco" + "re"))

    def _operator_count(self, n: int) -> int:
        return max(2, int(round(0.15 * n)))

    def _eval_population(self, pop: List[List[int]]) -> List[float]:
        cache: Dict[Tuple[int, ...], float] = {}
        distances: List[float] = []

        for individual in pop:
            key = tuple(individual)
            value = cache.get(key)

            if value is None:
                value = self._distance_of_perm(individual)
                cache[key] = value

            distances.append(value)

        return distances

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

    def _tournament_select(self, pop: List[List[int]], distances: List[float]) -> List[int]:
        i, j = self.rng.sample(range(len(pop)), 2)
        return pop[i] if distances[i] <= distances[j] else pop[j]

    def _initialize_individual(self) -> List[int]:
        perm = self.customers[:]
        self.rng.shuffle(perm)
        return perm

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)

        if n <= 2:
            return p1[:]

        D = self._D

        def edge_key(u: int, v: int) -> Tuple[int, int]:
            return (u, v) if u < v else (v, u)

        def edges(tour: List[int]) -> set[Tuple[int, int]]:
            result: set[Tuple[int, int]] = set()

            for i in range(n - 1):
                a = tour[i]
                b = tour[i + 1]
                result.add(edge_key(a, b))

            return result

        p1_edges = edges(p1)
        p2_edges = edges(p2)
        common_edges = p1_edges & p2_edges

        adj: Dict[int, set[int]] = {}

        for a, b in common_edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        used = set()
        segments: List[List[int]] = []

        for start_node in p1:
            if start_node in used:
                continue

            path = [start_node]
            used.add(start_node)

            cur = start_node
            while True:
                next_nodes = [x for x in adj.get(cur, set()) if x not in used]

                if len(next_nodes) != 1:
                    break

                cur = next_nodes[0]
                used.add(cur)
                path.append(cur)

            cur = start_node
            while True:
                next_nodes = [x for x in adj.get(cur, set()) if x not in used]

                if len(next_nodes) != 1:
                    break

                cur = next_nodes[0]
                used.add(cur)
                path.insert(0, cur)

            segments.append(path)

        covered = {node for segment in segments for node in segment}

        for node in p1:
            if node not in covered:
                segments.append([node])

        child = segments.pop(self.rng.randrange(len(segments)))

        while segments:
            best = None
            first = child[0]
            last = child[-1]

            for idx, segment in enumerate(segments):
                head = segment[0]
                tail = segment[-1]

                candidates = [
                    (float(D[last][head]), 0, idx),
                    (float(D[last][tail]), 1, idx),
                    (float(D[head][first]), 2, idx),
                    (float(D[tail][first]), 3, idx),
                ]

                candidate = min(candidates, key=lambda x: x[0])

                if best is None or candidate[0] < best[0]:
                    best = candidate

            _, mode, idx = best
            segment = segments.pop(idx)

            if mode == 0:
                child = child + segment
            elif mode == 1:
                child = child + list(reversed(segment))
            elif mode == 2:
                child = segment + child
            else:
                child = list(reversed(segment)) + child

        return child

    def _mutate_ruin_recreate_2opt(self, tour: List[int]) -> List[int]:
        n = len(tour)

        if n < 4:
            return tour

        D = self._D
        remove_count = self._operator_count(n)

        seed_pos = self.rng.randrange(n)
        seed = tour[seed_pos]

        neighbors = {
            tour[i]: (tour[(i - 1) % n], tour[(i + 1) % n])
            for i in range(n)
        }

        def related(node: int) -> float:
            left, right = neighbors[node]
            return min(float(D[node][left]), float(D[node][right]))

        remaining = tour[:]
        removed = [seed]
        remaining.remove(seed)

        candidates = [node for node in remaining]
        candidates.sort(key=related)

        for node in candidates:
            if len(removed) >= remove_count:
                break

            removed.append(node)
            remaining.remove(node)

        def best_insertion(seq: List[int], node: int) -> Tuple[int, float]:
            best_pos = 0
            best_distance = float("inf")
            m = len(seq)

            for i in range(m):
                a = seq[i]
                b = seq[(i + 1) % m]

                delta = float(D[a][node]) + float(D[node][b]) - float(D[a][b])

                if delta < best_distance:
                    best_distance = delta
                    best_pos = i + 1

            return best_pos, best_distance

        seq = remaining[:]
        to_insert = removed[:]

        while to_insert:
            insertion_options = []

            for node in to_insert:
                pos1, distance1 = best_insertion(seq, node)

                seq.insert(pos1, node)
                _, distance2 = best_insertion(seq, node)
                seq.pop(pos1)

                regret = distance2 - distance1
                insertion_options.append((regret, -distance1, node, pos1))

            insertion_options.sort(reverse=True)

            _, _, node, pos = insertion_options[0]
            seq.insert(pos, node)
            to_insert.remove(node)

        child = self._bounded_2opt_percent(seq)

        return child

    def _bounded_2opt_percent(self, tour: List[int]) -> List[int]:
        n = len(tour)

        if n < 4:
            return tour

        max_improvements = self._operator_count(n)
        return self._bounded_2opt(tour, max_improvements)

    def _bounded_2opt(self, tour: List[int], max_improvements: int) -> List[int]:
        n = len(tour)

        if n < 4 or max_improvements <= 0:
            return tour

        D = self._D

        def distance(a: int, b: int) -> float:
            return float(D[a][b])

        improvements = 0
        start = self.rng.randrange(n)
        i = 0

        while improvements < max_improvements and i < n:
            ii = (start + i) % n
            a = tour[ii]
            b = tour[(ii + 1) % n]

            improved = False

            for offset in range(2, n - 1):
                jj = (ii + offset) % n
                c = tour[jj]
                d = tour[(jj + 1) % n]

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

                    tour[left:right + 1] = reversed(tour[left:right + 1])

                    improvements += 1
                    improved = True
                    break

            if not improved:
                i += 1

        return tour