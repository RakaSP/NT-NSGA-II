from __future__ import annotations

import math
import time

from typing import Any, Dict, List, Tuple, Mapping

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Algorithm.BaseBugReplicated import BaseBugReplicated
from Utils.Logger import log_info, log_trace


class NSGA2(BaseBugReplicated):
    EPS = 1e-9

    def __init__(self, vrp: Dict[str, Any], params: Dict[str, Any], seed: int):
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

        self.population: List[List[int]] = []
        self.objectives: List[Tuple[float, float]] = []
        self.fronts: List[List[int]] = []
        self._crowding_distance: Dict[int, float] = {}

        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float]] = []

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        self._T: Mapping[int, Mapping[int, float]] = self.vrp["T"]

        self.iteration_index: int = 0
        self.total_iters: int = 1

        log_info(
            "NSGA2 params: pop=%d cx=%.3f mut=%.3f | objectives=(distance,time) "
            "crossover=DPX-like mutation=RR+2opt operator_size=15%% chromosome",
            self.population_size,
            self.crossover_rate,
            self.mutation_rate,
        )

    # ============================================================
    # Public API
    # ============================================================

    def solve(self, iters: int, time_limit_s: float):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        self.total_iters = iters
        start_time = time.time()

        log_info("NSGA2 iterations: %d", iters)

        self.initialize_run_state()

        for gen in range(1, iters + 1):
            if time.time() - start_time >= time_limit_s:
                break

            self.iteration_index = gen
            self.run_one_generation()

        if self.best_perm is None and self.pareto_front:
            self._update_global_best_from_pareto()

        runtime_s = time.time() - start_time

        return self.best_perm, self.best_score, self.metrics, runtime_s, self.best_info
    
    def run_one_generation(self) -> None:
        offspring = self._create_offspring_nsga2()

        offspring_objectives = [
            self._evaluate_multi_objective(ind)
            for ind in offspring
        ]

        combined_population = self.population + offspring
        combined_objectives = self.objectives + offspring_objectives

        combined_fronts = self._fast_non_dominated_sort(combined_objectives)

        (
            self.population,
            self.objectives,
            self.fronts,
            self._crowding_distance,
        ) = self._select_new_population(
            combined_fronts,
            combined_population,
            combined_objectives,
        )

        if self.fronts and self.fronts[0]:
            self._update_pareto_front(self.fronts[0], self.population, self.objectives)

        primary_scores = [
            self._primary(obj)
            for obj in self.objectives
        ]

        self.record_iteration(self.iteration_index, primary_scores, self.population)
        self._update_global_best_from_pareto()

        log_trace(
            f"[NSGA2] Gen {self.iteration_index}: "
            f"pop={len(self.population)} pareto={len(self.pareto_front)} "
            f"best_distance={(min(primary_scores) if primary_scores else float('inf')):.6f} "
            f"cx={self.crossover_rate:.3f} mut={self.mutation_rate:.3f}"
        )

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float]]]:
        return self.pareto_front, self.pareto_scores

    def initialize_run_state(self) -> None:
        self.start_run()

        self.iteration_index = 0

        self.population = []
        self.objectives = []
        self.fronts = []
        self._crowding_distance = {}

        self.pareto_front = []
        self.pareto_scores = []

        self.population = [
            self._initialize_individual()
            for _ in range(self.population_size)
        ]

        self.objectives = [
            self._evaluate_multi_objective(ind)
            for ind in self.population
        ]

        self.fronts = self._fast_non_dominated_sort(self.objectives)

        # Initial population: all fronts belong to current population,
        # so crowding is needed for parent tournament selection.
        self._crowding_distance_assignment(self.fronts, self.objectives)

        if self.fronts and self.fronts[0]:
            self._update_pareto_front(self.fronts[0], self.population, self.objectives)
            self._update_global_best_from_pareto()


    # ============================================================
    # Initialization
    # ============================================================

    def _initialize_individual(self) -> List[int]:
        perm = self.customers[:]
        self.rng.shuffle(perm)
        return perm

    # ============================================================
    # Objectives
    # ============================================================

    def _evaluate_multi_objective(self, perm: List[int]) -> Tuple[float, float]:
        self.check_constraints(perm)


        solution = self._solution_from_perm(perm)

        from vrp_core.scorer.distance import score_solution as sdist

        distance = float(sdist(solution))
        time_val = float(self._calculate_total_time(solution))

        return distance, time_val

    def _calculate_total_time(self, solution: List[Dict[str, Any]]) -> float:
        total_time = 0.0

        for route_data in solution:
            route = route_data["route"]

            for a, b in zip(route[:-1], route[1:]):
                total_time += float(self._T[a][b])

        return total_time

    def _primary(self, obj: Tuple[float, float]) -> float:
        return obj[0]

    # ============================================================
    # Offspring
    # ============================================================

    def _create_offspring_nsga2(self) -> List[List[int]]:
        rank: Dict[int, int] = {}

        for r, front in enumerate(self.fronts):
            for idx in front:
                rank[idx] = r

        offspring: List[List[int]] = []
        n = len(self.population)

        while len(offspring) < self.population_size:
            a, b = self.rng.randrange(n), self.rng.randrange(n)
            c, d = self.rng.randrange(n), self.rng.randrange(n)

            p1_idx = self._tournament_pick(a, b, rank)
            p2_idx = self._tournament_pick(c, d, rank)

            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            if self.rng.random() < self.crossover_rate:
                child = self._crossover(p1, p2)
            else:
                child = p1.copy()

            if self.rng.random() < self.mutation_rate:
                child = self._mutate_ruin_recreate_2opt(child)

            offspring.append(child)

        return offspring

    def _tournament_pick(self, i: int, j: int, rank: Dict[int, int]) -> int:
        ri = rank.get(i, math.inf)
        rj = rank.get(j, math.inf)

        if ri < rj:
            return i
        if rj < ri:
            return j

        ci = self._crowding_distance.get(i, 0.0)
        cj = self._crowding_distance.get(j, 0.0)

        if ci > cj:
            return i
        if cj > ci:
            return j

        return i if self.rng.random() < 0.5 else j

    # ============================================================
    # NSGA-II Survival
    # ============================================================

    def _select_new_population(
        self,
        fronts: List[List[int]],
        combined_population: List[List[int]],
        combined_objectives: List[Tuple[float, float]],
    ) -> Tuple[
        List[List[int]],
        List[Tuple[float, float]],
        List[List[int]],
        Dict[int, float],
    ]:
        new_population: List[List[int]] = []
        new_objectives: List[Tuple[float, float]] = []
        new_fronts: List[List[int]] = []
        new_crowding_distance: Dict[int, float] = {}

        front_index = 0

        # Take full fronts while they fit.
        while (
            front_index < len(fronts)
            and len(new_population) + len(fronts[front_index]) <= self.population_size
        ):
            current_front = fronts[front_index]

            # Only compute crowding for fronts that enter next generation.
            # These values are needed for next generation's parent tournament.
            current_crowding = self._crowding_distance_for_front(
                current_front,
                combined_objectives,
            )

            remapped_front: List[int] = []

            for old_idx in current_front:
                new_idx = len(new_population)

                new_population.append(combined_population[old_idx])
                new_objectives.append(combined_objectives[old_idx])

                remapped_front.append(new_idx)
                new_crowding_distance[new_idx] = current_crowding.get(old_idx, 0.0)

            new_fronts.append(remapped_front)
            front_index += 1

        # Splitting front: only part of this front enters next generation.
        if len(new_population) < self.population_size and front_index < len(fronts):
            remaining = self.population_size - len(new_population)
            current_front = list(fronts[front_index])

            # Crowding distance is required here to choose which individuals
            # from the splitting front survive.
            current_crowding = self._crowding_distance_for_front(
                current_front,
                combined_objectives,
            )

            current_front.sort(
                key=lambda idx: (
                    -current_crowding.get(idx, 0.0),
                    self._primary(combined_objectives[idx]),
                )
            )

            remapped_front: List[int] = []

            for old_idx in current_front[:remaining]:
                new_idx = len(new_population)

                new_population.append(combined_population[old_idx])
                new_objectives.append(combined_objectives[old_idx])

                remapped_front.append(new_idx)
                new_crowding_distance[new_idx] = current_crowding.get(old_idx, 0.0)

            new_fronts.append(remapped_front)

        return new_population, new_objectives, new_fronts, new_crowding_distance

    def _fast_non_dominated_sort(
        self,
        objectives: List[Tuple[float, float]],
    ) -> List[List[int]]:
        n = len(objectives)

        if n == 0:
            return []

        eps = self.EPS
        dominated_sets: List[List[int]] = [[] for _ in range(n)]
        domination_counts = [0] * n
        fronts: List[List[int]] = [[]]

        # Compare each pair only once: i < j
        for i in range(n - 1):
            a0, a1 = objectives[i]

            for j in range(i + 1, n):
                b0, b1 = objectives[j]

                # i dominates j
                if (
                    a0 <= b0 + eps
                    and a1 <= b1 + eps
                    and (a0 < b0 - eps or a1 < b1 - eps)
                ):
                    dominated_sets[i].append(j)
                    domination_counts[j] += 1

                # j dominates i
                elif (
                    b0 <= a0 + eps
                    and b1 <= a1 + eps
                    and (b0 < a0 - eps or b1 < a1 - eps)
                ):
                    dominated_sets[j].append(i)
                    domination_counts[i] += 1

        for i, count in enumerate(domination_counts):
            if count == 0:
                fronts[0].append(i)

        front_index = 0

        while front_index < len(fronts) and fronts[front_index]:
            next_front: List[int] = []

            for p in fronts[front_index]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1

                    if domination_counts[q] == 0:
                        next_front.append(q)

            front_index += 1

            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _crowding_distance_for_front(
        self,
        front: List[int],
        objectives: List[Tuple[float, float]],
    ) -> Dict[int, float]:
        crowding: Dict[int, float] = {}

        if not front:
            return crowding

        for idx in front:
            crowding[idx] = 0.0

        if len(front) == 1:
            crowding[front[0]] = float("inf")
            return crowding

        num_objectives = len(objectives[0])

        for m in range(num_objectives):
            sorted_idx = sorted(front, key=lambda idx: objectives[idx][m])

            crowding[sorted_idx[0]] = float("inf")
            crowding[sorted_idx[-1]] = float("inf")

            if len(sorted_idx) <= 2:
                continue

            min_obj = objectives[sorted_idx[0]][m]
            max_obj = objectives[sorted_idx[-1]][m]
            denom = max_obj - min_obj

            if denom <= self.EPS:
                continue

            for k in range(1, len(sorted_idx) - 1):
                left = objectives[sorted_idx[k - 1]][m]
                right = objectives[sorted_idx[k + 1]][m]

                crowding[sorted_idx[k]] += (right - left) / denom

        return crowding

    def _crowding_distance_assignment(
        self,
        fronts: List[List[int]],
        objectives: List[Tuple[float, float]],
    ) -> None:
        self._crowding_distance = {}

        for front in fronts:
            self._crowding_distance.update(
                self._crowding_distance_for_front(front, objectives)
            )

    # ============================================================
    # Pareto + Best
    # ============================================================

    def _update_pareto_front(
        self,
        front0: List[int],
        population: List[List[int]],
        objectives: List[Tuple[float, float]],
    ) -> None:
        seen = set()
        pareto_front: List[List[int]] = []
        pareto_scores: List[Tuple[float, float]] = []

        for idx in front0:
            key = tuple(population[idx])

            if key in seen:
                continue

            seen.add(key)
            pareto_front.append(population[idx])
            pareto_scores.append(objectives[idx])

        self.pareto_front = pareto_front
        self.pareto_scores = pareto_scores

    def _update_global_best_from_pareto(self) -> None:
        if not self.pareto_front:
            return

        best_idx = min(
            range(len(self.pareto_scores)),
            key=lambda i: self.pareto_scores[i][0],
        )

        best_score = self.pareto_scores[best_idx][0]
        self.update_global_best(self.pareto_front[best_idx], best_score)

    # ============================================================
    # Operators
    # ============================================================

    def _operator_count(self, n: int) -> int:
        return max(2, int(round(0.15 * n)))

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
                next_nodes = [
                    x for x in adj.get(cur, set())
                    if x not in used
                ]

                if len(next_nodes) != 1:
                    break

                cur = next_nodes[0]
                used.add(cur)
                path.append(cur)

            cur = start_node

            while True:
                next_nodes = [
                    x for x in adj.get(cur, set())
                    if x not in used
                ]

                if len(next_nodes) != 1:
                    break

                cur = next_nodes[0]
                used.add(cur)
                path.insert(0, cur)

            segments.append(path)

        covered = {
            node
            for segment in segments
            for node in segment
        }

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

                candidate = min(candidates, key=lambda item: item[0])

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
            best_cost = float("inf")
            m = len(seq)

            for i in range(m):
                a = seq[i]
                b = seq[(i + 1) % m]

                delta = float(D[a][node]) + float(D[node][b]) - float(D[a][b])

                if delta < best_cost:
                    best_cost = delta
                    best_pos = i + 1

            return best_pos, best_cost

        seq = remaining[:]
        to_insert = removed[:]

        while to_insert:
            insertion_options = []

            for node in to_insert:
                pos1, cost1 = best_insertion(seq, node)

                seq.insert(pos1, node)
                _, cost2 = best_insertion(seq, node)
                seq.pop(pos1)

                regret = cost2 - cost1
                insertion_options.append((regret, -cost1, node, pos1))

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

                old_cost = distance(a, b) + distance(c, d)
                new_cost = distance(a, c) + distance(b, d)
                delta = new_cost - old_cost

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