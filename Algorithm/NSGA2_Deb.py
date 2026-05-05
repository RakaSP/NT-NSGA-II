# Classic NSGA-II implementation.
#
# Reference:
# Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
# A fast and elitist multiobjective genetic algorithm: NSGA-II.
# IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
# DOI: 10.1109/4235.996017

from __future__ import annotations

import math
import time

from typing import Any, Dict, List, Tuple, Mapping

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace


class NSGA2_Deb(BaseAlgorithm):
    EPS = 1e-9

    def __init__(self, vrp: Dict[str, Any], params: Dict[str, Any], seed: int):
        super().__init__(vrp=vrp, seed=seed)

        self.population_size = int(params.get("population_size", 150))
        self.crossover_rate = float(params.get("crossover_rate", 0.9))
        self.mutation_rate = float(params.get("mutation_rate", 1.0 / max(1, len(self.customers))))

        self.eta_c = float(params.get("eta_c", 20.0))
        self.eta_m = float(params.get("eta_m", 20.0))

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.eta_c <= 0.0:
            raise ValueError("eta_c must be > 0")
        if self.eta_m <= 0.0:
            raise ValueError("eta_m must be > 0")

        self.population: List[List[float]] = []
        self.objectives: List[Tuple[float, float]] = []
        self.fronts: List[List[int]] = []
        self._crowding_distance: Dict[int, float] = {}

        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float]] = []

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        self._T: Mapping[int, Mapping[int, float]] = self.vrp["T"]

        self.iteration_index: int = 0

        log_info(
            "ClassicNSGA2 params: pop=%d cx=%.3f mut=%.3f eta_c=%.3f eta_m=%.3f",
            self.population_size,
            self.crossover_rate,
            self.mutation_rate,
            self.eta_c,
            self.eta_m,
        )

    # ============================================================
    # Public API
    # ============================================================

    def initialize_run_state(self) -> None:
        self.start_run()

        self.iteration_index = 0
        self.population = self._initialize_population()
        self.objectives = [self._evaluate_multi_objective(chrom) for chrom in self.population]

        self.fronts = self._fast_non_dominated_sort(self.objectives)
        self._crowding_distance_assignment(self.fronts, self.objectives)

        if self.fronts and self.fronts[0]:
            self._update_pareto_front(self.fronts[0], self.population, self.objectives)
            self._update_global_best_from_pareto()

    def solve(
        self,
        iters: int,
        time_limit_s: float | None = None,
    ):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        start_time = time.time()
        runtime_s = 0.0

        log_info("ClassicNSGA2 iterations: %d", iters)

        self.initialize_run_state()

        for gen in range(1, iters + 1):
            if time_limit_s is not None and time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            self.iteration_index = gen
            self.run_one_generation()

        if runtime_s == 0.0:
            runtime_s = time.time() - start_time

        if self.best_perm is None and self.pareto_front:
            self._update_global_best_from_pareto()

        return self.best_perm, self.best_score, self.metrics, runtime_s, self.best_info

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float]]]:
        return self.pareto_front, self.pareto_scores

    # ============================================================
    # One generation
    # ============================================================

    def run_one_generation(self) -> None:
        offspring = self._create_offspring_nsga2()
        offspring_objectives = [self._evaluate_multi_objective(chrom) for chrom in offspring]

        combined_population = self.population + offspring
        combined_objectives = self.objectives + offspring_objectives

        combined_fronts = self._fast_non_dominated_sort(combined_objectives)
        self._crowding_distance_assignment(combined_fronts, combined_objectives)

        self.population, self.objectives = self._select_new_population(
            combined_fronts,
            combined_population,
            combined_objectives,
        )

        self.fronts = self._fast_non_dominated_sort(self.objectives)
        self._crowding_distance_assignment(self.fronts, self.objectives)

        if self.fronts and self.fronts[0]:
            self._update_pareto_front(self.fronts[0], self.population, self.objectives)

        self._update_global_best_from_pareto()

        primary_scores = [self._primary(obj) for obj in self.objectives]
        decoded_population = [self._decode_chromosome(chrom) for chrom in self.population]
        self.record_iteration(self.iteration_index, primary_scores, decoded_population)

        log_trace(
            f"[ClassicNSGA2] Gen {self.iteration_index}: "
            f"pop={len(self.population)} pareto={len(self.pareto_front)} "
            f"best_primary={(min(primary_scores) if primary_scores else float('inf')):.6f}"
        )

    # ============================================================
    # Initialization
    # ============================================================

    def _initialize_population(self) -> List[List[float]]:
        return [self._initialize_individual() for _ in range(self.population_size)]

    def _initialize_individual(self) -> List[float]:
        return [self.rng.random() for _ in self.customers]

    def _decode_chromosome(self, chromosome: List[float]) -> List[int]:
        indexed_customers = list(zip(chromosome, self.customers))
        indexed_customers.sort(key=lambda item: item[0])
        return [customer for _, customer in indexed_customers]

    # ============================================================
    # Objectives
    # ============================================================

    def _evaluate_multi_objective(self, chromosome: List[float]) -> Tuple[float, float]:
        perm = self._decode_chromosome(chromosome)

        try:
            self.check_constraints(perm)
        except Exception:
            return float("inf"), float("inf")

        solution = self._solution_from_perm(perm)

        from vrp_core.scorer.distance import score_solution as score_distance

        distance = float(score_distance(solution))
        total_time = float(self._calculate_total_time(solution))

        if not math.isfinite(distance) or not math.isfinite(total_time):
            return float("inf"), float("inf")

        if distance < 0.0 or total_time < 0.0:
            return float("inf"), float("inf")

        return distance, total_time

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
    # Offspring creation
    # ============================================================

    def _create_offspring_nsga2(self) -> List[List[float]]:
        rank: Dict[int, int] = {}

        for r, front in enumerate(self.fronts):
            for idx in front:
                rank[idx] = r

        offspring: List[List[float]] = []

        while len(offspring) < self.population_size:
            p1_idx = self._binary_tournament(rank)
            p2_idx = self._binary_tournament(rank)

            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            if self.rng.random() < self.crossover_rate:
                c1, c2 = self._simulated_binary_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            self._polynomial_mutation(c1)
            self._polynomial_mutation(c2)

            offspring.append(c1)

            if len(offspring) < self.population_size:
                offspring.append(c2)

        return offspring

    def _binary_tournament(self, rank: Dict[int, int]) -> int:
        i = self.rng.randrange(len(self.population))
        j = self.rng.randrange(len(self.population))

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
    # NSGA-II survival selection
    # ============================================================

    def _select_new_population(
        self,
        fronts: List[List[int]],
        combined_population: List[List[float]],
        combined_objectives: List[Tuple[float, float]],
    ) -> Tuple[List[List[float]], List[Tuple[float, float]]]:
        new_population: List[List[float]] = []
        new_objectives: List[Tuple[float, float]] = []

        front_index = 0

        while (
            front_index < len(fronts)
            and len(new_population) + len(fronts[front_index]) <= self.population_size
        ):
            for idx in fronts[front_index]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

            front_index += 1

        if len(new_population) < self.population_size and front_index < len(fronts):
            remaining_slots = self.population_size - len(new_population)
            current_front = list(fronts[front_index])

            current_front.sort(
                key=lambda idx: self._crowding_distance.get(idx, 0.0),
                reverse=True,
            )

            for idx in current_front[:remaining_slots]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

        return new_population, new_objectives

    # ============================================================
    # Non-dominated sorting and crowding distance
    # ============================================================

    def _fast_non_dominated_sort(
        self,
        objectives: List[Tuple[float, float]],
    ) -> List[List[int]]:
        n = len(objectives)

        dominated_solutions: List[List[int]] = [[] for _ in range(n)]
        domination_count = [0] * n
        fronts: List[List[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue

                if self._dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        front_index = 0

        while front_index < len(fronts) and fronts[front_index]:
            next_front: List[int] = []

            for p in fronts[front_index]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1

                    if domination_count[q] == 0:
                        next_front.append(q)

            front_index += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _crowding_distance_assignment(
        self,
        fronts: List[List[int]],
        objectives: List[Tuple[float, float]],
    ) -> None:
        self._crowding_distance = {}

        if not objectives:
            return

        num_objectives = len(objectives[0])

        for front in fronts:
            if not front:
                continue

            for idx in front:
                self._crowding_distance[idx] = 0.0

            if len(front) <= 2:
                for idx in front:
                    self._crowding_distance[idx] = float("inf")
                continue

            for m in range(num_objectives):
                sorted_front = sorted(front, key=lambda idx: objectives[idx][m])

                self._crowding_distance[sorted_front[0]] = float("inf")
                self._crowding_distance[sorted_front[-1]] = float("inf")

                min_obj = objectives[sorted_front[0]][m]
                max_obj = objectives[sorted_front[-1]][m]
                denominator = max_obj - min_obj

                if denominator <= self.EPS:
                    continue

                for k in range(1, len(sorted_front) - 1):
                    prev_obj = objectives[sorted_front[k - 1]][m]
                    next_obj = objectives[sorted_front[k + 1]][m]

                    self._crowding_distance[sorted_front[k]] += (
                        next_obj - prev_obj
                    ) / denominator

    def _dominates(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> bool:
        better_in_any = False

        for i in range(len(a)):
            if a[i] > b[i] + self.EPS:
                return False

            if a[i] < b[i] - self.EPS:
                better_in_any = True

        return better_in_any

    # ============================================================
    # Pareto front and final best solution
    # ============================================================

    def _update_pareto_front(
        self,
        front0: List[int],
        population: List[List[float]],
        objectives: List[Tuple[float, float]],
    ) -> None:
        seen = set()

        pareto_front: List[List[int]] = []
        pareto_scores: List[Tuple[float, float]] = []

        for idx in front0:
            perm = self._decode_chromosome(population[idx])
            key = tuple(perm)

            if key in seen:
                continue

            seen.add(key)
            pareto_front.append(perm)
            pareto_scores.append(objectives[idx])

        self.pareto_front = pareto_front
        self.pareto_scores = pareto_scores

    def _update_global_best_from_pareto(self) -> None:
        if not self.pareto_front:
            return

        best_idx = min(
            range(len(self.pareto_scores)),
            key=lambda i: self._primary(self.pareto_scores[i]),
        )

        best_perm = self.pareto_front[best_idx]
        best_score = self._primary(self.pareto_scores[best_idx])

        self.update_global_best(best_perm, best_score)

    # ============================================================
    # Deb et al. real-coded variation operators
    # ============================================================

    def _simulated_binary_crossover(
        self,
        p1: List[float],
        p2: List[float],
    ) -> Tuple[List[float], List[float]]:
        c1 = p1[:]
        c2 = p2[:]

        for i in range(len(p1)):
            parent1 = p1[i]
            parent2 = p2[i]

            if self.rng.random() > 0.5:
                continue

            if abs(parent1 - parent2) <= self.EPS:
                c1[i] = parent1
                c2[i] = parent2
                continue

            y1 = min(parent1, parent2)
            y2 = max(parent1, parent2)

            lower_bound = 0.0
            upper_bound = 1.0

            rand = self.rng.random()

            beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
            alpha = 2.0 - pow(beta, -(self.eta_c + 1.0))

            if rand <= 1.0 / alpha:
                betaq = pow(rand * alpha, 1.0 / (self.eta_c + 1.0))
            else:
                betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.eta_c + 1.0))

            child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
            alpha = 2.0 - pow(beta, -(self.eta_c + 1.0))

            if rand <= 1.0 / alpha:
                betaq = pow(rand * alpha, 1.0 / (self.eta_c + 1.0))
            else:
                betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.eta_c + 1.0))

            child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            child1 = min(max(child1, lower_bound), upper_bound)
            child2 = min(max(child2, lower_bound), upper_bound)

            if self.rng.random() <= 0.5:
                c1[i] = child2
                c2[i] = child1
            else:
                c1[i] = child1
                c2[i] = child2

        return c1, c2

    def _polynomial_mutation(self, chromosome: List[float]) -> None:
        lower_bound = 0.0
        upper_bound = 1.0

        for i in range(len(chromosome)):
            if self.rng.random() > self.mutation_rate:
                continue

            y = chromosome[i]

            if y < lower_bound:
                y = lower_bound
            elif y > upper_bound:
                y = upper_bound

            delta1 = (y - lower_bound) / (upper_bound - lower_bound)
            delta2 = (upper_bound - y) / (upper_bound - lower_bound)

            rand = self.rng.random()
            mut_pow = 1.0 / (self.eta_m + 1.0)

            if rand <= 0.5:
                xy = 1.0 - delta1
                value = 2.0 * rand + (1.0 - 2.0 * rand) * pow(xy, self.eta_m + 1.0)
                deltaq = pow(value, mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                value = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * pow(
                    xy,
                    self.eta_m + 1.0,
                )
                deltaq = 1.0 - pow(value, mut_pow)

            y = y + deltaq * (upper_bound - lower_bound)
            y = min(max(y, lower_bound), upper_bound)

            chromosome[i] = y