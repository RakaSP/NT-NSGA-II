# Genetic Algorithm for the Traveling Salesman Problem using PMX crossover.
#
# Main reference:
# Goldberg, D. E., & Lingle, R. Jr. (1985).
# Alleles, loci, and the traveling salesman problem.
# Proceedings of the First International Conference on Genetic Algorithms
# and Their Applications, pp. 154–159.


from __future__ import annotations

import math
import time

from typing import Dict, List, Optional, Tuple

import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class GA_Goldberg(BaseAlgorithm):
    EPS = 1e-12

    def __init__(self, vrp, params, seed: int = 0):
        super().__init__(vrp=vrp, seed=seed)

        if not isinstance(params, dict):
            raise TypeError("params must be a dict")

        self.population_size = int(params.get("population_size", 100))
        self.crossover_rate = float(params.get("crossover_rate", 0.9))

        # This implementation intentionally uses reproduction + PMX only.
        # mutation_rate is accepted only to keep compatibility with existing configs.
        self.mutation_rate = float(params.get("mutation_rate", 0.0))

        if self.population_size <= 1:
            raise ValueError("population_size must be > 1")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")

        self._fitness_cache: Dict[Tuple[int, ...], float] = {}

        log_info(
            "GA-TSP-PMX params: pop=%d cx=%.3f mut=%.3f",
            self.population_size,
            self.crossover_rate,
            self.mutation_rate,
        )

    # ============================================================
    # Public API
    # ============================================================

    def solve(
        self,
        iters: int,
        time_limit_s: Optional[float] = None,
    ):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        self.start_run()

        start_time = time.time()
        runtime_s = 0.0

        log_info("GA-TSP-PMX iterations: %d", iters)

        population = self._initialize_population()
        fitness_values = self._evaluate_population(population)

        self._update_best_from_population(population, fitness_values)

        for iteration_index in range(1, iters + 1):
            if time_limit_s is not None and time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            offspring: List[List[int]] = []

            while len(offspring) < self.population_size:
                parent1 = self._fitness_proportionate_select(population, fitness_values)
                parent2 = self._fitness_proportionate_select(population, fitness_values)

                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self._pmx_crossover(parent1, parent2)
                else:
                    child1 = parent1[:]
                    child2 = parent2[:]

                # Optional safeguard only. Set mutation_rate=0.0 to use pure
                # reproduction + PMX as in the cited baseline.
                if self.mutation_rate > 0.0:
                    if self.rng.random() < self.mutation_rate:
                        self._swap_mutation(child1)
                    if self.rng.random() < self.mutation_rate:
                        self._swap_mutation(child2)

                offspring.append(child1)

                if len(offspring) < self.population_size:
                    offspring.append(child2)

            population = offspring
            fitness_values = self._evaluate_population(population)

            self._update_best_from_population(population, fitness_values)
            self.record_iteration(iteration_index, fitness_values, population)

        if runtime_s == 0.0:
            runtime_s = time.time() - start_time

        if self.best_perm is not None and self.best_score is not None:
            return (
                self.best_perm,
                float(self.best_score),
                self.metrics,
                runtime_s,
                self.best_info,
            )

        best_idx = int(np.argmin(fitness_values))
        return (
            population[best_idx],
            float(fitness_values[best_idx]),
            self.metrics,
            runtime_s,
            self.best_info,
        )

    # ============================================================
    # Initialization and evaluation
    # ============================================================

    def _initialize_population(self) -> List[List[int]]:
        return [self._initialize_individual() for _ in range(self.population_size)]

    def _initialize_individual(self) -> List[int]:
        tour = self.customers[:]
        self.rng.shuffle(tour)
        return tour

    def _evaluate_population(self, population: List[List[int]]) -> List[float]:
        fitness_values: List[float] = []

        for individual in population:
            key = tuple(individual)
            cached = self._fitness_cache.get(key)

            if cached is None:
                value = float(self.evaluate_perm(individual))
                self._fitness_cache[key] = value
            else:
                value = cached

            fitness_values.append(value)

        return fitness_values

    def _update_best_from_population(
        self,
        population: List[List[int]],
        fitness_values: List[float],
    ) -> None:
        if not fitness_values:
            return

        best_idx = int(np.argmin(fitness_values))
        best_score = float(fitness_values[best_idx])

        if not math.isfinite(best_score):
            return

        if self.best_perm is None or best_score < float(self.best_score) - self.EPS:
            self.update_global_best(population[best_idx], best_score)

    # ============================================================
    # Selection
    # ============================================================

    def _fitness_proportionate_select(
        self,
        population: List[List[int]],
        fitness_values: List[float],
    ) -> List[int]:
        weights: List[float] = []

        for cost in fitness_values:
            if math.isfinite(cost) and cost > 0.0:
                weights.append(1.0 / (cost + self.EPS))
            else:
                weights.append(0.0)

        total_weight = sum(weights)

        if total_weight <= 0.0 or not math.isfinite(total_weight):
            return self.rng.choice(population)[:]

        threshold = self.rng.random() * total_weight
        cumulative = 0.0

        for individual, weight in zip(population, weights):
            cumulative += weight

            if cumulative >= threshold:
                return individual[:]

        return population[-1][:]

    # ============================================================
    # PMX crossover
    # ============================================================

    def _pmx_crossover(
        self,
        parent1: List[int],
        parent2: List[int],
    ) -> Tuple[List[int], List[int]]:
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")

        n = len(parent1)

        if n < 2:
            return parent1[:], parent2[:]

        cut1, cut2 = sorted(self.rng.sample(range(n), 2))

        child1 = self._pmx_make_child(parent1, parent2, cut1, cut2)
        child2 = self._pmx_make_child(parent2, parent1, cut1, cut2)

        return child1, child2

    def _pmx_make_child(
        self,
        segment_parent: List[int],
        fill_parent: List[int],
        cut1: int,
        cut2: int,
    ) -> List[int]:
        n = len(segment_parent)

        child: List[Optional[int]] = [None] * n

        # Copy the mapped segment from the first parent.
        child[cut1 : cut2 + 1] = segment_parent[cut1 : cut2 + 1]

        # Resolve mappings from the second parent.
        for i in range(cut1, cut2 + 1):
            gene = fill_parent[i]

            if gene in child:
                continue

            position = i

            while True:
                mapped_gene = segment_parent[position]
                position = fill_parent.index(mapped_gene)

                if child[position] is None:
                    child[position] = gene
                    break

        # Fill remaining positions from the second parent.
        for i in range(n):
            if child[i] is None:
                child[i] = fill_parent[i]

        return [int(gene) for gene in child if gene is not None]

    # ============================================================
    # Optional mutation safeguard
    # ============================================================

    def _swap_mutation(self, individual: List[int]) -> None:
        if len(individual) < 2:
            return

        i, j = self.rng.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]