# Algorithm/GA.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class GA(BaseAlgorithm):
    # Descriptive hyperparameters (overridable via `params`)
    population_size: int
    crossover_rate: float
    mutation_rate: float
    elite_count: int
    # New parameters for operator selection
    crossover_method: str   # ox, pmx, cx, er
    mutation_method: str   # swap, inversion, scramble, displacement

    def __init__(self, vrp, scorer, params):
        """
        Genetic Algorithm for Vehicle Routing.
        Accepts exactly three arguments: (vrp, scorer, params).
        - vrp: problem dictionary required by BaseAlgorithm
        - scorer: "cost" or "distance"
        - params: dict with keys {population_size, crossover_rate, mutation_rate, elite_count, 
                  crossover_method, mutation_method}
                  or a tuple/list in that order.
        """
        super().__init__(vrp=vrp, scorer=scorer)

        if isinstance(params, dict):
            # use exactly what you provided
            population_size = int(params.get(
                "population_size"))
            crossover_rate = float(params.get(
                "crossover_rate"))
            mutation_rate = float(params.get(
                "mutation_rate"))
            elite_count = int(params.get("elite_count"))
            crossover_method = params.get(
                "crossover_method")
            mutation_method = params.get(
                "mutation_method")
        else:
            raise TypeError("params must be a dict")

        # Assign and validate
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        if self.population_size <= 0:
            raise ValueError("population_size must be > 0")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if not (0 <= self.elite_count <= self.population_size):
            raise ValueError("elite_count must be in [0, population_size]")

        valid_crossovers = ["ox", "pmx", "cx", "er"]
        if self.crossover_method not in valid_crossovers:
            raise ValueError(
                f"crossover_method must be one of {valid_crossovers}")

        valid_mutations = ["swap", "inversion", "scramble", "displacement"]
        if self.mutation_method not in valid_mutations:
            raise ValueError(
                f"mutation_method must be one of {valid_mutations}")

        log_info("GA params: population_size=%d, crossover_rate=%.3f, mutation_rate=%.3f, elite_count=%d, crossover_method=%s, mutation_method=%s",
                 self.population_size, self.crossover_rate, self.mutation_rate, self.elite_count, self.crossover_method, self.mutation_method)

    # ---------- public API ----------
    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        """
        Run the genetic algorithm.
        Note: parameter name `iters` retained for compatibility with caller.
        """
        if iters <= 0:
            raise ValueError("iters must be > 0")
        log_info("Iterations: %d", iters)
        self.start_run()

        population = [self._initialize_individual()
                      for _ in range(self.population_size)]
        fitness_values = [self.evaluate_perm(
            individual) for individual in population]

        # small, fixed tournament size (no new param plumbing)
        TOURNAMENT_K = 3

        def _tournament_select() -> List[int]:
            """Select one parent using tournament selection on current fitness."""
            idxs = self.rng.sample(range(self.population_size), TOURNAMENT_K)
            best_i = min(idxs, key=lambda i: fitness_values[i])
            return population[best_i]

        for iteration_index in range(1, iters + 1):
            # Elitism: keep the best individuals
            indices_ordered_by_fitness = np.argsort(fitness_values).tolist()
            new_population = [population[i][:]
                              for i in indices_ordered_by_fitness[: self.elite_count]]

            # Reproduction: crossover then mutation
            while len(new_population) < self.population_size:
                if self.rng.random() < self.crossover_rate:
                    # pick fitter parents via tournament selection
                    first_parent = _tournament_select()
                    second_parent = _tournament_select()

                    # Select crossover method
                    if self.crossover_method == "ox":
                        child = self._order_crossover(
                            first_parent, second_parent)
                    elif self.crossover_method == "pmx":
                        child = self._partially_mapped_crossover(
                            first_parent, second_parent)
                    elif self.crossover_method == "cx":
                        child = self._cycle_crossover(
                            first_parent, second_parent)
                    elif self.crossover_method == "er":
                        child = self._edge_recombination_crossover(
                            first_parent, second_parent)
                    else:
                        child = self._order_crossover(
                            first_parent, second_parent)
                else:
                    # clone a fitter parent instead of a random individual
                    child = _tournament_select().copy()

                if self.rng.random() < self.mutation_rate:
                    # Select mutation method
                    if self.mutation_method == "swap":
                        self._swap_mutation(child)
                    elif self.mutation_method == "inversion":
                        self._inversion_mutation(child)
                    elif self.mutation_method == "scramble":
                        self._scramble_mutation(child)
                    elif self.mutation_method == "displacement":
                        self._displacement_mutation(child)
                    else:
                        self._swap_mutation(child)

                new_population.append(child)

            population = new_population
            fitness_values = [self.evaluate_perm(
                individual) for individual in population]
            self.record_iteration(iteration_index, fitness_values)

            index_of_best = int(np.argmin(fitness_values))
            self.update_global_best(population[index_of_best], float(
                fitness_values[index_of_best]))

        runtime_seconds = self.finalize()
        final_best_index = int(np.argmin(fitness_values))
        return population[final_best_index], float(fitness_values[final_best_index]), self.metrics, runtime_seconds

    # ---------- initialization ----------
    def _initialize_individual(self) -> List[int]:
        """Create a random permutation of customer IDs (1..N-1)."""
        permutation = self.customers[:]
        self.rng.shuffle(permutation)
        return permutation

    # ---------- crossover methods ----------
    def _order_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        """Order Crossover (OX): keep a slice from first_parent, fill remaining by second_parent order."""
        length = len(first_parent)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))
        child: List[Optional[int]] = [None] * length  # type: ignore
        child[start_index:end_index + 1] = first_parent[start_index:end_index + 1]
        remaining_genes = [gene for gene in second_parent if gene not in child]
        fill_pointer = 0
        for position in range(length):
            if child[position] is None:
                child[position] = remaining_genes[fill_pointer]
                fill_pointer += 1
        return [int(x) for x in child]  # type: ignore[arg-type]

    def _partially_mapped_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        """Partially Mapped Crossover (PMX): better for preserving relative order."""
        length = len(first_parent)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))

        # Initialize child with segment from first parent
        child: List[Optional[int]] = [None] * length
        child[start_index:end_index + 1] = first_parent[start_index:end_index + 1]

        # Create mapping from the segment
        mapping = {}
        for i in range(start_index, end_index + 1):
            mapping[first_parent[i]] = second_parent[i]

        # Fill remaining positions using mapping
        for i in range(length):
            if child[i] is None:
                candidate = second_parent[i]
                while candidate in mapping:
                    candidate = mapping[candidate]
                child[i] = candidate

        return [int(x) for x in child]  # type: ignore[arg-type]

    def _cycle_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        """Cycle Crossover (CX): preserves absolute positions better."""
        length = len(first_parent)
        child: List[Optional[int]] = [None] * length
        cycles = []
        visited = [False] * length

        # Find cycles
        for i in range(length):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    value = first_parent[current]
                    current = second_parent.index(value)
                cycles.append(cycle)

        # Alternate cycles between parents
        for i, cycle in enumerate(cycles):
            parent = first_parent if i % 2 == 0 else second_parent
            for idx in cycle:
                child[idx] = parent[idx]

        return [int(x) for x in child]  # type: ignore[arg-type]

    def _edge_recombination_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        """Edge Recombination Crossover (ER): good for preserving adjacency information."""
        length = len(first_parent)

        # Build edge table
        edge_table: Dict[int, set] = {}
        for i in range(length):
            node = first_parent[i]
            left_neighbor = first_parent[(i - 1) % length]
            right_neighbor = first_parent[(i + 1) % length]

            if node not in edge_table:
                edge_table[node] = set()
            edge_table[node].update([left_neighbor, right_neighbor])

            node = second_parent[i]
            left_neighbor = second_parent[(i - 1) % length]
            right_neighbor = second_parent[(i + 1) % length]

            if node not in edge_table:
                edge_table[node] = set()
            edge_table[node].update([left_neighbor, right_neighbor])

        # Build child using edge table
        child = []
        current = self.rng.choice(first_parent)

        while len(child) < length:
            child.append(current)

            # Remove current from all neighbor lists
            for neighbors in edge_table.values():
                if current in neighbors:
                    neighbors.remove(current)

            if not edge_table[current]:
                # If no neighbors left, pick from remaining nodes
                remaining = [
                    node for node in first_parent if node not in child]
                if remaining:
                    current = self.rng.choice(remaining)
                else:
                    break
            else:
                # Pick neighbor with fewest remaining neighbors
                neighbors = list(edge_table[current])
                neighbors.sort(key=lambda x: len(edge_table[x]))
                current = neighbors[0]

        return child

    # ---------- mutation methods ----------
    def _swap_mutation(self, permutation: List[int]) -> None:
        """Mutation operator that swaps two positions in the permutation."""
        if len(permutation) < 2:
            return
        i, j = self.rng.sample(range(len(permutation)), 2)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    def _inversion_mutation(self, permutation: List[int]) -> None:
        """Inversion mutation: reverses a subsequence of the permutation."""
        if len(permutation) < 2:
            return
        start_index, end_index = sorted(
            self.rng.sample(range(len(permutation)), 2))
        permutation[start_index:end_index +
                    1] = reversed(permutation[start_index:end_index + 1])

    def _scramble_mutation(self, permutation: List[int]) -> None:
        """Scramble mutation: randomly shuffles a subsequence."""
        if len(permutation) < 2:
            return
        start_index, end_index = sorted(
            self.rng.sample(range(len(permutation)), 2))
        segment = permutation[start_index:end_index + 1]
        self.rng.shuffle(segment)
        permutation[start_index:end_index + 1] = segment

    def _displacement_mutation(self, permutation: List[int]) -> None:
        """Displacement mutation: cuts a subsequence and inserts it at a different position."""
        if len(permutation) < 2:
            return

        length = len(permutation)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))

        # Extract the segment
        segment = permutation[start_index:end_index + 1]
        remaining = permutation[:start_index] + permutation[end_index + 1:]

        # Choose insertion point
        insert_pos = self.rng.randint(0, len(remaining))

        # Reconstruct permutation
        permutation.clear()
        permutation.extend(remaining[:insert_pos])
        permutation.extend(segment)
        permutation.extend(remaining[insert_pos:])
