# Algorithm/NSGA2.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class NSGA2(BaseAlgorithm):
    """
    NSGA-II for permutation-based VRP. Multi-objective minimization over:
      0: cost, 1: distance, 2: time
    Reproduction uses binary tournament on (rank, -crowding) per standard NSGA-II.
    Survival uses non-dominated sorting + crowding-distance elitism.
    """

    EPS = 1e-9  # dominance & numerical guard

    def __init__(self, vrp: Dict[str, Any], scorer: str, params: Dict[str, Any]):
        super().__init__(vrp=vrp, scorer=scorer)

        self.population_size = int(params["population_size"])
        self.crossover_rate = float(params["crossover_rate"])
        self.mutation_rate = float(params["mutation_rate"])
        self.crossover_method = str(params["crossover_method"])
        self.mutation_method = str(params["mutation_method"])

        if self.population_size < 2:
            raise ValueError(
                "population_size must be >= 2 for tournament selection and crossover")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")

        valid_crossovers = ["ox", "pmx", "cx", "er"]
        if self.crossover_method not in valid_crossovers:
            raise ValueError(
                f"crossover_method must be one of {valid_crossovers}")

        valid_mutations = ["swap", "inversion", "scramble", "displacement"]
        if self.mutation_method not in valid_mutations:
            raise ValueError(
                f"mutation_method must be one of {valid_mutations}")

        # Archive / state
        self.population: List[List[int]] = []
        self.objectives: List[Tuple[float, float, float]] = []
        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float, float]] = []
        self._crowding_distance: Dict[int, float] = {}

        # Move hot-path imports out of the evaluator
        from Scorer.Distance import score_solution as sdist  # type: ignore
        from Scorer.Cost import score_solution as scost      # type: ignore
        self._sdist = sdist
        self._scost = scost

        log_info(
            "NSGA2 params: population_size=%d, crossover_rate=%.3f, mutation_rate=%.3f, "
            "crossover_method=%s, mutation_method=%s",
            self.population_size, self.crossover_rate, self.mutation_rate,
            self.crossover_method, self.mutation_method,
        )

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")

        log_info("NSGA2 iterations: %d", iters)
        self.start_run()

        # Initialize population
        population = [self._initialize_individual()
                      for _ in range(self.population_size)]

        # Evaluate and prepare ranking/crowding for reproduction
        objectives = [self._evaluate_multi_objective(
            individual) for individual in population]
        fronts = self._fast_non_dominated_sort(objectives)
        self._crowding_distance_assignment(fronts, objectives)

        # Initialize Pareto archive
        self._update_pareto_front(population, objectives)

        for iteration_index in range(1, iters + 1):
            # --- Reproduction (true NSGA-II: tournament on (rank, -crowding)) ---
            offspring = self._create_offspring_nsga2(
                population, objectives, fronts)

            offspring_objectives = [
                self._evaluate_multi_objective(ind) for ind in offspring]

            # --- Survival selection ---
            combined_population = population + offspring
            combined_objectives = objectives + offspring_objectives

            fronts = self._fast_non_dominated_sort(combined_objectives)
            self._crowding_distance_assignment(fronts, combined_objectives)

            population, objectives = self._select_new_population(
                fronts, combined_population, combined_objectives
            )

            # Update Pareto archive with the whole combined set
            self._update_pareto_front(combined_population, combined_objectives)

            # Record metrics on the primary objective
            primary_scores = [self._get_primary_score(
                obj) for obj in objectives]
            self.record_iteration(iteration_index, primary_scores)

            # Update global best from the Pareto archive
            self._update_global_best_from_pareto()

        runtime_seconds = self.finalize()

        # Return best from archive if global best not set for some reason
        if self.best_perm is None and self.pareto_front:
            self._update_global_best_from_pareto()

        return self.best_perm, self.best_score, self.metrics, runtime_seconds

    # ---------- NSGA-II building blocks ----------

    def _select_new_population(
        self,
        fronts: List[List[int]],
        combined_population: List[List[int]],
        combined_objectives: List[Tuple[float, float, float]],
    ) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        """Elitist replacement: fill next gen with whole fronts, then trim the last by crowding distance."""
        new_population: List[List[int]] = []
        new_objectives: List[Tuple[float, float, float]] = []
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
            remaining = self.population_size - len(new_population)
            current_front = list(fronts[front_index])  # copy

            # Sort by (crowding desc, primary score asc) to break ties stably
            current_front.sort(
                key=lambda idx: (
                    -(self._crowding_distance.get(idx, 0.0)),
                    self._get_primary_score(combined_objectives[idx]),
                )
            )

            for idx in current_front[:remaining]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

        return new_population, new_objectives

    def _initialize_individual(self) -> List[int]:
        individual = self.customers[:]
        self.rng.shuffle(individual)
        return individual

    def _evaluate_multi_objective(self, perm: List[int]) -> Tuple[float, float, float]:
        # Infeasible permutations get penalized instead of crashing
        try:
            self.check_constraints(perm)
        except Exception:
            return (float("inf"), float("inf"), float("inf"))

        solution = self._solution_from_perm(perm)

        # Distance & cost via cached callables; time via matrix T
        distance = self._sdist(solution)
        cost = self._scost(
            solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])
        time_val = self._calculate_total_time(solution)

        if not (math.isfinite(distance) and math.isfinite(cost) and math.isfinite(time_val)):
            return (float("inf"),) * 3

        # Zero can be valid; only reject negatives
        if distance < 0.0 or cost < 0.0 or time_val < 0.0:
            return (float("inf"),) * 3

        return float(cost), float(distance), float(time_val)

    def _calculate_total_time(self, solution: List[Dict[str, Any]]) -> float:
        total_time = 0.0
        T = self.vrp["T"]
        for route_data in solution:
            route = route_data["route"]
            for i in range(len(route) - 1):
                total_time += float(T[route[i], route[i + 1]])
        return total_time

    # ---------- Reproduction (NSGA-II style parent selection) ----------

    def _create_offspring_nsga2(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
        fronts: List[List[int]],
    ) -> List[List[int]]:
        """Create offspring using binary tournament on (rank, -crowding), then crossover/mutation."""
        # Precompute index -> rank for fast tournaments
        rank: Dict[int, int] = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        offspring: List[List[int]] = []
        n = len(population)

        # Ensure we have crowding for current generation
        # (caller guarantees _crowding_distance_assignment was called)
        while len(offspring) < self.population_size:
            # Pick two indices at random for parent 1 tournament
            a, b = self.rng.randrange(n), self.rng.randrange(n)
            p1_idx = self._tournament_pick(a, b, rank)

            # Pick two indices at random for parent 2 tournament
            c, d = self.rng.randrange(n), self.rng.randrange(n)
            p2_idx = self._tournament_pick(c, d, rank)

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover
            if self.rng.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                # Copy to avoid aliasing into population
                child = parent1.copy()

            # Mutation
            if self.rng.random() < self.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_pick(self, i: int, j: int, rank: Dict[int, int]) -> int:
        """Binary tournament using (rank, -crowding). Lower rank wins; tie -> higher crowding wins."""
        ri, rj = rank.get(i, math.inf), rank.get(j, math.inf)
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
        # Final random tie-break to avoid determinism
        return i if self.rng.random() < 0.5 else j

    # ---------- Sorting & crowding ----------

    def _fast_non_dominated_sort(self, objectives: List[Tuple[float, float, float]]) -> List[List[int]]:
        n = len(objectives)
        S: List[List[int]] = [[] for _ in range(n)]
        n_count = [0] * n
        rank = [0] * n
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    S[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    n_count[i] += 1
            if n_count[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            Q: List[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts[:-1]  # drop the trailing empty front

    def _crowding_distance_assignment(
        self,
        fronts: List[List[int]],
        objectives: List[Tuple[float, float, float]],
    ) -> None:
        """Calculate crowding distance for each front without mutating front order."""
        self._crowding_distance = {}
        num_objectives = 3

        for front in fronts:
            if not front:
                continue

            # Initialize distances
            for idx in front:
                self._crowding_distance[idx] = 0.0

            # Per-objective contribution
            for m in range(num_objectives):
                # Sort indices by objective m (do not mutate the original front list)
                sorted_idx = sorted(front, key=lambda idx: objectives[idx][m])
                # Boundary points are set to inf
                self._crowding_distance[sorted_idx[0]] = float("inf")
                self._crowding_distance[sorted_idx[-1]] = float("inf")

                if len(sorted_idx) > 2:
                    min_obj = objectives[sorted_idx[0]][m]
                    max_obj = objectives[sorted_idx[-1]][m]
                    denom = max_obj - min_obj
                    if denom > self.EPS:
                        for k in range(1, len(sorted_idx) - 1):
                            left = objectives[sorted_idx[k - 1]][m]
                            right = objectives[sorted_idx[k + 1]][m]
                            self._crowding_distance[sorted_idx[k]
                                                    ] += (right - left) / denom
                    # If denom ~ 0, all points are equal on this objective; distance stays as is.

    def _dominates(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        """Return True if a Pareto-dominates b (all <= and at least one <)."""
        better_in_any = False
        for i in range(3):
            if a[i] > b[i] + self.EPS:  # worse in this objective
                return False
            if a[i] < b[i] - self.EPS:  # strictly better in this objective
                better_in_any = True
        return better_in_any

    def _get_primary_score(self, objectives: Tuple[float, float, float]) -> float:
        if self.scorer == "cost":
            return objectives[0]
        elif self.scorer == "distance":
            return objectives[1]
        elif self.scorer == "time":
            return objectives[2]
        else:
            return objectives[0]

    def _update_pareto_front(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
    ) -> None:
        """Update global archive: non-dominated solutions from (current archive + new population)."""
        candidate_population = self.pareto_front + population
        candidate_objectives = self.pareto_scores + objectives

        new_pareto: List[List[int]] = []
        new_scores: List[Tuple[float, float, float]] = []

        for i, (individual, obj) in enumerate(zip(candidate_population, candidate_objectives)):
            is_dominated = False
            for j, other_obj in enumerate(candidate_objectives):
                if i == j:
                    continue
                if self._dominates(other_obj, obj):
                    is_dominated = True
                    break
            if not is_dominated:
                # prevent duplicates by genotype
                if individual not in new_pareto:
                    new_pareto.append(individual)
                    new_scores.append(obj)

        self.pareto_front = new_pareto
        self.pareto_scores = new_scores

    def _update_global_best_from_pareto(self) -> None:
        if not self.pareto_front:
            return

        if self.scorer == "cost":
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][0])
            best_score = self.pareto_scores[best_idx][0]
        elif self.scorer == "distance":
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][1])
            best_score = self.pareto_scores[best_idx][1]
        elif self.scorer == "time":
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][2])
            best_score = self.pareto_scores[best_idx][2]
        else:
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][0])
            best_score = self.pareto_scores[best_idx][0]

        self.update_global_best(self.pareto_front[best_idx], best_score)

    # ---------- Crossover & mutation ----------

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        if self.crossover_method == "ox":
            return self._order_crossover(parent1, parent2)
        elif self.crossover_method == "pmx":
            return self._partially_mapped_crossover(parent1, parent2)
        elif self.crossover_method == "cx":
            return self._cycle_crossover(parent1, parent2)
        elif self.crossover_method == "er":
            return self._edge_recombination_crossover(parent1, parent2)
        else:
            return self._order_crossover(parent1, parent2)

    def _mutate(self, individual: List[int]) -> List[int]:
        mutated = individual[:]
        if self.mutation_method == "swap":
            self._swap_mutation(mutated)
        elif self.mutation_method == "inversion":
            self._inversion_mutation(mutated)
        elif self.mutation_method == "scramble":
            self._scramble_mutation(mutated)
        elif self.mutation_method == "displacement":
            self._displacement_mutation(mutated)
        else:
            self._swap_mutation(mutated)
        return mutated

    def _order_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        length = len(first_parent)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))
        child: List[Optional[int]] = [None] * length
        child[start_index: end_index +
              1] = first_parent[start_index: end_index + 1]
        remaining_genes = [gene for gene in second_parent if gene not in child]
        fill_pointer = 0
        for pos in range(length):
            if child[pos] is None:
                child[pos] = remaining_genes[fill_pointer]
                fill_pointer += 1
        return [int(x) for x in child]  # type: ignore

    def _partially_mapped_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        """Canonical PMX with bidirectional mapping within cut."""
        length = len(first_parent)
        start, end = sorted(self.rng.sample(range(length), 2))
        child: List[Optional[int]] = [None] * length

        # Copy middle segment from P1
        child[start: end + 1] = first_parent[start: end + 1]

        # Build bidirectional map between the cut segments
        mapping: Dict[int, int] = {}
        for i in range(start, end + 1):
            a, b = first_parent[i], second_parent[i]
            mapping[a] = b
            mapping[b] = a

        # Fill remaining positions with P2 genes mapped through the mapping until unused
        for i in range(length):
            if child[i] is not None:
                continue
            candidate = second_parent[i]
            while candidate in mapping and candidate in child:
                candidate = mapping[candidate]
            while candidate in mapping and candidate in child:
                candidate = mapping[candidate]
            # Resolve chains until value not already placed in child
            while candidate in mapping and candidate in child:
                candidate = mapping[candidate]
            # If candidate still duplicates, chase until free
            while candidate in mapping and candidate in child:
                candidate = mapping[candidate]
            # Final resolve: if candidate is already placed (rare), find mapped chain end
            while candidate in mapping and candidate in child:
                candidate = mapping[candidate]
            child[i] = candidate

        # Any None leftover (edge cases) -> fill from P2 respecting permutation
        if any(g is None for g in child):
            used = set(x for x in child if x is not None)
            tail = [g for g in second_parent if g not in used]
            for i in range(length):
                if child[i] is None:
                    child[i] = tail.pop(0)

        return [int(x) for x in child]  # type: ignore

    def _cycle_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        """
        Correct Cycle Crossover implementation.
        Returns one child by alternating cycles between parents.
        """
        length = len(first_parent)
        child = [None] * length
        visited = [False] * length
        
        # Start with first unvisited position
        start_index = 0
        
        # Alternate between parents for cycles
        use_parent1 = True
        
        while start_index < length:
            if visited[start_index]:
                start_index += 1
                continue
                
            # Find a complete cycle starting from start_index
            cycle_indices = []
            current_index = start_index
            
            while not visited[current_index]:
                visited[current_index] = True
                cycle_indices.append(current_index)
                
                # Get value from first_parent at current position
                current_value = first_parent[current_index]
                # Find this value in second_parent to get next position
                current_index = second_parent.index(current_value)
            
            # Fill this cycle with values from alternating parent
            for idx in cycle_indices:
                if use_parent1:
                    child[idx] = first_parent[idx]
                else:
                    child[idx] = second_parent[idx]
            
            # Switch parent for next cycle
            use_parent1 = not use_parent1
            start_index += 1
        
        # Convert to proper list (should have no None values)
        return [x for x in child if x is not None]
    
    def _edge_recombination_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        length = len(first_parent)
        edge_table: Dict[int, set] = {}

        # Build adjacency sets from both parents
        for i in range(length):
            a = first_parent[i]
            a_neighbors = [
                first_parent[(i - 1) % length], first_parent[(i + 1) % length]]
            b = second_parent[i]
            b_neighbors = [
                second_parent[(i - 1) % length], second_parent[(i + 1) % length]]
            edge_table.setdefault(a, set()).update(a_neighbors)
            edge_table.setdefault(b, set()).update(b_neighbors)

        child: List[int] = []
        current = self.rng.choice(first_parent)

        while len(child) < length:
            child.append(current)
            # Remove current from all neighbor sets
            for neighbors in edge_table.values():
                neighbors.discard(current)

            # Choose next
            neighbors = edge_table.get(current, set())
            if neighbors:
                # Pick neighbor with the fewest remaining edges
                current = min(neighbors, key=lambda x: len(
                    edge_table.get(x, ())))
            else:
                # No neighbors left: pick a random remaining city
                remaining = [
                    node for node in first_parent if node not in child]
                if remaining:
                    current = self.rng.choice(remaining)
                else:
                    break

        # Safety net: ensure full permutation
        if len(child) != length:
            missing = [g for g in first_parent if g not in child]
            child.extend(missing)

        return child

    # ---------- Mutations ----------

    def _swap_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        i, j = self.rng.sample(range(len(permutation)), 2)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    def _inversion_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        permutation[start: end +
                    1] = list(reversed(permutation[start: end + 1]))

    def _scramble_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        segment = permutation[start: end + 1]
        self.rng.shuffle(segment)
        permutation[start: end + 1] = segment

    def _displacement_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        length = len(permutation)
        start, end = sorted(self.rng.sample(range(length), 2))
        segment = permutation[start: end + 1]
        remaining = permutation[:start] + permutation[end + 1:]
        insert_pos = self.rng.randint(0, len(remaining))
        permutation.clear()
        permutation.extend(remaining[:insert_pos])
        permutation.extend(segment)
        permutation.extend(remaining[insert_pos:])

    # ---------- Public ----------

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        return self.pareto_front, self.pareto_scores
