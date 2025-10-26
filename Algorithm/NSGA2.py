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

        valid_crossovers = ["ox", "ox2", "pos", "uox",
                            "apx", "ppx", "erx", "pmx", "cx", "er"]
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
        """Dispatch to the chosen permutation crossover operator (default: CX)."""
        method = getattr(self, "crossover_method").lower()
        if method == "cx":
            return self._cycle_crossover(parent1, parent2)
        elif method == "pmx":
            return self._pmx_crossover(parent1, parent2)
        elif method == "ox":
            return self._order_crossover(parent1, parent2)      # OX / Order-1
        elif method == "ox2":
            return self._order_crossover2(parent1, parent2)     # OX2
        elif method == "pos":
            return self._position_based_crossover(parent1, parent2)
        elif method == "uox":
            return self._uniform_order_crossover(parent1, parent2)
        elif method == "ppx":
            return self._ppx_crossover(parent1, parent2)
        elif method == "apx":
            return self._alternating_positions_crossover(parent1, parent2)
        elif method == "erx":
            return self._edge_recombination_crossover(parent1, parent2)
        else:
            # fallback: CX
            return self._cycle_crossover(parent1, parent2)

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

    # -----------------------------
    # Additional permutation crossovers
    # -----------------------------
    def _cycle_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        child: List[Optional[int]] = [None] * n  # type: ignore
        pos2 = {v: i for i, v in enumerate(p2)}
        visited = [False] * n
        cycle_id = 0
        for start in range(n):
            if visited[start]:
                continue
            i = start
            cycle = []
            while not visited[i]:
                visited[i] = True
                cycle.append(i)
                i = pos2[p1[i]]
            parent = p1 if (cycle_id % 2 == 0) else p2
            for idx in cycle:
                child[idx] = parent[idx]
            cycle_id += 1
        # type: ignore: we ensured all positions filled
        return [int(x) for x in child]  # type: ignore

    def _pmx_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """PMX: Partially Mapped Crossover."""
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n

        # copy slice from p1
        child[a:b+1] = p1[a:b+1]

        # build bidirectional mapping within the slice
        map_a2b = {p1[i]: p2[i] for i in range(a, b+1)}
        map_b2a = {p2[i]: p1[i] for i in range(a, b+1)}

        used = set(child[a:b+1])

        for i in range(n):
            if a <= i <= b:
                continue
            val = p2[i]
            # follow mapping chain until value not in used
            while val in used and val in map_b2a:
                val = map_b2a[val]
            if val in used:
                # last resort: find first unused from p2 (rare)
                for g in p2:
                    if g not in used:
                        val = g
                        break
            child[i] = val
            used.add(val)

        return child

    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """OX (Order-1): copy a slice from p1; fill remaining by p2 order circularly."""
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n
        child[a:b+1] = p1[a:b+1]
        used = set(child[a:b+1])

        # fill remaining by p2 in circular order starting from b+1
        idx = (b + 1) % n
        for k in range(n):
            gene = p2[(b + 1 + k) % n]
            if gene in used:
                continue
            while child[idx] != -1:
                idx = (idx + 1) % n
            child[idx] = gene
            used.add(gene)
            idx = (idx + 1) % n

        return child

    def _order_crossover2(self, p1: List[int], p2: List[int]) -> List[int]:
        """OX2: select random positions from p1; keep them; fill the rest by p2 order."""
        n = len(p1)
        mask = [self.rng.random() < 0.5 for _ in range(n)]
        child = [-1] * n
        used = set()
        for i, m in enumerate(mask):
            if m:
                child[i] = p1[i]
                used.add(p1[i])
        for gene in p2:
            if gene in used:
                continue
            j = next(k for k in range(n) if child[k] == -1)
            child[j] = gene
            used.add(gene)
        return child

    def _position_based_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """POS: choose random positions from p1; others filled by p2 order."""
        n = len(p1)
        k = max(1, int(0.5 * n))
        pos = set(self.rng.sample(range(n), k))
        child = [-1] * n
        used = set()
        for i in pos:
            child[i] = p1[i]
            used.add(p1[i])
        for gene in p2:
            if gene in used:
                continue
            j = next(k for k in range(n) if child[k] == -1)
            child[j] = gene
            used.add(gene)
        return child

    def _uniform_order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """UOX: uniform mask from p1; others from p2 in order."""
        n = len(p1)
        mask = [self.rng.random() < 0.5 for _ in range(n)]
        child = [-1] * n
        used = set()
        for i, m in enumerate(mask):
            if m:
                child[i] = p1[i]
                used.add(p1[i])
        for gene in p2:
            if gene in used:
                continue
            j = next(k for k in range(n) if child[k] == -1)
            child[j] = gene
            used.add(gene)
        return child

    def _ppx_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """PPX: Precedence-Preservative Crossover."""
        n = len(p1)
        s1, s2 = p1[:], p2[:]
        used = set()
        child: List[int] = []
        while len(child) < n:
            src = s1 if (self.rng.random() < 0.5) else s2
            while src and src[0] in used:
                src.pop(0)
            if not src:
                src = s2 if src is s1 else s1
                while src and src[0] in used:
                    src.pop(0)
                if not src:
                    break
            g = src.pop(0)
            if g in used:
                continue
            child.append(g)
            used.add(g)
            if g in s1:
                s1.remove(g)
            if g in s2:
                s2.remove(g)
        if len(child) < n:
            for g in p2:
                if g not in used:
                    child.append(g)
                    used.add(g)
        return child

    def _alternating_positions_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """APX: alternate picks from p1 and p2, skipping used genes."""
        n = len(p1)
        child: List[int] = []
        used = set()
        i = j = 0
        take_p1 = True
        while len(child) < n and (i < n or j < n):
            if take_p1:
                while i < n and p1[i] in used:
                    i += 1
                if i < n:
                    child.append(p1[i])
                    used.add(p1[i])
                    i += 1
            else:
                while j < n and p2[j] in used:
                    j += 1
                if j < n:
                    child.append(p2[j])
                    used.add(p2[j])
                    j += 1
            take_p1 = not take_p1
        if len(child) < n:
            for g in p2:
                if g not in used:
                    child.append(g)
                    used.add(g)
        return child

    def _edge_recombination_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """ERX: Edge Recombination Crossover."""
        n = len(p1)
        adj: Dict[int, set] = {g: set() for g in p1}

        def add_edges(p: List[int]) -> None:
            for i in range(n):
                a = p[i]
                adj[a].add(p[(i - 1) % n])
                adj[a].add(p[(i + 1) % n])

        add_edges(p1)
        add_edges(p2)

        remaining = set(p1)
        current = self.rng.choice(p1)
        child: List[int] = []

        while remaining:
            child.append(current)
            remaining.remove(current)
            for s in adj.values():
                s.discard(current)
            neigh = [v for v in adj[current] if v in remaining]
            if neigh:
                min_deg = min(len(adj[v]) for v in neigh)
                candidates = [v for v in neigh if len(adj[v]) == min_deg]
                current = self.rng.choice(candidates)
            else:
                if not remaining:
                    break
                current = self.rng.choice(list(remaining))
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
