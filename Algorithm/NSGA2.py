# Algorithm/NSGA2.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace


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

        self.iteration_index: int = 0

        if self.population_size < 2:
            raise ValueError(
                "population_size must be >= 2 for tournament selection and crossover"
            )
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")

        valid_crossovers = [
            "ox", "ox2", "pos", "uox", "apx", "ppx",
            "erx", "pmx", "cx", "er", "eax"  # <- added 'eax'
        ]
        if self.crossover_method not in valid_crossovers:
            raise ValueError(f"crossover_method must be one of {valid_crossovers}")

        valid_mutations = ["swap", "inversion", "scramble", "displacement"]
        if self.mutation_method not in valid_mutations:
            raise ValueError(f"mutation_method must be one of {valid_mutations}")

        # --- Internal state (now stored on self) ---
        self.population: List[List[int]] = []
        self.objectives: List[Tuple[float, float, float]] = []
        self.fronts: List[List[int]] = []  # non-dominated fronts for current generation

        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float, float]] = []
        self._crowding_distance: Dict[int, float] = {}

        # Move hot-path imports out of the evaluator
        from Scorer.Distance import score_solution as sdist  # type: ignore
        from Scorer.Cost import score_solution as scost  # type: ignore

        self._sdist = sdist
        self._scost = scost

        log_info(
            "NSGA2 params: population_size=%d, crossover_rate=%.3f, mutation_rate=%.3f, "
            "crossover_method=%s, mutation_method=%s",
            self.population_size,
            self.crossover_rate,
            self.mutation_rate,
            self.crossover_method,
            self.mutation_method,
        )

    # ---------- Public entrypoints ----------

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")

        log_info("NSGA2 iterations: %d", iters)
        self.start_run()

        # Initialize population
        self.population = [self._initialize_individual() for _ in range(self.population_size)]

        # Evaluate and prepare ranking/crowding for reproduction
        self.objectives = [self._evaluate_multi_objective(ind) for ind in self.population]
        self.fronts = self._fast_non_dominated_sort(self.objectives)
        self._crowding_distance_assignment(self.fronts, self.objectives)

        # Initialize Pareto archive
        self._update_pareto_front(self.population, self.objectives)

        for self.iteration_index in range(1, iters + 1):
            self.run_one_generation()

        runtime_seconds = self.finalize()

        # Return best from archive if global best not set for some reason
        if self.best_perm is None and self.pareto_front:
            self._update_global_best_from_pareto()

        return self.best_perm, self.best_score, self.metrics, runtime_seconds

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        return self.pareto_front, self.pareto_scores

    # ---------- One generation (stateful) ----------

    def run_one_generation(self) -> None:
        """
        One full NSGA-II generation (uses internal state):
        - Reproduction via tournament on (rank, -crowding)
        - Evaluation of offspring
        - Elitist survival (non-dominated sort + crowding)
        - Pareto archive update
        - Metrics & global-best update
        """
        # Reproduction
        offspring = self._create_offspring_nsga2()
        offspring_objectives = [self._evaluate_multi_objective(ind) for ind in offspring]

        # Survival on combined pool
        combined_population = self.population + offspring
        combined_objectives = self.objectives + offspring_objectives

        new_fronts = self._fast_non_dominated_sort(combined_objectives)
        self._crowding_distance_assignment(new_fronts, combined_objectives)

        new_population, new_objectives = self._select_new_population(
            new_fronts, combined_population, combined_objectives
        )

        # Update current state
        self.population = new_population
        self.objectives = new_objectives
        self.fronts = new_fronts

        # Archive & metrics
        self._update_pareto_front(combined_population, combined_objectives)

        primary_scores = [self._get_primary_score(obj) for obj in self.objectives]
        self.record_iteration(self.iteration_index, primary_scores)

        log_trace(
            f"[NSGA2] Generation {self.iteration_index} complete: "
            f"population_size={len(self.population)} pareto_size={len(self.pareto_front)} "
            f"best_{self.scorer}={(min(primary_scores) if primary_scores else float('inf')):.6f} "
            f"cx={self.crossover_rate:.3f} mut={self.mutation_rate:.3f}"
        )

        self._update_global_best_from_pareto()

        self.iteration_index += 1

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
        cost = self._scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])
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

    def _create_offspring_nsga2(self) -> List[List[int]]:
        """Create offspring using binary tournament on (rank, -crowding), then crossover/mutation."""
        # Precompute index -> rank for fast tournaments from current fronts
        rank: Dict[int, int] = {}
        for r, front in enumerate(self.fronts):
            for idx in front:
                rank[idx] = r

        offspring: List[List[int]] = []
        n = len(self.population)

        # Ensure we have crowding for current generation
        # (caller guarantees _crowding_distance_assignment was called)
        while len(offspring) < self.population_size:
            # Tournament for parent 1
            a, b = self.rng.randrange(n), self.rng.randrange(n)
            p1_idx = self._tournament_pick(a, b, rank)

            # Tournament for parent 2
            c, d = self.rng.randrange(n), self.rng.randrange(n)
            p2_idx = self._tournament_pick(c, d, rank)

            parent1 = self.population[p1_idx]
            parent2 = self.population[p2_idx]

            # Crossover
            if self.rng.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            if self.rng.random() < self.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_pick(self, i: int, j: int, rank: Dict[int, int]) -> int:
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
                            self._crowding_distance[sorted_idx[k]] += (right - left) / denom
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
            best_idx = min(range(len(self.pareto_scores)), key=lambda i: self.pareto_scores[i][0])
            best_score = self.pareto_scores[best_idx][0]
        elif self.scorer == "distance":
            best_idx = min(range(len(self.pareto_scores)), key=lambda i: self.pareto_scores[i][1])
            best_score = self.pareto_scores[best_idx][1]
        elif self.scorer == "time":
            best_idx = min(range(len(self.pareto_scores)), key=lambda i: self.pareto_scores[i][2])
            best_score = self.pareto_scores[best_idx][2]
        else:
            best_idx = min(range(len(self.pareto_scores)), key=lambda i: self.pareto_scores[i][0])
            best_score = self.pareto_scores[best_idx][0]

        self.update_global_best(self.pareto_front[best_idx], best_score)

    # ---------- Crossover & mutation ----------

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Dispatch to the chosen permutation crossover operator."""
        method = self.crossover_method.lower()
        if method in {"er", "erx"}:
            return self._edge_recombination_crossover(parent1, parent2)
        if method == "eax":
            return self._edge_assembly_crossover(parent1, parent2)
        # The rest acknowledged but not implemented here
        raise NotImplementedError(f"Crossover '{self.crossover_method}' not implemented yet.")

    def _mutate(self, individual: List[int]) -> List[int]:
        method = self.mutation_method.lower()
        mutated = individual[:]
        if method == "inversion":
            self._inversion_mutation(mutated)
        else:
            # The rest are acknowledged but not implemented in this snippet
            raise NotImplementedError(f"Mutation '{self.mutation_method}' not implemented yet.")
        return mutated

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

    # ---------- EAX helpers & operator ----------

    @staticmethod
    def _eax_ek(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _eax_edges(self, tour: List[int]) -> set[tuple[int, int]]:
        n = len(tour)
        E: set[tuple[int, int]] = set()
        ek = self._eax_ek
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            E.add(ek(a, b))
        return E

    def _eax_adj(self, edges: set[tuple[int, int]]) -> Dict[int, set]:
        adj: Dict[int, set] = {}
        for a, b in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        return adj

    def _eax_find_ab_cycles(
        self, A: set[tuple[int, int]], B: set[tuple[int, int]]
    ) -> List[List[int]]:
        """Find AB-cycles in the symmetric difference by alternating A/B edges."""
        ek = self._eax_ek
        AB = (A | B) - (A & B)
        if not AB:
            return []
        adjA, adjB = self._eax_adj(A), self._eax_adj(B)
        used: set[tuple[int, int]] = set()
        cycles: List[List[int]] = []

        for (s, t) in list(AB):
            e0 = ek(s, t)
            if e0 in used:
                continue
            cyc = [s]
            u, use_A = s, True
            while True:
                neigh = adjA.get(u, ()) if use_A else adjB.get(u, ())
                nxt = None
                for v in neigh:
                    e = ek(u, v)
                    if e in AB and e not in used:
                        nxt = v
                        break
                if nxt is None:
                    break
                used.add(ek(u, nxt))
                u = nxt
                cyc.append(u)
                use_A = not use_A
                if u == s:
                    break
            if len(cyc) > 2 and cyc[0] == cyc[-1]:
                cycles.append(cyc[:-1])
        return cycles

    def _eax_apply_cycles_and_repair(
        self,
        base: List[int],
        A: set[tuple[int, int]],
        B: set[tuple[int, int]],
        chosen: List[List[int]],
    ) -> List[int]:
        """Toggle A-only/B-only edges along cycles, then greedily reconnect subtours."""
        ek = self._eax_ek
        E = set(A)

        # toggle edges via chosen AB-cycles
        for cyc in chosen:
            m = len(cyc)
            for i in range(m):
                u, v = cyc[i], cyc[(i + 1) % m]
                e = ek(u, v)
                inA, inB = (e in A), (e in B)
                if inA and not inB:
                    E.discard(e)
                elif inB and not inA:
                    E.add(e)

        # split into subtours
        adj = self._eax_adj(E)
        unvis = set(base)
        subs: List[List[int]] = []
        while unvis:
            s = next(iter(unvis))
            tour = [s]
            unvis.remove(s)
            prev, cur = None, s
            while True:
                nxts = [x for x in adj.get(cur, ()) if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt in unvis:
                    unvis.remove(nxt)
                tour.append(nxt)
                prev, cur = cur, nxt
                if cur == s:
                    break
            subs.append(tour)

        if len(subs) == 1:
            return subs[0]

        # greedy reconnect subtours by nearest endpoints
        def dist(a: int, b: int) -> float:
            D = self.vrp.get("D")
            return float(D[a, b]) if D is not None else 1.0

        ends = [(t[0], t[-1], t) for t in subs]  # (head, tail, seq)
        while len(ends) > 1:
            best = None  # (d, mode, i, j)
            for i in range(len(ends)):
                for j in range(i + 1, len(ends)):
                    h1, t1, L1 = ends[i]
                    h2, t2, L2 = ends[j]
                    cands = [
                        (dist(t1, h2), 0),   # L1 + L2
                        (dist(t2, h1), 1),   # L2 + L1
                        (dist(h1, h2), 2),   # rev(L1) + L2
                        (dist(t1, t2), 3),   # L1 + rev(L2)
                    ]
                    dd, mode = min(cands, key=lambda z: z[0])
                    if best is None or dd < best[0]:
                        best = (dd, mode, i, j)
            _, mode, i, j = best
            h1, t1, L1 = ends[i]
            h2, t2, L2 = ends[j]
            if mode == 0:
                merged = L1 + L2
            elif mode == 1:
                merged = L2 + L1
            elif mode == 2:
                merged = list(reversed(L1)) + L2
            else:
                merged = L1 + list(reversed(L2))
            ends[j] = (merged[0], merged[-1], merged)
            ends.pop(i)

        return ends[0][2]

    def _edge_assembly_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """EAX (Edge Assembly Crossover) for giant tours (single function that uses local helpers)."""
        if not p1 or not p2 or len(p1) != len(p2):
            return p1[:]

        A = self._eax_edges(p1)
        B = self._eax_edges(p2)
        cycles = self._eax_find_ab_cycles(A, B)
        if not cycles:
            return p1[:]

        best_child = None
        best_obj = (float("inf"),) * 3
        trials = 20  # a few attempts

        for _ in range(trials):
            k = self.rng.randint(1, max(1, min(4, len(cycles))))  # pick a small random subset
            chosen = self.rng.sample(cycles, k)
            child = self._eax_apply_cycles_and_repair(p1, A, B, chosen)
            obj = self._evaluate_multi_objective(child)

            if (
                best_child is None
                or self._dominates(obj, best_obj)
                or (
                    not self._dominates(best_obj, obj)
                    and self._get_primary_score(obj) < self._get_primary_score(best_obj)
                )
            ):
                best_child, best_obj = child, obj

        return best_child if best_child is not None else p1[:]

    # ---------- Mutations ----------

    def _inversion_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        permutation[start: end + 1] = list(reversed(permutation[start: end + 1]))
