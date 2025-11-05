# Algorithm/NSGA2.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace


class NSGA2(BaseAlgorithm):
    """
    NSGA-II for permutation-based VRP. Multi-objective minimization over:
      0: cost, 1: distance, 2: time
    Parent selection: binary tournament on (rank, -crowding).
    Replacement: non-dominated sorting + crowding elitism.
    Operators:
      - Crossover: DPX-lite (keep common edges; greedy segment reconnection)
      - Mutation: Ruin-&-Recreate (regret-2) + bounded 2-opt polish
    """

    EPS = 1e-9  # dominance & numerical guard

    def __init__(self, vrp: Dict[str, Any], scorer: str, params: Dict[str, Any]):
        super().__init__(vrp=vrp, scorer=scorer)

        # Core knobs
        self.population_size = int(params.get("population_size", 150))
        self.crossover_rate = float(params.get("crossover_rate", 0.9))
        self.mutation_rate = float(params.get("mutation_rate", 0.35))

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")

        # Local-search / Ruin&Recreate knobs
        self.ls_max_improvements = int(params.get("ls_max_improvements", 10))  # 2-opt accepts per mutation
        self.rr_frac = float(params.get("rr_frac", 0.10))   # fraction of nodes to remove (≈10%)
        self.rr_max = int(params.get("rr_max", 30))         # cap removed nodes

        # Internal state
        self.population: List[List[int]] = []
        self.objectives: List[Tuple[float, float, float]] = []
        self.fronts: List[List[int]] = []  # non-dominated fronts for current generation

        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float, float]] = []
        self._crowding_distance: Dict[int, float] = {}
        self.iteration_index: int = 0

        # Hot-path scorers once
        from Scorer.Distance import score_solution as sdist  # type: ignore
        from Scorer.Cost import score_solution as scost      # type: ignore
        self._sdist = sdist
        self._scost = scost

        log_info(
            "NSGA2 params: population_size=%d, crossover_rate=%.3f, mutation_rate=%.3f, "
            "ls_max_improvements=%d, rr_frac=%.3f, rr_max=%d",
            self.population_size, self.crossover_rate, self.mutation_rate,
            self.ls_max_improvements, self.rr_frac, self.rr_max,
        )

    # ---------- Public entrypoints ----------

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")

        log_info("NSGA2 iterations: %d", iters)
        self.start_run()

        # Initialize population
        self.population = [self._initialize_individual() for _ in range(self.population_size)]

        # Evaluate + initial fronts/crowding
        self.objectives = [self._evaluate_multi_objective(ind) for ind in self.population]
        self.fronts = self._fast_non_dominated_sort(self.objectives)
        self._crowding_distance_assignment(self.fronts, self.objectives)

        # Initialize Pareto archive from front 0
        self._update_pareto_front(
            front0=self.fronts[0],
            population=self.population,
            objectives=self.objectives,
        )

        for gen in range(1, iters + 1):
            self.iteration_index = gen
            self.run_one_generation()

        runtime_seconds = self.finalize()

        # Ensure global best (use archive if needed)
        if self.best_perm is None and self.pareto_front:
            self._update_global_best_from_pareto()

        return self.best_perm, self.best_score, self.metrics, runtime_seconds

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        return self.pareto_front, self.pareto_scores

    # ---------- One generation ----------

    def run_one_generation(self) -> None:
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

        # Archive directly from front 0 (dedup by genotype)
        self._update_pareto_front(
            front0=new_fronts[0],
            population=combined_population,
            objectives=combined_objectives,
        )

        # Metrics (primary score trace for current population)
        primary_scores = [self._get_primary_score(obj) for obj in self.objectives]
        self.record_iteration(self.iteration_index, primary_scores)
        
        self.iteration_index += 1

        log_trace(
            f"[NSGA2] Gen {self.iteration_index}: "
            f"pop={len(self.population)} pareto={len(self.pareto_front)} "
            f"best_{self.scorer}={(min(primary_scores) if primary_scores else float('inf')):.6f} "
            f"cx={self.crossover_rate:.3f} mut={self.mutation_rate:.3f}"
        )

        self._update_global_best_from_pareto()

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

        # Fill whole fronts
        while (
            front_index < len(fronts)
            and len(new_population) + len(fronts[front_index]) <= self.population_size
        ):
            for idx in fronts[front_index]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])
            front_index += 1

        # Trim last needed by crowding (desc) then primary score (asc)
        if len(new_population) < self.population_size and front_index < len(fronts):
            remaining = self.population_size - len(new_population)
            current_front = list(fronts[front_index])

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
        # Constraint check
        try:
            self.check_constraints(perm)
        except Exception:
            return (float("inf"), float("inf"), float("inf"))

        solution = self._solution_from_perm(perm)

        distance = self._sdist(solution)
        cost = self._scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])
        time_val = self._calculate_total_time(solution)

        if not (math.isfinite(distance) and math.isfinite(cost) and math.isfinite(time_val)):
            return (float("inf"),) * 3
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

    # ---------- Reproduction (tournament + DPX-lite + R&R-2opt) ----------

    def _create_offspring_nsga2(self) -> List[List[int]]:
        """Create offspring using binary tournament on (rank, -crowding), then crossover/mutation."""
        # Precompute index -> rank for fast tournaments from current fronts
        rank: Dict[int, int] = {}
        for r, front in enumerate(self.fronts):
            for idx in front:
                rank[idx] = r

        offspring: List[List[int]] = []
        n = len(self.population)

        while len(offspring) < self.population_size:
            # Tournament picks
            a, b = self.rng.randrange(n), self.rng.randrange(n)
            c, d = self.rng.randrange(n), self.rng.randrange(n)
            p1_idx = self._tournament_pick(a, b, rank)
            p2_idx = self._tournament_pick(c, d, rank)

            parent1 = self.population[p1_idx]
            parent2 = self.population[p2_idx]

            # Crossover
            if self.rng.random() < self.crossover_rate:
                child = self._crossover_dpx_lite(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation: Ruin&Recreate + bounded 2-opt (gated by mutation_rate)
            if self.rng.random() < self.mutation_rate:
                child = self._mutate_ruin_recreate_2opt(child)

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
        return i if self.rng.random() < 0.5 else j

    # ---------- Sorting & crowding ----------

    def _fast_non_dominated_sort(self, objectives: List[Tuple[float, float, float]]) -> List[List[int]]:
        n = len(objectives)
        S: List[List[int]] = [[] for _ in range(n)]
        n_count = [0] * n
        rank = [0] * n
        fronts: List[List[int]] = [[]]

        for i in range(n):
            oi = objectives[i]
            for j in range(n):
                if i == j:
                    continue
                oj = objectives[j]
                if self._dominates(oi, oj):
                    S[i].append(j)
                elif self._dominates(oj, oi):
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

        return fronts[:-1]  # drop trailing empty front

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

            for idx in front:
                self._crowding_distance[idx] = 0.0

            for m in range(num_objectives):
                sorted_idx = sorted(front, key=lambda idx: objectives[idx][m])
                # Boundaries
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
                    # else: all equal on this objective -> no contribution

    def _dominates(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        """Return True if a Pareto-dominates b (all <= and at least one <)."""
        better_in_any = False
        for i in range(3):
            if a[i] > b[i] + self.EPS:
                return False
            if a[i] < b[i] - self.EPS:
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

    # ---------- Pareto archive (fast path from front 0) ----------

    def _update_pareto_front(
        self,
        front0: List[int],
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
    ) -> None:
        """Use non-dominated front 0 from the already-sorted pool; deduplicate by genotype."""
        seen = set()
        pf: List[List[int]] = []
        ps: List[Tuple[float, float, float]] = []
        for idx in front0:
            key = tuple(population[idx])
            if key in seen:
                continue
            seen.add(key)
            pf.append(population[idx])
            ps.append(objectives[idx])
        self.pareto_front = pf
        self.pareto_scores = ps

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

    # ---------- Operators: DPX-lite crossover ----------

    def _crossover_dpx_lite(self, p1: List[int], p2: List[int]) -> List[int]:
        """Keep common edges; reconnect components by cheapest endpoints (greedy)."""
        n = len(p1)
        if n <= 2:
            return p1[:]

        D = self.vrp["D"]

        def ek(u: int, v: int) -> Tuple[int, int]:
            return (u, v) if u < v else (v, u)

        def edges(t: List[int]) -> set[Tuple[int, int]]:
            E: set[Tuple[int, int]] = set()
            for i in range(n):
                a, b = t[i], t[(i + 1) % n]
                E.add(ek(a, b))
            return E

        A, B = edges(p1), edges(p2)
        C = A & B  # common edges

        # adjacency from common edges -> segments (paths)
        adj: Dict[int, set] = {}
        for a, b in C:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        used = set()
        segments: List[List[int]] = []

        # grow maximal paths along degree-2 common-edge graph
        for s in p1:  # deterministic seed order
            if s in used:
                continue
            path = [s]
            used.add(s)
            # forward
            cur = s
            while True:
                nxts = [x for x in adj.get(cur, ()) if x not in used]
                if len(nxts) != 1:
                    break
                cur = nxts[0]
                used.add(cur)
                path.append(cur)
            # backward
            cur = s
            while True:
                nxts = [x for x in adj.get(cur, ()) if x not in used]
                if len(nxts) != 1:
                    break
                cur = nxts[0]
                used.add(cur)
                path.insert(0, cur)
            segments.append(path)

        # nodes not covered by common edges -> singleton segments
        covered = {x for seg in segments for x in seg}
        for v in p1:
            if v not in covered:
                segments.append([v])

        # Greedy reconnect by closest endpoints; may reverse segments
        child = segments.pop(self.rng.randrange(len(segments)))
        while segments:
            best = None  # (dist, mode, idx)
            a, b = child[0], child[-1]
            for i, seg in enumerate(segments):
                h, t = seg[0], seg[-1]
                cands = [
                    (float(D[b, h]), 0, i),  # child + seg
                    (float(D[b, t]), 1, i),  # child + rev(seg)
                    (float(D[h, a]), 2, i),  # seg + child
                    (float(D[t, a]), 3, i),  # rev(seg) + child
                ]
                m = min(cands, key=lambda z: z[0])
                if best is None or m[0] < best[0]:
                    best = m
            _, mode, idx = best
            seg = segments.pop(idx)
            if mode == 0:
                child = child + seg
            elif mode == 1:
                child = child + list(reversed(seg))
            elif mode == 2:
                child = seg + child
            else:
                child = list(reversed(seg)) + child

        return child

    # ---------- Operators: Ruin-&-Recreate + bounded 2-opt ----------

    def _mutate_ruin_recreate_2opt(self, tour: List[int]) -> List[int]:
        n = len(tour)
        if n < 8:
            return self._mutate_2opt_only(tour)

        D = self.vrp["D"]

        # --- 1) Ruin: pick seed and remove a related set (distance-based) ---
        k = min(max(3, int(self.rr_frac * n)), self.rr_max)
        seed_pos = self.rng.randrange(n)
        seed = tour[seed_pos]

        # neighborhood map (for relatedness score)
        neigh = {tour[i]: (tour[(i - 1) % n], tour[(i + 1) % n]) for i in range(n)}

        def related(v: int) -> float:
            a, b = neigh[v]
            return min(float(D[v, a]), float(D[v, b]))

        remaining = tour[:]
        removed = [seed]
        remaining.remove(seed)

        cand = [v for v in remaining]
        cand.sort(key=related)
        for v in cand:
            if len(removed) >= k:
                break
            removed.append(v)
            remaining.remove(v)

        # --- 2) Recreate: regret-2 cheapest insertion into remaining cycle ---
        def best_insertion(seq: List[int], node: int) -> Tuple[int, float]:
            best_pos, best_cost = 0, float("inf")
            m = len(seq)
            for i in range(m):
                a, b = seq[i], seq[(i + 1) % m]
                delta = float(D[a, node]) + float(D[node, b]) - float(D[a, b])
                if delta < best_cost:
                    best_cost, best_pos = delta, i + 1
            return best_pos, best_cost

        if len(remaining) < 2:
            remaining = tour[:2]

        R = removed[:]
        seq = remaining[:]
        while R:
            infos = []
            for node in R:
                pos1, c1 = best_insertion(seq, node)
                seq.insert(pos1, node)
                pos2, c2 = best_insertion(seq, node)
                seq.pop(pos1)
                regret = c2 - c1
                infos.append((regret, -c1, node, pos1))
            infos.sort(reverse=True)
            _, _, node, pos = infos[0]
            seq.insert(pos, node)
            R.remove(node)

        child = seq

        # --- 3) Local polish: bounded 2-opt first-improvement ---
        child = self._bounded_2opt(child, self.ls_max_improvements)
        return child

    def _mutate_2opt_only(self, tour: List[int]) -> List[int]:
        return self._bounded_2opt(tour[:], max(2, self.ls_max_improvements // 2))

    def _bounded_2opt(self, tour: List[int], max_improve: int) -> List[int]:
        n = len(tour)
        if n < 4 or max_improve <= 0:
            return tour
        D = self.vrp["D"]

        def d(a: int, b: int) -> float:
            return float(D[a, b])

        imp = 0
        start = self.rng.randrange(n)
        i = 0
        while imp < max_improve and i < n:
            ii = (start + i) % n
            a, b = tour[ii], tour[(ii + 1) % n]
            improved = False
            for off in range(2, n - 1):
                jj = (ii + off) % n
                c, d_ = tour[jj], tour[(jj + 1) % n]
                if b == c or a == d_ or ii == jj:
                    continue
                delta = (d(a, c) + d(b, d_)) - (d(a, b) + d(c, d_))
                if delta < -1e-9:
                    # reverse segment between ii+1 .. jj (cyclic)
                    i1, j1 = (ii + 1) % n, jj
                    if i1 <= j1:
                        tour[i1:j1 + 1] = reversed(tour[i1:j1 + 1])
                    else:
                        seg = list(reversed(tour[i1:] + tour[:j1 + 1]))
                        k = 0
                        for t in range(i1, n):
                            tour[t] = seg[k]; k += 1
                        for t in range(0, j1 + 1):
                            tour[t] = seg[k]; k += 1
                    imp += 1
                    improved = True
                    break
            if not improved:
                i += 1
        return tour
