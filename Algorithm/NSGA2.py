from __future__ import annotations

import math
import time

from typing import Any, Dict, List, Tuple, Mapping, Set

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace

class NSGA2(BaseAlgorithm):
    EPS = 1e-9

    def __init__(self, vrp: Dict[str, Any], params: Dict[str, Any], seed: int):
        super().__init__(vrp=vrp, seed=seed)

        # --- core EA params ---
        self.population_size = int(params.get("population_size", 150))
        self.crossover_rate = float(params.get("crossover_rate", 0.9))
        self.mutation_rate = float(params.get("mutation_rate", 0.35))

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")


        # --- local search / ruin-recreate  ---
        self.ls_max_improvements = int(params.get("ls_max_improvements", 10))
        self.rr_frac = float(params.get("rr_frac", 0.10))
        self.rr_max = int(params.get("rr_max", 30))

        self.greedy_init_frac = float(params.get("greedy_init_frac", 0.20))
        self.dedupe_max_tries = int(params.get("dedupe_max_tries", 8))

        self.elite_frac = float(params.get("elite_frac", 0.20))
        self.elite_local_steps = int(params.get("elite_local_steps", 4))
        self.elite_local_prob = float(params.get("elite_local_prob", 0.70))

        self.relink_prob = float(params.get("relink_prob", 0.20))
        self.elite_pool_size = int(params.get("elite_pool_size", max(5, self.population_size // 2)))

        self.force_best_survival = bool(params.get("force_best_survival", True))
        self.light_mutation_mix = float(params.get("light_mutation_mix", 0.40))
        self.light_2opt_steps = int(params.get("light_2opt_steps", max(1, self.ls_max_improvements // 3)))

        # --- state ---
        self.population: List[List[int]] = []
        self.objectives: List[Tuple[float, float, float]] = []
        self.fronts: List[List[int]] = []
        self._crowding_distance: Dict[int, float] = {}

        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float, float]] = []

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]
        self._T: Mapping[int, Mapping[int, float]] = self.vrp["T"]
        self._T_lookup: Dict[int, Mapping[int, float]] = {}

        self.elite_pool: List[Tuple[List[int], Tuple[float, float, float]]] = []

        self.iteration_index: int = 0
        self.total_iters: int = 1

        self.stagnation_count: int = 0
        self.previous_best_primary: float = float("inf")

        log_info(
            "NSGA2 params: pop=%d cx=%.3f mut=%.3f"
            "greedy_init_frac=%.2f elite_frac=%.2f elite_local_steps=%d relink_prob=%.2f "
            "dedupe_max_tries=%d rr_frac=%.2f rr_max=%d",
            self.population_size, self.crossover_rate, self.mutation_rate,
            self.greedy_init_frac, self.elite_frac, self.elite_local_steps, self.relink_prob,
            self.dedupe_max_tries, self.rr_frac, self.rr_max,
        )

    # ============================================================
    # Public API
    # ============================================================

    def initialize_run_state(self) -> None:
        # run-level reset (metrics/best/etc managed by BaseAlgorithm)
        self.start_run()

        self.iteration_index = 0
        self.stagnation_count = 0
        self.previous_best_primary = float("inf")

        self.elite_pool = []
        self.pareto_front = []
        self.pareto_scores = []
        self._crowding_distance = {}
        self.fronts = []
        self.population = []
        self.objectives = []

        # create population
 
        self.population = self._initialize_population()


        # evaluate
        self.objectives = [self._evaluate_multi_objective(ind) for ind in self.population]

        # fronts/crowding
        self.fronts = self._fast_non_dominated_sort(self.objectives)
        self._crowding_distance_assignment(self.fronts, self.objectives)

        # pareto archive
        if self.fronts and self.fronts[0]:
            self._update_pareto_front(self.fronts[0], self.population, self.objectives)

        # elite pool
        self._refresh_elite_pool_from_population()

        # best trackers
        if self.objectives:
            self.previous_best_primary = min(self._primary(o) for o in self.objectives)

    def solve(self, iters: int, time_limit_s: float) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")

        self.total_iters = iters
        start_time = time.time()
        runtime_s = 0.0

        log_info("NSGA2 iterations: %d", iters)

        self.initialize_run_state()

        for gen in range(1, iters + 1):
            if time.time() - start_time >= time_limit_s:
                runtime_s = time.time() - start_time
                break

            self.iteration_index = gen
            self.run_one_generation()

        if self.best_perm is None and self.pareto_front:
            self._update_global_best_from_pareto()

        return self.best_perm, self.best_score, self.metrics, runtime_s, self.best_info

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        return self.pareto_front, self.pareto_scores

    # ============================================================
    # One generation
    # ============================================================

    def run_one_generation(self) -> None:
        # adapt rates from diversity + schedule
        div = self._diversity(self.population)
        self._adapt_rates(self.iteration_index, self.total_iters, div)

        # create offspring
        offspring = self._create_offspring_nsga2()
        offspring_objectives = [self._evaluate_multi_objective(ind) for ind in offspring]

        # combined pool
        combined_population = self.population + offspring
        combined_objectives = self.objectives + offspring_objectives

        # NSGA2 survival (rank + crowding)
        new_fronts = self._fast_non_dominated_sort(combined_objectives)
        self._crowding_distance_assignment(new_fronts, combined_objectives)

        new_population, new_objectives = self._select_new_population(
            new_fronts, combined_population, combined_objectives
        )

        self.population = new_population
        self.objectives = new_objectives

        if self.force_best_survival:
            self._ensure_global_best_survives()
            
        # elite local improvement
        self._elite_local_improvement()

        # recompute fronts/crowding after polishing
        self.fronts = self._fast_non_dominated_sort(self.objectives)
        self._crowding_distance_assignment(self.fronts, self.objectives)

        # update pareto archive
        if self.fronts and self.fronts[0]:
            self._update_pareto_front(self.fronts[0], self.population, self.objectives)

        # refresh elite pool
        self._refresh_elite_pool_from_population()

        # update stagnation
        if self.objectives:
            cur_best = min(self._primary(obj) for obj in self.objectives)
            if cur_best + 1e-9 < self.previous_best_primary:
                self.previous_best_primary = cur_best
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

        # record iteration
        primary_scores = [self._primary(obj) for obj in self.objectives]
        self.record_iteration(self.iteration_index, primary_scores, self.population)

        # update global best from pareto
        self._update_global_best_from_pareto()

        log_trace(
            f"[NSGA2] Gen {self.iteration_index}: "
            f"pop={len(self.population)} pareto={len(self.pareto_front)} "
            f"best_primary={(min(primary_scores) if primary_scores else float('inf')):.6f} "
            f"cx={self.crossover_rate:.3f} mut={self.mutation_rate:.3f} div={div:.3f}"
        )

    # ============================================================
    # Initialization
    # ============================================================

    def _initialize_population(self) -> List[List[int]]:
        pop: List[List[int]] = []

        n_greedy = int(round(self.greedy_init_frac * self.population_size))
        n_greedy = max(0, min(n_greedy, self.population_size))

        for _ in range(n_greedy):
            base = self._initialize_greedy_individual()
            pop.append(base)
            if len(pop) + 2 <= self.population_size:
                v1 = base[:]
                v2 = base[:]
                self._scramble_mutation(v1)
                self._inversion_mutation(v2)
                pop.append(v1)
                pop.append(v2)

        while len(pop) < self.population_size:
            pop.append(self._initialize_individual())

        return pop

    def _initialize_individual(self) -> List[int]:
        perm = self.customers[:]
        self.rng.shuffle(perm)
        return perm

    def _initialize_greedy_individual(self) -> List[int]:
        D = self._D
        remaining = set(self.customers)
        start = self.rng.choice(self.customers)
        tour = [start]
        remaining.remove(start)
        cur = start
        while remaining:
            nxt = min(remaining, key=lambda v: float(D[cur][v]))
            tour.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        # small random kick
        if len(tour) >= 6 and self.rng.random() < 0.50:
            i, j = sorted(self.rng.sample(range(len(tour)), 2))
            tour[i:j + 1] = reversed(tour[i:j + 1])
        return tour

    # ============================================================
    # Objectives
    # ============================================================

    def _evaluate_multi_objective(self, perm: List[int]) -> Tuple[float, float, float]:
        try:
            self.check_constraints(perm)
        except Exception:
            return (float("inf"), float("inf"), float("inf"))

        solution = self._solution_from_perm(perm)

        from vrp_core.scorer.distance import score_solution as sdist
        distance = float(sdist(solution))

        time_val = float(self._calculate_total_time(solution))

        if not (math.isfinite(distance) and math.isfinite(time_val)):
            return (float("inf"),) * 3
        if distance < 0.0 or time_val < 0.0:
            return (float("inf"),) * 3

        return (distance, time_val, time_val)

    def _calculate_total_time(self, solution: List[Dict[str, Any]]) -> float:
        if not self._T_lookup:
            for a in self._T:
                self._T_lookup[a] = self._T[a]

        total_time = 0.0
        t_lookup = self._T_lookup
        for route_data in solution:
            route = route_data["route"]
            for a, b in zip(route[:-1], route[1:]):
                total_time += float(t_lookup[a][b])
        return total_time

    def _primary(self, obj: Tuple[float, float, float]) -> float:
        return obj[0]

    # ============================================================
    # Offsprin
    # ============================================================

    def _create_offspring_nsga2(self) -> List[List[int]]:
        # build rank map from current fronts
        rank: Dict[int, int] = {}
        for r, front in enumerate(self.fronts):
            for idx in front:
                rank[idx] = r

        offspring: List[List[int]] = []
        n = len(self.population)

        # for dedupe
        pop_seen: Set[Tuple[int, ...]] = {tuple(ind) for ind in self.population}

        while len(offspring) < self.population_size:
            # NSGA2 tournament (rank, crowding)
            a, b = self.rng.randrange(n), self.rng.randrange(n)
            c, d = self.rng.randrange(n), self.rng.randrange(n)
            p1_idx = self._tournament_pick(a, b, rank)
            p2_idx = self._tournament_pick(c, d, rank)

            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            # GA-like path relinking sometimes
            if self.elite_pool and self.rng.random() < self.relink_prob:
                elite_ind, _ = self.rng.choice(self.elite_pool)
                child = self._path_relink(p1, elite_ind, max_moves=10)
            else:
                # crossover
                if self.rng.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()

            # mutation: mix heavy RR with light mutation
            if self.rng.random() < self.mutation_rate:
                child = self._mutate_ruin_recreate_2opt(child)
                # if self.rng.random() < self.light_mutation_mix:
                #     if self.rng.random() < 0.5:
                #         self._swap_mutation(child)
                #     else:
                #         self._inversion_mutation(child)
                #     child = self._bounded_2opt(child, self.light_2opt_steps)
                # else:
                #     child = self._mutate_ruin_recreate_2opt(child)

            # deduplication against population + current offspring
            if self.dedupe_max_tries > 0:
                tries = 0
                seen = pop_seen | {tuple(ind) for ind in offspring}
                ct = tuple(child)
                while ct in seen and tries < self.dedupe_max_tries:
                    self._swap_mutation(child)
                    ct = tuple(child)
                    tries += 1
                if ct in seen:
                    self._inversion_mutation(child)

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

    # ============================================================
    # NSGA2 Survival
    # ============================================================

    def _select_new_population(
        self,
        fronts: List[List[int]],
        combined_population: List[List[int]],
        combined_objectives: List[Tuple[float, float, float]],
    ) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
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
            current_front = list(fronts[front_index])

            current_front.sort(
                key=lambda idx: (
                    -(self._crowding_distance.get(idx, 0.0)),
                    self._primary(combined_objectives[idx]),
                )
            )
            for idx in current_front[:remaining]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

        return new_population, new_objectives

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
        while i < len(fronts) and fronts[i]:
            Q: List[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts[:-1]

    def _crowding_distance_assignment(
        self,
        fronts: List[List[int]],
        objectives: List[Tuple[float, float, float]],
    ) -> None:
        self._crowding_distance = {}
        num_objectives = 3

        for front in fronts:
            if not front:
                continue

            for idx in front:
                self._crowding_distance[idx] = 0.0

            for m in range(num_objectives):
                sorted_idx = sorted(front, key=lambda idx: objectives[idx][m])
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

    def _dominates(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        better_in_any = False
        for i in range(3):
            if a[i] > b[i] + self.EPS:
                return False
            if a[i] < b[i] - self.EPS:
                better_in_any = True
        return better_in_any

    # ============================================================
    # Pareto + Best
    # ============================================================

    def _update_pareto_front(
        self,
        front0: List[int],
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
    ) -> None:
        seen = set()
        pf: List[List[int]] = []
        ps: List[Tuple[float, float, float]] = []
        for idx in front0:
            t = tuple(population[idx])
            if t not in seen:
                seen.add(t)
                pf.append(population[idx])
                ps.append(objectives[idx])
        self.pareto_front = pf
        self.pareto_scores = ps

    def _update_global_best_from_pareto(self) -> None:
        if not self.pareto_front:
            return


        best_idx = min(range(len(self.pareto_scores)), key=lambda i: self.pareto_scores[i][0])
        best_score = self.pareto_scores[best_idx][0]

        self.update_global_best(self.pareto_front[best_idx], best_score)

    def _ensure_global_best_survives(self) -> None:
        if getattr(self, "best_perm", None) is None:
            return
        best_t = tuple(self.best_perm)
        if any(tuple(ind) == best_t for ind in self.population):
            return

        worst_i = max(range(len(self.population)), key=lambda i: self._primary(self.objectives[i]))
        self.population[worst_i] = list(self.best_perm)
        self.objectives[worst_i] = self._evaluate_multi_objective(self.population[worst_i])

    # ============================================================
    # GA-like elite pool
    # ============================================================

    def _refresh_elite_pool_from_population(self) -> None:
        if not self.population:
            return
        idxs = sorted(range(len(self.population)), key=lambda i: self._primary(self.objectives[i]))
        topk = max(3, self.population_size // 10)
        for i in idxs[:topk]:
            self._add_to_elite_pool(self.population[i], self.objectives[i])

    def _add_to_elite_pool(self, ind: List[int], obj: Tuple[float, float, float]) -> None:
        key = tuple(ind)
        for s, _ in self.elite_pool:
            if tuple(s) == key:
                return
        self.elite_pool.append((ind[:], obj))
        self.elite_pool.sort(key=lambda x: self._primary(x[1]))
        if len(self.elite_pool) > self.elite_pool_size:
            self.elite_pool.pop()

    def _path_relink(self, a: List[int], b: List[int], max_moves: int = 20) -> List[int]:
        if len(a) != len(b):
            return a
        cur = a[:]
        best = cur[:]
        best_fit = float(self._primary(self._evaluate_multi_objective(best)))

        for _ in range(max_moves):
            idx = None
            for i in range(len(cur)):
                if cur[i] != b[i]:
                    idx = i
                    break
            if idx is None:
                break

            target_val = b[idx]
            j = cur.index(target_val)
            cur[idx], cur[j] = cur[j], cur[idx]

            f = float(self._primary(self._evaluate_multi_objective(cur)))
            if f < best_fit:
                best_fit = f
                best = cur[:]

        return best

    # ============================================================
    # GA-like elite local improvement
    # ============================================================

    def _elite_local_improvement(self) -> None:
        if self.elite_local_steps <= 0 or not self.population:
            return

        num_to_improve = max(1, int(self.population_size * self.elite_frac))
        idxs = sorted(range(len(self.population)), key=lambda i: self._primary(self.objectives[i]))

        for i in idxs[:num_to_improve]:
            if self.rng.random() > self.elite_local_prob:
                continue
            orig = self.population[i]
            improved = self._bounded_2opt(orig[:], self.elite_local_steps)
            o2 = self._evaluate_multi_objective(improved)
            if self._primary(o2) < self._primary(self.objectives[i]):
                self.population[i] = improved
                self.objectives[i] = o2

    # ============================================================
    # Diversity + adaptive rates
    # ============================================================

    def _diversity(self, pop: List[List[int]]) -> float:
        if len(pop) < 2:
            return 1.0
        n = len(pop[0])
        total = 0.0
        cnt = 0
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                diff = sum(1 for a, b in zip(pop[i], pop[j]) if a != b)
                total += diff / n
                cnt += 1
        return total / max(1, cnt)

    def _adapt_rates(self, gen: int, iters: int, div: float) -> None:
        t = gen / max(1, iters)

        if div < 0.25:
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.25)
            self.crossover_rate = max(self.min_crossover_rate, self.crossover_rate * 0.85)
        elif div > 0.6:
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.85)
            self.crossover_rate = min(self.max_crossover_rate, self.crossover_rate * 1.05)

        if t > 0.6:
            self.crossover_rate = min(self.max_crossover_rate, self.crossover_rate + 0.05)
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.9)

        if self.stagnation_count > 10:
            self.mutation_rate = min(0.8, self.mutation_rate * 1.2)
            self.stagnation_count = 0

    # ============================================================
    # Operators
    # ============================================================

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        if n <= 2:
            return p1[:]

        D = self._D

        def ek(u: int, v: int) -> Tuple[int, int]:
            return (u, v) if u < v else (v, u)

        def edges(t: List[int]) -> set[Tuple[int, int]]:
            E: set[Tuple[int, int]] = set()
            for i in range(n):
                a, b = t[i], t[(i + 1) % n]
                E.add(ek(a, b))
            return E

        A, B = edges(p1), edges(p2)
        C = A & B

        adj: Dict[int, set] = {}
        for a, b in C:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        used = set()
        segments: List[List[int]] = []

        for s in p1:
            if s in used:
                continue
            path = [s]
            used.add(s)
            cur = s
            while True:
                nxts = [x for x in adj.get(cur, ()) if x not in used]
                if len(nxts) != 1:
                    break
                cur = nxts[0]
                used.add(cur)
                path.append(cur)
            cur = s
            while True:
                nxts = [x for x in adj.get(cur, ()) if x not in used]
                if len(nxts) != 1:
                    break
                cur = nxts[0]
                used.add(cur)
                path.insert(0, cur)
            segments.append(path)

        covered = {x for seg in segments for x in seg}
        for v in p1:
            if v not in covered:
                segments.append([v])

        child = segments.pop(self.rng.randrange(len(segments)))
        while segments:
            best = None  # (dist, mode, idx)
            a, b = child[0], child[-1]
            for i, seg in enumerate(segments):
                h, t = seg[0], seg[-1]
                cands = [
                    (float(D[b][h]), 0, i),
                    (float(D[b][t]), 1, i),
                    (float(D[h][a]), 2, i),
                    (float(D[t][a]), 3, i),
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

    # --- mutation helpers (light) ---
    def _swap_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        i, j = self.rng.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    def _inversion_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        a, b = sorted(self.rng.sample(range(len(perm)), 2))
        perm[a:b + 1] = reversed(perm[a:b + 1])

    def _scramble_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        a, b = sorted(self.rng.sample(range(len(perm)), 2))
        seg = perm[a:b + 1]
        self.rng.shuffle(seg)
        perm[a:b + 1] = seg

    # --- heavy mutation (ruin-recreate + 2opt) ---
    def _mutate_ruin_recreate_2opt(self, tour: List[int]) -> List[int]:
        n = len(tour)
        if n < 8:
            return self._mutate_2opt_only(tour)

        D = self._D

        k = min(max(3, int(self.rr_frac * n)), self.rr_max)
        seed_pos = self.rng.randrange(n)
        seed = tour[seed_pos]

        neigh = {tour[i]: (tour[(i - 1) % n], tour[(i + 1) % n]) for i in range(n)}

        def related(v: int) -> float:
            a, b = neigh[v]
            return min(float(D[v][a]), float(D[v][b]))

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

        def best_insertion(seq: List[int], node: int) -> Tuple[int, float]:
            best_pos, best_cost = 0, float("inf")
            m = len(seq)
            for i in range(m):
                a, b = seq[i], seq[(i + 1) % m]
                delta = float(D[a][node]) + float(D[node][b]) - float(D[a][b])
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
        child = self._bounded_2opt(child, self.ls_max_improvements)
        return child

    def _mutate_2opt_only(self, tour: List[int]) -> List[int]:
        return self._bounded_2opt(tour[:], max(2, self.ls_max_improvements // 2))

    def _bounded_2opt(self, tour: List[int], max_improve: int) -> List[int]:
        n = len(tour)
        if n < 4 or max_improve <= 0:
            return tour
        D = self._D

        def d(a: int, b: int) -> float:
            return float(D[a][b])

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