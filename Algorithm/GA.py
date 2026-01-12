from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import math
import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class GA(BaseAlgorithm):
    """
    Hyper GA on a "giant tour" permutation (customer IDs). Fitness is evaluate_perm(perm),
    which implicitly decodes/splits into VRP routes.
    """

    def __init__(self, vrp, scorer, params, seed: int = 0):
        super().__init__(vrp=vrp, scorer=scorer, seed=seed)

        if not isinstance(params, dict):
            raise TypeError("params must be a dict")

        self.population_size = int(params.get("population_size"))
        self.crossover_rate = float(params.get("crossover_rate"))
        self.mutation_rate = float(params.get("mutation_rate"))
        self.elite_count = int(params.get("elite_count"))

        self.crossover_method = str(params.get("crossover_method"))
        self.mutation_method = str(params.get("mutation_method"))

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
            raise ValueError(f"crossover_method must be one of {valid_crossovers}")

        valid_mutations = ["swap", "inversion", "scramble", "displacement"]
        if self.mutation_method not in valid_mutations:
            raise ValueError(f"mutation_method must be one of {valid_mutations}")

        # original “free” improvements
        self.greedy_init_frac = float(params.get("greedy_init_frac", 0.20))
        self.dedupe_max_tries = int(params.get("dedupe_max_tries", 8))
        self.post_mut_2opt_steps = int(params.get("post_mut_2opt_steps", 3))

        if not (0.0 <= self.greedy_init_frac <= 1.0):
            raise ValueError("greedy_init_frac must be in [0,1]")
        if self.dedupe_max_tries < 0:
            raise ValueError("dedupe_max_tries must be >= 0")
        if self.post_mut_2opt_steps < 0:
            raise ValueError("post_mut_2opt_steps must be >= 0")

        self._D = self.vrp["D"]

        # Hyper parameters
        self.tournament_k = int(params.get("tournament_k", 4))
        self.elite_local_steps = int(params.get("elite_local_steps", 4))
        self.relink_prob = float(params.get("relink_prob", 0.20))
        self.elite_pool_size = int(params.get("elite_pool_size", max(5, self.elite_count * 2)))

        self.min_mutation_rate = float(params.get("min_mutation_rate", 0.02))
        self.max_mutation_rate = float(params.get("max_mutation_rate", 0.6))
        self.min_crossover_rate = float(params.get("min_crossover_rate", 0.4))
        self.max_crossover_rate = float(params.get("max_crossover_rate", 0.98))

        log_info(
            "GA params: pop=%d cx=%.3f mut=%.3f elite=%d cx_method=%s mut_method=%s | "
            "greedy_init_frac=%.2f dedupe_max_tries=%d post_mut_2opt_steps=%d "
            "Tk=%d elite_local_steps=%d relink_prob=%.2f pool=%d",
            self.population_size, self.crossover_rate, self.mutation_rate, self.elite_count,
            self.crossover_method, self.mutation_method,
            self.greedy_init_frac, self.dedupe_max_tries, self.post_mut_2opt_steps,
            self.tournament_k, self.elite_local_steps, self.relink_prob, self.elite_pool_size
        )

    # ---------- public API ----------
    def solve(self, iters: int, stop_event: Optional[object] = None) -> Tuple[List[int], float, List[dict], float]:
        def _stop() -> bool:
            return stop_event is not None and getattr(stop_event, "is_set", lambda: False)()

        if iters <= 0:
            raise ValueError("iters must be > 0")

        log_info("Iterations: %d", iters)
        self.start_run()

        # --- init population (mix greedy + random + perturbed greedy) ---
        population: List[List[int]] = []
        n_greedy = int(round(self.greedy_init_frac * self.population_size))
        n_greedy = max(0, min(n_greedy, self.population_size))

        for _ in range(n_greedy):
            if _stop():
                break
            base = self._initialize_greedy_individual()
            population.append(base)
            # add a couple of light variants if there is space
            if len(population) + 2 <= self.population_size:
                v1 = base[:]
                v2 = base[:]
                self._scramble_mutation(v1)
                self._inversion_mutation(v2)
                population.append(v1)
                population.append(v2)

        while len(population) < self.population_size:
            if _stop():
                break
            population.append(self._initialize_individual())

        # --- evaluation with per-gen cache (duplicates happen a lot) ---
        def _eval_population(pop: List[List[int]]) -> List[float]:
            cache: Dict[Tuple[int, ...], float] = {}
            fits: List[float] = []
            for ind in pop:
                if _stop():
                    break
                key = tuple(ind)
                v = cache.get(key)
                if v is None:
                    v = float(self.evaluate_perm(ind))
                    cache[key] = v
                fits.append(v)
            return fits

        fitness_values = _eval_population(population)

        if fitness_values:
            bi = int(np.argmin(fitness_values))
            self.update_global_best(population[bi], float(fitness_values[bi]))

        elite_pool: List[Tuple[List[int], float]] = []
        stopped = False

        def _add_to_elite_pool(ind: List[int], fit: float):
            key = tuple(ind)
            for s, f in elite_pool:
                if tuple(s) == key:
                    return
            elite_pool.append((ind[:], fit))
            elite_pool.sort(key=lambda x: x[1])
            if len(elite_pool) > self.elite_pool_size:
                elite_pool.pop()

        TOURNAMENT_K = max(2, self.tournament_k)

        def _tournament_select(pop: List[List[int]], fits: List[float]) -> List[int]:
            k = min(TOURNAMENT_K, len(pop))
            idxs = self.rng.sample(range(len(pop)), k)
            best_i = min(idxs, key=lambda i: fits[i])
            return pop[best_i]

        def _diversity(pop: List[List[int]]) -> float:
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

        def _adapt_rates(iteration_index: int, iters: int, div: float):
            t = iteration_index / max(1, iters)
            if div < 0.25:
                self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.25)
                self.crossover_rate = max(self.min_crossover_rate, self.crossover_rate * 0.85)
            elif div > 0.6:
                self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.85)
                self.crossover_rate = min(self.max_crossover_rate, self.crossover_rate * 1.05)
            if t > 0.6:
                self.crossover_rate = min(self.max_crossover_rate, self.crossover_rate + 0.05)
                self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.9)

        def _path_relink(a: List[int], b: List[int], max_moves: int = 20) -> List[int]:
            if len(a) != len(b):
                return a
            cur = a[:]
            best = cur[:]
            best_fit = float(self.evaluate_perm(best))
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
                f = float(self.evaluate_perm(cur))
                if f < best_fit:
                    best_fit = f
                    best = cur[:]
            return best

        for iteration_index in range(1, iters + 1):
            if _stop():
                stopped = True
                break

            div = _diversity(population)
            _adapt_rates(iteration_index, iters, div)

            order = np.argsort(fitness_values).tolist()
            elite_n = min(self.elite_count, len(population))
            new_population: List[List[int]] = [population[i][:] for i in order[:elite_n]]

            # local search on elites
            if self.elite_local_steps > 0:
                for e_i in range(len(new_population)):
                    new_population[e_i] = self._path_2opt_delta(new_population[e_i], self.elite_local_steps)

            if elite_n == 0 and getattr(self, "best_perm", None) is not None:
                new_population.append(list(self.best_perm))  # type: ignore[arg-type]

            seen = {tuple(ind) for ind in new_population}

            for idx in order[:max(elite_n, 3)]:
                _add_to_elite_pool(population[idx], float(fitness_values[idx]))

            while len(new_population) < self.population_size:
                if _stop():
                    stopped = True
                    break

                p1 = _tournament_select(population, fitness_values)
                p2 = _tournament_select(population, fitness_values)

                if elite_pool and self.rng.random() < self.relink_prob:
                    elite_ind, _ = self.rng.choice(elite_pool)
                    child = _path_relink(p1, elite_ind, max_moves=10)
                else:
                    if self.rng.random() < self.crossover_rate:
                        if self.crossover_method == "ox":
                            child = self._order_crossover(p1, p2)
                        elif self.crossover_method == "pmx":
                            child = self._partially_mapped_crossover(p1, p2)
                        elif self.crossover_method == "cx":
                            child = self._cycle_crossover(p1, p2)
                        else:
                            child = self._edge_recombination_crossover(p1, p2)
                    else:
                        child = p1.copy()

                mutated = False
                if self.rng.random() < self.mutation_rate:
                    mutated = True
                    if self.mutation_method == "swap":
                        self._swap_mutation(child)
                    elif self.mutation_method == "inversion":
                        self._inversion_mutation(child)
                    elif self.mutation_method == "scramble":
                        self._scramble_mutation(child)
                    else:
                        self._displacement_mutation(child)

                if self.post_mut_2opt_steps > 0 and (mutated or self.rng.random() < 0.2):
                    child = self._path_2opt_delta(child, self.post_mut_2opt_steps)

                if self.dedupe_max_tries > 0:
                    tries = 0
                    while tuple(child) in seen and tries < self.dedupe_max_tries:
                        self._swap_mutation(child)
                        tries += 1
                    if tuple(child) in seen:
                        child = self._initialize_individual()

                seen.add(tuple(child))
                new_population.append(child)

            if stopped:
                break

            new_fitness_values = _eval_population(new_population)
            if not new_fitness_values:
                break

            population = new_population
            fitness_values = new_fitness_values

            self.record_iteration(iteration_index, fitness_values, population)

            bi = int(np.argmin(fitness_values))
            self.update_global_best(population[bi], float(fitness_values[bi]))
            _add_to_elite_pool(population[bi], float(fitness_values[bi]))

        runtime_s = self.finalize()

        if getattr(self, "best_perm", None) is not None and getattr(self, "best_score", None) is not None:
            return self.best_perm, float(self.best_score), self.metrics, runtime_s  # type: ignore[attr-defined]

        bi = int(np.argmin(fitness_values)) if fitness_values else 0
        return population[bi], float(fitness_values[bi]), self.metrics, runtime_s

    # ---------- initialization ----------
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
        if len(tour) >= 6 and self.rng.random() < 0.50:
            i, j = sorted(self.rng.sample(range(len(tour)), 2))
            tour[i:j + 1] = reversed(tour[i:j + 1])
        return tour

    # ---------- cheap local polish ----------
    def _path_2opt_delta(self, perm: List[int], steps: int) -> List[int]:
        if steps <= 0:
            return perm
        n = len(perm)
        if n < 4:
            return perm

        D = self._D

        def dist(a: int, b: int) -> float:
            return float(D[a][b])

        out = perm[:]
        for _ in range(steps):
            best_move = None
            trials = min(100, n * 4)
            for _t in range(trials):
                i = self.rng.randrange(0, n - 3)
                j = self.rng.randrange(i + 2, n - 1)
                a, b = out[i], out[i + 1]
                c, d = out[j], out[j + 1]
                delta = (dist(a, c) + dist(b, d)) - (dist(a, b) + dist(c, d))
                if delta < -1e-9:
                    best_move = (i, j)
                    break
            if best_move is None:
                break
            i, j = best_move
            out[i + 1 : j + 1] = reversed(out[i + 1 : j + 1])

        return out

    # ---------- crossover methods (your originals) ----------
    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child: List[Optional[int]] = [None] * n  # type: ignore[list-item]
        child[a : b + 1] = p1[a : b + 1]
        remaining = [g for g in p2 if g not in child]
        k = 0
        for i in range(n):
            if child[i] is None:
                child[i] = remaining[k]
                k += 1
        return [int(x) for x in child]  # type: ignore[arg-type]

    def _partially_mapped_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child: List[Optional[int]] = [None] * n
        child[a : b + 1] = p1[a : b + 1]

        mapping: Dict[int, int] = {}
        for i in range(a, b + 1):
            mapping[p2[i]] = p1[i]

        for i in range(n):
            if child[i] is not None:
                continue
            x = p2[i]
            while x in mapping and x in p2[a : b + 1]:
                x = mapping[x]
            while x in child:
                x = self.rng.choice(self.customers)
            child[i] = x

        return [int(x) for x in child]  # type: ignore[arg-type]

    def _cycle_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        child: List[Optional[int]] = [None] * n
        pos_in_p2 = {val: i for i, val in enumerate(p2)}

        visited = [False] * n
        cycle_id = 0

        for start in range(n):
            if visited[start]:
                continue
            idx = start
            cycle = []
            while not visited[idx]:
                visited[idx] = True
                cycle.append(idx)
                idx = pos_in_p2[p1[idx]]

            src = p1 if (cycle_id % 2 == 0) else p2
            for i in cycle:
                child[i] = src[i]
            cycle_id += 1

        return [int(x) for x in child]  # type: ignore[arg-type]

    def _edge_recombination_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        D = self._D

        edge_table: Dict[int, set] = {}
        for parent in (p1, p2):
            for i in range(n):
                node = parent[i]
                left = parent[(i - 1) % n]
                right = parent[(i + 1) % n]
                edge_table.setdefault(node, set()).update([left, right])

        current = self.rng.choice(p1)
        child: List[int] = []

        while len(child) < n:
            child.append(current)
            for s in edge_table.values():
                s.discard(current)

            neighbors = list(edge_table.get(current, set()))
            if not neighbors:
                remaining = [x for x in p1 if x not in child]
                if not remaining:
                    break
                current = self.rng.choice(remaining)
                continue

            neighbors.sort(key=lambda x: (len(edge_table.get(x, set())), float(D[current][x])))
            current = neighbors[0]

        return child

    # ---------- mutation methods ----------
    def _swap_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        i, j = self.rng.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    def _inversion_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        a, b = sorted(self.rng.sample(range(len(perm)), 2))
        perm[a : b + 1] = reversed(perm[a : b + 1])

    def _scramble_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        a, b = sorted(self.rng.sample(range(len(perm)), 2))
        seg = perm[a : b + 1]
        self.rng.shuffle(seg)
        perm[a : b + 1] = seg

    def _displacement_mutation(self, perm: List[int]) -> None:
        if len(perm) < 2:
            return
        n = len(perm)
        a, b = sorted(self.rng.sample(range(n), 2))
        seg = perm[a : b + 1]
        rest = perm[:a] + perm[b + 1 :]
        ins = self.rng.randint(0, len(rest))
        perm[:] = rest[:ins] + seg + rest[ins:]
