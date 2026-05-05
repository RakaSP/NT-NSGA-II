# Algorithm/PSO.py
from __future__ import annotations

import time
from typing import List, Tuple, Mapping, Optional

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace


class PSO(BaseAlgorithm):
    def __init__(self, vrp, scorer, params, seed):
        super().__init__(vrp=vrp, scorer=scorer, seed=seed)

        self.population_size = int(params.get("population_size", 60))
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")

        self.inertia_prob = float(params.get("inertia_prob", 0.30))
        self.cognitive_prob = float(params.get("cognitive_prob", 0.50))
        self.social_prob = float(params.get("social_prob", 0.70))

        self.local_search_prob = float(params.get("local_search_prob", 0.5))
        self.mutation_prob = float(params.get("mutation_prob", 0.35))

        self.elite_fraction = float(params.get("elite_fraction", 0.20))
        self.max_velocity_len = int(params.get("max_velocity_len", 10))
        self.destroy_ratio = float(params.get("destroy_ratio", 0.15))
        self.lns_prob = float(params.get("lns_prob", 0.4))

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        log_info(
            "PSO params: pop=%d cog=%.2f soc=%.2f ls=%.2f mut=%.2f elite=%.2f vmax=%d destroy=%.2f lns=%.2f",
            self.population_size, self.cognitive_prob, self.social_prob,
            self.local_search_prob, self.mutation_prob,
            self.elite_fraction, self.max_velocity_len,
            self.destroy_ratio, self.lns_prob,
        )

    def _perm_distance(self, a: List[int], b: List[int]) -> List[Tuple[int, int]]:
        pos = {v: i for i, v in enumerate(a)}
        a = a[:]
        swaps: List[Tuple[int, int]] = []
        for i in range(len(a)):
            if a[i] != b[i]:
                j = pos[b[i]]
                swaps.append((i, j))
                pos[a[i]] = j
                a[i], a[j] = a[j], a[i]
                if len(swaps) >= self.max_velocity_len:
                    break
        return swaps

    def _apply_velocity(self, perm: List[int], velocity: List[Tuple[int, int]], prob: float):
        perm = perm[:]
        velocity = sorted(velocity, key=lambda s: abs(s[0] - s[1]), reverse=True)
        for i, j in velocity:
            if self.rng.random() < prob:
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _mutate_perm(self, perm: List[int]) -> List[int]:
        n = len(perm)
        if n < 4:
            return perm
        for _ in range(self.rng.randint(1, 2)):
            i = self.rng.randrange(0, n - 2)
            j = self.rng.randrange(i + 2, n)
            perm = perm[:i] + perm[i:j][::-1] + perm[j:]
            if self.rng.random() < 0.5:
                a = self.rng.randrange(0, n)
                b = self.rng.randrange(0, n)
                if a != b:
                    perm[a], perm[b] = perm[b], perm[a]
        return perm

    def _two_opt(self, perm: List[int], max_checks: int = 200) -> List[int]:
        n = len(perm)
        if n < 4:
            return perm
        D = self._D
        checks = 0

        for _ in range(3):
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    checks += 1
                    if checks >= max_checks:
                        return perm
                        
                    a, b = perm[i], perm[i + 1]
                    c, d = perm[j], perm[j + 1]
                    delta = (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])
                    if delta < -1e-9:
                        perm = perm[: i + 1] + perm[i + 1 : j + 1][::-1] + perm[j + 1 :]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return perm

    def _nearest_neighbor(self, start_idx: int) -> List[int]:
        customers = self.customers[:]
        D = self._D
        start_idx %= len(customers)
        cur = customers[start_idx]

        perm = [cur]
        remaining = set(customers)
        remaining.remove(cur)

        while remaining:
            nxt = min(remaining, key=lambda c: float(D[cur][c]))
            perm.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        return perm

    def _lns_destroy_repair(self, perm: List[int]) -> List[int]:
        n = len(perm)
        if n <= 4:
            return perm

        k = max(2, int(self.destroy_ratio * n))
        indices = sorted(self.rng.sample(range(n), k))
        remaining = [perm[i] for i in range(n) if i not in indices]
        removed = [perm[i] for i in indices]

        D = self._D
        for node in removed:
            best_pos = 0
            best_inc = float("inf")
            m = len(remaining)
            for pos in range(m + 1):
                prev = remaining[pos - 1] if pos > 0 else remaining[-1]
                nxt = remaining[pos] if pos < m else remaining[0]
                inc = D[prev][node] + D[node][nxt] - D[prev][nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            remaining.insert(best_pos, node)
        return remaining

    def _population_diversity(self, X: List[List[int]]) -> float:
        if len(X) < 2:
            return 1.0
        n = len(X[0])
        total = 0.0
        count = 0
        checked = 0
        max_pairs = min(50, len(X) * (len(X) - 1) // 2)
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if checked >= max_pairs:
                    break
                diff = sum(1 for a, b in zip(X[i], X[j]) if a != b)
                total += diff / n
                count += 1
                checked += 1
            if checked >= max_pairs:
                break
        return total / max(1, count)

    def solve(self, iters: int, time_limit_s: Optional[float] = None):
        if iters <= 0:
            raise ValueError("iters must be > 0")

        # Start timer
        start_time = time.time()
        
        self.start_run()
        log_info("PSO iterations: %d", iters)

        # Initialize population
        X: List[List[int]] = []
        P: List[List[int]] = []
        P_fit: List[float] = []

        for i in range(self.population_size):
            perm = self._nearest_neighbor(i)
            perm = self._two_opt(perm, max_checks=400)
            X.append(perm)
            P.append(perm[:])
            P_fit.append(self.evaluate_perm(perm))


        # Initialize global best
        g = min(range(len(P_fit)), key=lambda i: P_fit[i])
        G = P[g][:]
        G_fit = P_fit[g]
        self.update_global_best(G, G_fit)
        runtime_s = 0.0

        # Main loop - check time ONLY at start of each iteration
        for it in range(1, iters + 1):
            # Check time limit
            if time_limit_s and (time.time() - start_time) > time_limit_s:
                runtime_s = time.time() - start_time
                break


            X_next: List[List[int]] = [None] * len(X)
            P_next = [p[:] for p in P]
            P_fit_next = list(P_fit)

            for i in range(len(X)):
                cur = X[i][:]
                new_pos = cur

                # Cognitive component
                v_p = self._perm_distance(new_pos, P[i])
                new_pos = self._apply_velocity(new_pos, v_p, self.cognitive_prob)

                # Social component
                v_g = self._perm_distance(new_pos, G)
                new_pos = self._apply_velocity(new_pos, v_g, self.social_prob)

                # Mutation
                if self.rng.random() < self.mutation_prob:
                    new_pos = self._mutate_perm(new_pos)

                # LNS
                if self.rng.random() < self.lns_prob:
                    new_pos = self._lns_destroy_repair(new_pos)

                # Local search
                if self.rng.random() < self.local_search_prob:
                    new_pos = self._two_opt(new_pos, max_checks=200)

                new_fit = self.evaluate_perm(new_pos)

                X_next[i] = new_pos
                if new_fit < P_fit[i] - 1e-9:
                    P_next[i] = new_pos[:]
                    P_fit_next[i] = new_fit

            X = X_next
            P = P_next
            P_fit = P_fit_next

            # Update global best
            g2 = min(range(len(P_fit)), key=lambda i: P_fit[i])
            if P_fit[g2] < G_fit - 1e-9:
                G = P[g2][:]
                G_fit = P_fit[g2]
                log_info("Iter %d: NEW BEST %.3f", it, G_fit)

            self.update_global_best(G, G_fit)
            self.record_iteration(it, P_fit, X)

            if it % 10 == 0:
                mean_fit = sum(P_fit) / max(1, len(P_fit))
                log_trace("[PSO] iter=%d best=%.6f mean=%.6f", it, G_fit, mean_fit)

        return self.best_perm, float(self.best_score), self.metrics, runtime_s, self.best_info