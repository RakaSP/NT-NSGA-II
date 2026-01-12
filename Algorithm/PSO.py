from __future__ import annotations
from typing import List, Tuple, Mapping, Optional

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info, log_trace


class PSO(BaseAlgorithm):
    """
    HyperPSO: enhanced discrete / permutation-based PSO for VRP/TSP.
    - NO numpy
    - Velocity = list of swaps
    - Uses self.rng ONLY
    """

    def __init__(self, vrp, scorer, params, seed):
        super().__init__(vrp=vrp, scorer=scorer, seed=seed)

        self.population_size = int(params.get("population_size", 60))
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")

        # Base parameters (will be adapted in time)
        self.inertia_prob = float(params.get("inertia_prob", 0.30))
        self.cognitive_prob = float(params.get("cognitive_prob", 0.50))
        self.social_prob = float(params.get("social_prob", 0.70))

        self.local_search_prob = float(params.get("local_search_prob", 0.5))
        self.mutation_prob = float(params.get("mutation_prob", 0.35))

        # Hyper stuff
        self.elite_fraction = float(params.get("elite_fraction", 0.20))
        self.max_velocity_len = int(params.get("max_velocity_len", 10))
        self.destroy_ratio = float(params.get("destroy_ratio", 0.15))
        self.lns_prob = float(params.get("lns_prob", 0.4))

        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        log_info(
            "HyperPSO params: pop=%d inertia=%.2f cognitive=%.2f social=%.2f "
            "ls=%.2f mut=%.2f elite=%.2f vmax=%d destroy=%.2f lns=%.2f",
            self.population_size,
            self.inertia_prob,
            self.cognitive_prob,
            self.social_prob,
            self.local_search_prob,
            self.mutation_prob,
            self.elite_fraction,
            self.max_velocity_len,
            self.destroy_ratio,
            self.lns_prob,
        )

    # ------------------------------------------------------------------
    # permutation helpers
    # ------------------------------------------------------------------

    def _perm_distance(self, a: List[int], b: List[int]) -> List[Tuple[int, int]]:
        """Return swap sequence to transform a -> b (clamped length)."""
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
        # apply "stronger" (longer) swaps first
        velocity = sorted(velocity, key=lambda s: abs(s[0] - s[1]), reverse=True)
        for i, j in velocity:
            if self.rng.random() < prob:
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _mutate_perm(self, perm: List[int]) -> List[int]:
        n = len(perm)
        if n < 4:
            return perm
        # combo of segment reversal and random swap
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

        for _ in range(3):  # one more pass than original
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
                        perm = perm[:i + 1] + perm[i + 1:j + 1][::-1] + perm[j + 1:]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return perm

    def _nearest_neighbor(self, start_idx: int) -> List[int]:
        customers = self.customers[:]  # KEEP original naming
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

    # --- Hyper: LNS-style destroy/repair on a single tour ---

    def _lns_destroy_repair(self, perm: List[int]) -> List[int]:
        n = len(perm)
        if n <= 4:
            return perm

        k = max(2, int(self.destroy_ratio * n))
        indices = sorted(self.rng.sample(range(n), k))
        remaining = [perm[i] for i in range(n) if i not in indices]
        removed = [perm[i] for i in indices]

        # greedy insertion
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

    # --- Adaptation: probabilities over time & diversity heuristic ---

    def _adapt_probs(self, it: int, iters: int, diversity: float):
        t = it / max(1, iters)
        # Early: more cognitive/mutation; late: more social/local search
        self.cognitive_prob = 0.2 + 0.5 * (1.0 - t)
        self.social_prob = 0.3 + 0.5 * t
        # diversity in [0,1]; if low, push mutation/lns
        if diversity < 0.3:
            self.mutation_prob = min(0.7, self.mutation_prob * 1.2)
            self.local_search_prob = min(0.8, self.local_search_prob * 1.1)
        else:
            self.mutation_prob = max(0.1, self.mutation_prob * 0.9)
        # inertia slightly decays
        self.inertia_prob = max(0.1, self.inertia_prob * (0.95 + 0.05 * (1.0 - t)))

    def _population_diversity(self, X: List[List[int]]) -> float:
        if len(X) < 2:
            return 1.0
        n = len(X[0])
        # Hamming distance over positions
        total = 0.0
        count = 0
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                diff = sum(1 for a, b in zip(X[i], X[j]) if a != b)
                total += diff / n
                count += 1
        return total / max(1, count)

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, iters: int, stop_event: Optional[object] = None):
        def _stop():
            return stop_event is not None and getattr(stop_event, "is_set", lambda: False)()

        if iters <= 0:
            raise ValueError("iters must be > 0")

        self.start_run()
        log_info("HyperPSO iterations: %d", iters)

        # ---------------- initialization ----------------

        X: List[List[int]] = []
        P: List[List[int]] = []
        P_fit: List[float] = []

        # Hyper: more diverse init (NN + random perturb + strong 2-opt)
        for i in range(self.population_size):
            perm = self._nearest_neighbor(i)
            # random perturb to break NN bias
            if len(perm) > 5:
                for _ in range(2 + (i % 3)):
                    a = self.rng.randrange(0, len(perm))
                    b = self.rng.randrange(0, len(perm))
                    if a != b:
                        perm[a], perm[b] = perm[b], perm[a]
            perm = self._two_opt(perm, max_checks=400)
            X.append(perm)
            P.append(perm[:])
            P_fit.append(self.evaluate_perm(perm))

        g = min(range(self.population_size), key=lambda i: P_fit[i])
        G = P[g][:]
        G_fit = P_fit[g]

        self.update_global_best(G, G_fit)
        log_info("Initial best: %.3f", G_fit)

        # ---------------- main loop ----------------

        for it in range(1, iters + 1):
            if _stop():
                break

            # Adapt probabilities based on diversity & time
            diversity = self._population_diversity(X)
            self._adapt_probs(it, iters, diversity)

            # sort indices by fitness (for elitism)
            idx_sorted = sorted(range(self.population_size), key=lambda i: P_fit[i])
            elite_cut = max(1, int(self.elite_fraction * self.population_size))
            elite_idx = set(idx_sorted[:elite_cut])

            for i in range(self.population_size):
                if _stop():
                    break

                # Elites: mostly local search + mild social
                if i in elite_idx:
                    # slight attraction to global best
                    v_g = self._perm_distance(X[i], G)
                    X[i] = self._apply_velocity(X[i], v_g, self.social_prob * 0.5)
                    # strong local search
                    if self.rng.random() < self.local_search_prob + 0.2:
                        X[i] = self._two_opt(X[i], max_checks=250)
                        if self.rng.random() < self.lns_prob * 0.5:
                            X[i] = self._lns_destroy_repair(X[i])
                    fit = self.evaluate_perm(X[i])
                    if fit < P_fit[i]:
                        P[i] = X[i][:]
                        P_fit[i] = fit
                    continue

                # Non-elites: full PSO update

                # inertia: keep part of current structure (implicit)
                # cognitive component
                v_p = self._perm_distance(X[i], P[i])
                X[i] = self._apply_velocity(X[i], v_p, self.cognitive_prob)

                # social component
                v_g = self._perm_distance(X[i], G)
                X[i] = self._apply_velocity(X[i], v_g, self.social_prob)

                # mutation (MANDATORY for diversity)
                if self.rng.random() < self.mutation_prob:
                    X[i] = self._mutate_perm(X[i])

                # LNS occasionally to kick hard
                if self.rng.random() < self.lns_prob:
                    X[i] = self._lns_destroy_repair(X[i])

                # local search
                if self.rng.random() < self.local_search_prob:
                    X[i] = self._two_opt(X[i], max_checks=200)

                fit = self.evaluate_perm(X[i])

                if fit < P_fit[i]:
                    P[i] = X[i][:]
                    P_fit[i] = fit

            # update global best
            g = min(range(self.population_size), key=lambda i: P_fit[i])
            if P_fit[g] < G_fit - 1e-9:
                G = P[g][:]
                G_fit = P_fit[g]
                log_info("Iter %d: NEW BEST %.3f (div=%.3f)", it, G_fit, diversity)

            # bookkeeping
            self.record_iteration(it, P_fit, X)
            self.update_global_best(G, G_fit)

            if it % 10 == 0:
                mean_fit = sum(P_fit) / len(P_fit)
                log_trace(
                    "[HyperPSO] iter=%d best=%.6f mean=%.6f div=%.3f "
                    "in=%.3f cog=%.3f soc=%.3f mut=%.3f ls=%.3f",
                    it, G_fit, mean_fit, diversity,
                    self.inertia_prob, self.cognitive_prob, self.social_prob,
                    self.mutation_prob, self.local_search_prob,
                )

        runtime = self.finalize()
        return self.best_perm, float(self.best_score), self.metrics, runtime
