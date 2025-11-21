from __future__ import annotations

import numpy as np
from typing import Mapping

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info

class PSO(BaseAlgorithm):
    def __init__(self, vrp, scorer, params):
        super().__init__(vrp=vrp, scorer=scorer)

        population_size = int(params.get("population_size"))
        inertia_weight = float(params.get("inertia_weight"))
        cognitive_weight = float(params.get("cognitive_weight"))
        social_weight = float(params.get("social_weight"))

        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        for name, value in [
            ("inertia_weight", inertia_weight),
            ("cognitive_weight", cognitive_weight),
            ("social_weight", social_weight),
        ]:
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        log_info("PSO params: population_size=%d, inertia_weight=%.3f, cognitive_weight=%.3f, social_weight=%.3f",
                 self.population_size, self.inertia_weight, self.cognitive_weight, self.social_weight)

        self.num_dimensions = len(self.customers)

        # Map from customer ID -> local index [0..n-1] (for key canonicalization only)
        self._cust_index = {cid: i for i, cid in enumerate(self.customers)}
        # Distance dict (ID-based) used directly in 2-opt surrogate below
        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

    # ---- helpers for permutation <-> keys ----
    def _decode(self, keys: np.ndarray):
        """Order keys ascending, then map indices to actual customer IDs."""
        order = np.argsort(keys, kind="quicksort")
        return [self.customers[int(i)] for i in order]

    def _perm_to_keys(self, perm):
        """Canonical keys for a permutation: ranks in [0,1)."""
        n = self.num_dimensions
        rank = np.empty(n, dtype=float)
        for r, cid in enumerate(perm):
            rank[self._cust_index[cid]] = r
        return (rank / max(n - 1, 1)).astype(float)

    # ---- tiny 2-opt for intensification (uses dict D on IDs) ----
    def _two_opt_once(self, perm):
        """Try one improving 2-opt move; return (improved_perm, improved_flag)."""
        n = len(perm)
        if n < 4:
            return perm, False
        D = self._D
        # random sample of pairs to keep cost bounded
        trials = min(200, n * (n - 1) // 2)
        for _ in range(trials):
            a = np.random.randint(0, n - 3)
            b = np.random.randint(a + 2, n - 1)
            i, ip1 = perm[a], perm[a + 1]
            j, jp1 = perm[b], perm[b + 1]

            delta = (float(D[i][j]) + float(D[ip1][jp1])) - (float(D[i][ip1]) + float(D[j][jp1]))
            if delta < -1e-9:
                # apply reversal on perm between a+1..b
                new_perm = perm[:a + 1] + perm[a + 1:b + 1][::-1] + perm[b + 1:]
                return new_perm, True
        return perm, False

    def solve(self, iters: int):
        if iters <= 0:
            raise ValueError("iters must be > 0")
        log_info("Iterations: %d", iters)
        self.start_run()

        # numpy RNG seeded from python RNG
        np_seed = self.rng.randrange(2**32)
        rng_np = np.random.default_rng(np_seed)

        # Initialize swarm with canonicalized keys of decoded permutations
        X = rng_np.random((self.population_size, self.num_dimensions))
        V = np.zeros((self.population_size, self.num_dimensions), dtype=float)

        # Personal bests (canonicalized)
        P = np.empty_like(X)
        P_fit = np.empty(self.population_size, dtype=float)
        for i in range(self.population_size):
            perm = self._decode(X[i])
            keys = self._perm_to_keys(perm)
            X[i] = keys
            P[i] = keys
            P_fit[i] = self.evaluate_perm(perm)

        # Global best
        g_idx = int(np.argmin(P_fit))
        G = P[g_idx].copy()
        G_perm = self._decode(G)
        G_fit = float(P_fit[g_idx])

        w = self.inertia_weight
        noise_sigma = 0.01  # small jitter

        for iteration_index in range(1, iters + 1):
            # Velocity/position update toward canonical P and G
            R1 = rng_np.random((self.population_size, self.num_dimensions))
            R2 = rng_np.random((self.population_size, self.num_dimensions))

            V = (
                w * V
                + self.cognitive_weight * R1 * (P - X)
                + self.social_weight * R2 * (G - X)
                + noise_sigma * rng_np.standard_normal(V.shape)
            )
            X = np.clip(X + V, 0.0, 1.0)

            # Evaluate new positions; canonicalize X to the decoded permutation
            fits = np.empty(self.population_size, dtype=float)
            for i in range(self.population_size):
                perm = self._decode(X[i])
                X[i] = self._perm_to_keys(perm)
                fits[i] = self.evaluate_perm(perm)

            # Personal best updates
            improved = fits < P_fit
            if np.any(improved):
                P[improved] = X[improved]
                P_fit[improved] = fits[improved]

            # Global best update
            g_idx = int(np.argmin(P_fit))
            if P_fit[g_idx] < G_fit:
                G = P[g_idx].copy()
                G_perm = self._decode(G)
                G_fit = float(P_fit[g_idx])

            # Intensify occasionally
            if iteration_index % 5 == 0:
                improved_perm, ok = self._two_opt_once(G_perm)
                if ok:
                    new_fit = self.evaluate_perm(improved_perm)
                    if new_fit < G_fit:
                        G_perm = improved_perm
                        G = self._perm_to_keys(G_perm)
                        G_fit = float(new_fit)

            # Metrics and global best tracking
            self.record_iteration(iteration_index, P_fit)
            self.update_global_best(G_perm, float(G_fit))

            # mild inertia decay
            w = max(0.4, w * 0.995)

        runtime_s = self.finalize()
        return G_perm, float(G_fit), self.metrics, runtime_s
