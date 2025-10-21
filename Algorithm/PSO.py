# Algorithm/PSO.py
from __future__ import annotations

import numpy as np

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

        # Precompute pairwise customer-to-customer distance (surrogate for 2-opt)
        # Build an index map from customer id -> local index [0..n-1]
        self._cust_index = {cid: i for i, cid in enumerate(self.customers)}
        D = self.vrp["D"]
        idx = [cid for cid in self.customers]
        self._D_cust = D[np.ix_(idx, idx)].astype(float)

    # ---- helpers for permutation <-> keys ----
    def _decode(self, keys: np.ndarray):
        """Order keys ascending, then map indices to actual customer IDs."""
        order = np.argsort(keys, kind="quicksort")
        return [self.customers[int(i)] for i in order]

    def _perm_to_keys(self, perm):
        """Canonical keys for a permutation: ranks in [0,1)."""
        n = self.num_dimensions
        # position rank for each customer id
        rank = np.empty(n, dtype=float)
        for r, cid in enumerate(perm):
            rank[self._cust_index[cid]] = r
        return (rank / max(n - 1, 1)).astype(float)

    # ---- tiny 2-opt (surrogate on customer->customer distances) for intensification ----
    def _two_opt_once(self, perm):
        """Try one improving 2-opt move; return (improved_perm, improved_flag)."""
        n = len(perm)
        if n < 4:
            return perm, False
        # work on indices in the local customer-index space
        idx = [self._cust_index[c] for c in perm]
        D = self._D_cust

        # random sample of pairs to keep cost bounded
        # try up to ~min(200, n*(n-1)/2) random pairs
        trials = min(200, n * (n - 1) // 2)
        for _ in range(trials):
            a = np.random.randint(0, n - 3)
            b = np.random.randint(a + 2, n - 1)
            i, j, ip1, jp1 = idx[a], idx[b], idx[a + 1], idx[b + 1]

            # edges: (i,ip1) + (j,jp1) -> (i,j) + (ip1,jp1)
            delta = (D[i, j] + D[ip1, jp1]) - (D[i, ip1] + D[j, jp1])
            if delta < -1e-9:
                # apply reversal on perm between a+1..b
                new_perm = perm[:a + 1] + \
                    perm[a + 1:b + 1][::-1] + perm[b + 1:]
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
            # light improvement only on global best later (keep init cheap)
            keys = self._perm_to_keys(perm)
            X[i] = keys
            P[i] = keys
            P_fit[i] = self.evaluate_perm(perm)

        # Global best
        g_idx = int(np.argmin(P_fit))
        G = P[g_idx].copy()
        G_perm = self._decode(G)     # canonical -> exact perm
        G_fit = float(P_fit[g_idx])

        w = self.inertia_weight
        noise_sigma = 0.01  # small jitter to escape key ties

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
                # canonicalize the key vector to ensure attraction has effect on order
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

            # Intensify on the current iteration-best occasionally (very cheap)
            if iteration_index % 5 == 0:
                # try one 2-opt improvement step on G_perm using surrogate distance
                improved_perm, ok = self._two_opt_once(G_perm)
                if ok:
                    # accept if real scorer improved
                    new_fit = self.evaluate_perm(improved_perm)
                    if new_fit < G_fit:
                        G_perm = improved_perm
                        G = self._perm_to_keys(G_perm)
                        G_fit = float(new_fit)

            # Metrics and global best tracking
            self.record_iteration(iteration_index, P_fit)
            self.update_global_best(G_perm, float(G_fit))

            # very mild inertia decay
            w = max(0.4, w * 0.995)

        runtime_s = self.finalize()
        return G_perm, float(G_fit), self.metrics, runtime_s
