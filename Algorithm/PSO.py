from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class PSO(BaseAlgorithm):
    def __init__(self, vrp, scorer, params, seed: int = 0):
        super().__init__(vrp=vrp, scorer=scorer, seed=seed)

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

        log_info(
            "PSO params: population_size=%d, inertia_weight=%.3f, cognitive_weight=%.3f, social_weight=%.3f",
            self.population_size, self.inertia_weight, self.cognitive_weight, self.social_weight
        )

        self.num_dimensions = len(self.customers)
        self._cust_index = {cid: i for i, cid in enumerate(self.customers)}
        self._D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

    def _decode(self, keys: np.ndarray):
        order = np.argsort(keys, kind="quicksort")
        return [self.customers[int(i)] for i in order]

    def _perm_to_keys(self, perm):
        n = self.num_dimensions
        rank = np.empty(n, dtype=float)
        for r, cid in enumerate(perm):
            rank[self._cust_index[cid]] = r
        return (rank / max(n - 1, 1)).astype(float)

    def _exhaustive_two_opt(self, perm, max_no_improve=50):
        """Exhaustive 2-opt until no improvement found"""
        n = len(perm)
        if n < 4:
            return perm
        
        D = self._D
        improved = True
        no_improve_count = 0
        
        while improved and no_improve_count < max_no_improve:
            improved = False
            best_delta = 0
            best_move = None
            
            # Check all possible 2-opt moves
            for a in range(n - 3):
                for b in range(a + 2, min(n - 1, a + 30)):  # Window size
                    i, ip1 = perm[a], perm[a + 1]
                    j, jp1 = perm[b], perm[b + 1]

                    delta = (float(D[i][j]) + float(D[ip1][jp1])) - (float(D[i][ip1]) + float(D[j][jp1]))
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (a, b)
                        improved = True
            
            if improved and best_move:
                a, b = best_move
                perm = perm[:a + 1] + perm[a + 1:b + 1][::-1] + perm[b + 1:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return perm

    def _or_opt(self, perm):
        """Or-opt: relocate sequences of 1-3 nodes"""
        n = len(perm)
        if n < 4:
            return perm, False
        
        D = self._D
        improved_total = False
        
        for seq_len in [1, 2, 3]:
            if seq_len >= n - 1:
                continue
                
            improved = True
            while improved:
                improved = False
                best_delta = 0
                best_move = None
                
                for i in range(n - seq_len):
                    # Try moving sequence starting at i
                    for j in range(n):
                        if j >= i and j <= i + seq_len:
                            continue
                        
                        # Calculate removal cost
                        if i == 0:
                            remove_cost = -float(D[perm[i + seq_len]][perm[i + seq_len + 1]]) if i + seq_len + 1 < n else 0
                        else:
                            before = perm[i - 1]
                            after = perm[i + seq_len] if i + seq_len < n else perm[0]
                            remove_cost = float(D[before][after]) - float(D[before][perm[i]])
                            if i + seq_len < n:
                                remove_cost -= float(D[perm[i + seq_len - 1]][after])
                        
                        # Calculate insertion cost
                        if j == 0:
                            insert_cost = float(D[perm[i]][perm[0]]) if n > 0 else 0
                        else:
                            before_j = perm[j - 1] if j > 0 else perm[-1]
                            after_j = perm[j] if j < n else perm[0]
                            insert_cost = float(D[before_j][perm[i]]) + float(D[perm[i + seq_len - 1]][after_j]) - float(D[before_j][after_j])
                        
                        delta = remove_cost + insert_cost
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_move = (i, j, seq_len)
                            improved = True
                
                if improved and best_move:
                    i, j, seq_len = best_move
                    sequence = perm[i:i + seq_len]
                    perm = perm[:i] + perm[i + seq_len:]
                    if j > i:
                        j -= seq_len
                    perm = perm[:j] + sequence + perm[j:]
                    improved_total = True
        
        return perm, improved_total

    def _nearest_neighbor_init(self, start_idx):
        """Greedy nearest neighbor heuristic for initialization"""
        D = self._D
        customers = self.customers[:]
        
        perm = [customers[start_idx]]
        remaining = set(customers)
        remaining.remove(customers[start_idx])
        
        while remaining:
            current = perm[-1]
            nearest = min(remaining, key=lambda c: float(D[current][c]))
            perm.append(nearest)
            remaining.remove(nearest)
        
        return perm

    def _adaptive_parameters(self, iteration, max_iters, stagnation):
        """Aggressive adaptive parameters"""
        progress = iteration / max_iters
        
        # More aggressive decay
        w_max, w_min = self.inertia_weight, 0.2
        w = w_max - (w_max - w_min) * (progress ** 2)
        
        # Boost exploration when stagnating
        if stagnation > 15:
            w = min(w * 1.5, w_max)
        
        c1 = self.cognitive_weight * (0.5 + progress)
        c2 = self.social_weight * (2.0 - progress)
        
        return w, c1, c2

    def _swap_mutation(self, perm, strength=3):
        """Random swap mutation for diversity"""
        n = len(perm)
        perm = perm[:]
        for _ in range(strength):
            i, j = np.random.randint(0, n, 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def solve(self, iters: int, stop_event: Optional[object] = None):
        def _stop() -> bool:
            return stop_event is not None and getattr(stop_event, "is_set", lambda: False)()

        if iters <= 0:
            raise ValueError("iters must be > 0")
        log_info("Iterations: %d", iters)
        self.start_run()

        np_seed = self.rng.randrange(2**32)
        rng_np = np.random.default_rng(np_seed)

        # Superior initialization strategy
        X = np.empty((self.population_size, self.num_dimensions))
        init_perms = []
        
        for i in range(self.population_size):
            if i < min(5, self.population_size):
                # Greedy nearest neighbor from different starts
                perm = self._nearest_neighbor_init(i % self.num_dimensions)
                perm = self._exhaustive_two_opt(perm, max_no_improve=30)
            elif i < self.population_size // 3:
                # Random with local search
                perm = self.customers[:]
                np.random.shuffle(perm)
                perm = self._exhaustive_two_opt(perm, max_no_improve=20)
            else:
                # Pure random for diversity
                perm = self.customers[:]
                np.random.shuffle(perm)
            
            init_perms.append(perm)
            X[i] = self._perm_to_keys(perm)
        
        V = rng_np.standard_normal((self.population_size, self.num_dimensions)) * 0.05

        P = X.copy()
        P_fit = np.empty(self.population_size, dtype=float)

        # Evaluate initial population
        for i in range(self.population_size):
            if _stop():
                break
            P_fit[i] = self.evaluate_perm(init_perms[i])

        for i in range(self.population_size):
            if not np.isfinite(P_fit[i]):
                P_fit[i] = float("inf")

        g_idx = int(np.argmin(P_fit))
        G = P[g_idx].copy()
        G_perm = self._decode(G)
        G_fit = float(P_fit[g_idx])

        if np.isfinite(G_fit):
            self.update_global_best(G_perm, float(G_fit))

        stagnation_counter = 0
        last_improvement = 0
        best_history = [G_fit]

        stopped = False

        for iteration_index in range(1, iters + 1):
            if _stop():
                stopped = True
                break

            # Adaptive parameters
            w, c1, c2 = self._adaptive_parameters(iteration_index, iters, stagnation_counter)
            
            # Dynamic noise based on stagnation
            noise_sigma = 0.005 * (1 + stagnation_counter * 0.3)

            R1 = rng_np.random((self.population_size, self.num_dimensions))
            R2 = rng_np.random((self.population_size, self.num_dimensions))

            V = (
                w * V
                + c1 * R1 * (P - X)
                + c2 * R2 * (G - X)
                + noise_sigma * rng_np.standard_normal(V.shape)
            )
            
            V = np.clip(V, -0.3, 0.3)
            X = np.clip(X + V, 0.0, 1.0)

            fits = np.empty(self.population_size, dtype=float)
            fits.fill(float("inf"))

            for i in range(self.population_size):
                if _stop():
                    stopped = True
                    break
                perm = self._decode(X[i])
                
                # Apply light local search to some particles
                if i < self.population_size // 3 and iteration_index % 2 == 0:
                    perm = self._exhaustive_two_opt(perm, max_no_improve=10)
                
                X[i] = self._perm_to_keys(perm)
                fits[i] = self.evaluate_perm(perm)

            if stopped:
                break

            improved = fits < P_fit
            if np.any(improved):
                P[improved] = X[improved]
                P_fit[improved] = fits[improved]

            prev_g_fit = G_fit
            g_idx = int(np.argmin(P_fit))
            if P_fit[g_idx] < G_fit:
                G = P[g_idx].copy()
                G_perm = self._decode(G)
                G_fit = float(P_fit[g_idx])
                stagnation_counter = 0
                last_improvement = iteration_index
            else:
                stagnation_counter += 1

            # AGGRESSIVE LOCAL SEARCH on global best
            if not _stop():
                # Full 2-opt every iteration
                improved_perm = self._exhaustive_two_opt(G_perm, max_no_improve=40)
                new_fit = self.evaluate_perm(improved_perm)
                if new_fit < G_fit:
                    G_perm = improved_perm
                    G = self._perm_to_keys(G_perm)
                    G_fit = float(new_fit)
                    stagnation_counter = 0
                
                # Or-opt every few iterations
                if iteration_index % 2 == 0:
                    improved_perm, ok = self._or_opt(G_perm)
                    if ok:
                        new_fit = self.evaluate_perm(improved_perm)
                        if new_fit < G_fit:
                            G_perm = improved_perm
                            G = self._perm_to_keys(G_perm)
                            G_fit = float(new_fit)
                            stagnation_counter = 0

            # Diversity mechanisms
            if stagnation_counter > 25:
                # Restart worst 40% of population
                worst_indices = np.argsort(P_fit)[-int(self.population_size * 0.4):]
                for idx in worst_indices:
                    if rng_np.random() < 0.5:
                        # Mutation of global best
                        mutated = self._swap_mutation(G_perm, strength=5)
                        X[idx] = self._perm_to_keys(mutated)
                    else:
                        # Random restart with greedy init
                        start = rng_np.integers(0, self.num_dimensions)
                        new_perm = self._nearest_neighbor_init(start)
                        X[idx] = self._perm_to_keys(new_perm)
                    
                    V[idx] = rng_np.standard_normal(self.num_dimensions) * 0.1
                    P[idx] = X[idx]
                    P_fit[idx] = self.evaluate_perm(self._decode(X[idx]))
                
                stagnation_counter = 0
            
            # Emergency restart if stuck too long
            if iteration_index - last_improvement > 50:
                log_info(f"Emergency restart at iteration {iteration_index}")
                for i in range(self.population_size // 2, self.population_size):
                    new_perm = self._nearest_neighbor_init(i % self.num_dimensions)
                    new_perm = self._exhaustive_two_opt(new_perm, max_no_improve=20)
                    X[i] = self._perm_to_keys(new_perm)
                    V[i] = rng_np.standard_normal(self.num_dimensions) * 0.1
                    P[i] = X[i]
                    P_fit[i] = self.evaluate_perm(new_perm)
                last_improvement = iteration_index

            best_history.append(G_fit)
            perms = [self._decode(P[i]) for i in range(self.population_size)]
            best_history.append(G_fit)
            self.record_iteration(iteration_index, P_fit, perms)
            self.update_global_best(G_perm, float(G_fit))

            if iteration_index % 10 == 0:
                log_info(f"Iter {iteration_index}: Best={G_fit:.2f}, Stagnation={stagnation_counter}")

        runtime_s = self.finalize()

        if getattr(self, "best_perm", None) is not None and getattr(self, "best_score", None) is not None:
            return self.best_perm, float(self.best_score), self.metrics, runtime_s

        return G_perm, float(G_fit), self.metrics, runtime_s