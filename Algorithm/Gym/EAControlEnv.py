from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import gym
from gym import spaces
import torch
import copy
from Utils.Logger import log_trace


class EAControlEnv(gym.Env):
    """
    Controls one NSGA-II generation by choosing (p_c, p_m).
    - Action: Box([0,1]^2) -> (crossover_rate, mutation_rate)
    - Reward: best-change + median-shift + diversity-floor + plateau-exploration
    NOTE: The env does NOT store GA state. It reads/writes on the provided nsga2 object.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        nsga2,
        reward_params: Optional[Dict[str, float]] = None,
        diversity_floor: float = 0.15,
        scale_ema_tau: float = 0.2,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.nsga2 = nsga2

        # Keep a copy of initial NSGA-II state so reset() can restart cleanly
        self.initial_nsga2 = copy.deepcopy(nsga2)

        # Reward weights
        rp = reward_params or {}
        self.rw_alpha: float = float(rp.get("alpha", 1.0))    # best-change gain
        self.rw_lambda: float = float(rp.get("lambda", 1.5))  # best-change loss
        self.rw_beta: float = float(rp.get("beta", 0.5))      # population shift
        self.rw_gamma: float = float(rp.get("gamma", 0.1))    # diversity floor bonus
        self.rw_div_floor: float = float(diversity_floor)
        self.rw_ema_tau: float = float(scale_ema_tau)

        # NEW: exploration shaping params (for plateaus / stagnation)
        self.rw_zeta: float = float(rp.get("zeta", 0.05))           # bonus scale
        self.rw_stall_threshold: int = int(rp.get("stall_threshold", 5))
        self.rw_target_mut: float = float(rp.get("target_mut", 0.3))
        self._no_improve_tol: float = float(rp.get("no_improve_tol", 1e-6))

        # Progress anchors (env-only scalars)
        self._reward_scale_ema: float = 0.0
        self._prev_best: float = float("inf")
        self._prev_median: float = float("inf")

        # Track best-so-far and stagnation length (for plateau detection)
        self._best_so_far: float = float("inf")
        self._stall_steps: int = 0

        # Last chosen action (cx, mut)
        self._last_action: Tuple[float, float] = (0.5, 0.5)  # neutral defaults

        # Spaces
        # 10 baseline stats + 2 last action + 1 stall counter = 13-D
        self._obs_dim: int = 13

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        # Seed RNG (optional)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # --------------- Gym API ---------------

    def reset(self):
        """
        Reinitialize NSGA-II and all env-only state.
        """
        # Reset NSGA-II to initial configuration and start a fresh run
        self.nsga2 = copy.deepcopy(self.initial_nsga2)
        self.nsga2.start_run()
        n = int(self.nsga2.population_size)

        # Initialize population and evaluate
        self.nsga2.population = [
            self.nsga2._initialize_individual() for _ in range(n)
        ]
        self.nsga2.objectives = [
            self.nsga2._evaluate_multi_objective(ind)
            for ind in self.nsga2.population
        ]
        self.nsga2.fronts = self.nsga2._fast_non_dominated_sort(
            self.nsga2.objectives
        )
        self.nsga2._crowding_distance_assignment(
            self.nsga2.fronts, self.nsga2.objectives
        )
        self.nsga2._update_pareto_front(
            front0=self.nsga2.fronts[0],
            population=self.nsga2.population,
            objectives=self.nsga2.objectives,
        )

        # Reset anchors
        self._reward_scale_ema = 0.0
        self._prev_best, self._prev_median = self._best_and_median(
            self.nsga2.objectives
        )
        self._best_so_far = self._prev_best
        self._stall_steps = 0
        self._last_action = (0.5, 0.5)

        obs = self._extract_features()
        return obs

    def step(self, action: np.ndarray):
        """
        action: np.array([cx, mut]) in [0,1]
        """
        assert len(self.nsga2.population) > 0 and len(self.nsga2.objectives) > 0, \
            "Call reset() before step()."

        # Set hyperparams on the nsga2 instance
        cx = float(action[0])
        mut = float(action[1])
        self.nsga2.crossover_rate = cx
        self.nsga2.mutation_rate = mut

        # Snapshot previous summary (for reward)
        prev_best, prev_med = self._best_and_median(self.nsga2.objectives)

        # Advance one generation inside NSGA-II
        self.nsga2.run_one_generation()

        # Current summary
        curr_best, curr_med = self._best_and_median(self.nsga2.objectives)

        # ------ plateau / stagnation tracking ------
        tol = self._no_improve_tol

        # Update "best-so-far" and stall length
        if curr_best < self._best_so_far - tol:
            # New global improvement -> reset stagnation
            self._best_so_far = curr_best
            self._stall_steps = 0
        else:
            # No new global improvement, look at 1-step change
            if (abs(curr_best - prev_best) < tol) and (abs(curr_med - prev_med) < tol):
                self._stall_steps += 1
            else:
                self._stall_steps = 0

        # Reward from transition (Terms 1..3 + plateau exploration)
        reward = self._compute_reward_from_transition(
            prev_best=prev_best,
            prev_med=prev_med,
            curr_best=curr_best,
            curr_med=curr_med,
            cx=cx,
            mut=mut,
            stall_steps=self._stall_steps,
        )

        # Update anchors
        self._prev_best, self._prev_median = curr_best, curr_med
        self._last_action = (cx, mut)

        obs = self._extract_features()
        # You were returning (obs, reward) originally, so keep that API
        return obs, float(reward)

    def render(self):  # no-op
        return

    def close(self):   # no-op
        return

    # --------------- Reward helpers ---------------

    def _primary_scores(self, objectives: List[Tuple[float, float, float]]) -> List[float]:
        return [self.nsga2._get_primary_score(o) for o in objectives]

    def _best_and_median(self, objectives: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        arr = sorted(self._primary_scores(objectives))
        if not arr:
            return float("inf"), float("inf")
        n = len(arr)
        best = arr[0]
        median = arr[n // 2] if (n % 2 == 1) else 0.5 * (arr[n // 2 - 1] + arr[n // 2])
        return best, median

    def _phenotypic_diversity01(self, objectives: List[Tuple[float, float, float]]) -> float:
        scores = self._primary_scores(objectives)
        n = len(scores)
        if n == 0:
            return 0.0
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
        denom = abs(mean) + float(self.nsga2.EPS)
        cv = std / denom
        return float(np.clip(cv, 0.0, 1.0))

    def _update_reward_scale(self, d_best: float, d_med: float) -> float:
        raw = abs(d_best) + abs(d_med)
        tau = self.rw_ema_tau
        self._reward_scale_ema = (1.0 - tau) * self._reward_scale_ema + tau * raw
        self._reward_scale_ema = max(self._reward_scale_ema, float(self.nsga2.EPS))
        return self._reward_scale_ema

    def _compute_reward_from_transition(
        self,
        prev_best: float,
        prev_med:  float,
        curr_best: float,
        curr_med:  float,
        cx: float,
        mut: float,
        stall_steps: int,
    ) -> float:
        """
        Main reward:
        - Term 1: best-change (gain vs loss)
        - Term 2: median shift
        - Term 3: diversity floor
        - Term 4: exploration bonus when plateauing (encourages higher mutation)
        """
        S_t = self._update_reward_scale(
            d_best=curr_best - prev_best,
            d_med=curr_med - prev_med,
        )

        # --- Term 1: best-change (gain vs loss) ---
        gain = max(0.0, prev_best - curr_best)
        loss = max(0.0, curr_best - prev_best)
        t1 = (self.rw_alpha * gain - self.rw_lambda * loss) / max(
            S_t, float(self.nsga2.EPS)
        )

        # --- Term 2: median shift ---
        improv = prev_med - curr_med
        t2 = self.rw_beta * (improv / max(S_t, float(self.nsga2.EPS)))

        # --- Term 3: diversity floor ---
        div01 = self._phenotypic_diversity01(self.nsga2.objectives)
        t3 = self.rw_gamma * max(0.0, div01 - self.rw_div_floor)

        base_reward = t1 + t2 + t3

        # --- Term 4: exploration bonus on plateau ---
        # When stagnation gets long, the reward slightly prefers higher mutation rates.
        plateau_bonus = 0.0
        if stall_steps >= self.rw_stall_threshold:
            # Linear in (mut - target_mut) and in stall_steps
            plateau_bonus = (
                self.rw_zeta * float(stall_steps) * (mut - self.rw_target_mut)
            )

        return base_reward + plateau_bonus

    # --------------- 13-D observation extractor ---------------

    def _extract_features(self) -> torch.Tensor:
        """
        13-D robust feature vector:
        [best, median, mean, std, range, cv, f0_frac, d_best, d_med, pop_size,
         last_cx, last_mut, stall_steps]
        Returns torch.float32 with no NaNs/Infs.
        """
        eps = float(self.nsga2.EPS)
        objectives = self.nsga2.objectives
        scores = np.asarray(self._primary_scores(objectives), dtype=np.float64)

        n = int(len(scores))
        if n == 0 or not np.isfinite(scores).any():
            return torch.zeros(self._obs_dim, dtype=torch.float32)

        finite_scores = scores[np.isfinite(scores)]
        best = float(np.min(finite_scores))
        med  = float(np.median(finite_scores))
        mean = float(np.mean(finite_scores))
        std  = float(np.std(finite_scores, ddof=1)) if finite_scores.size > 1 else 0.0
        smax = float(np.max(finite_scores))
        rng  = smax - best

        # bounded, scale-free spread
        cv = float(np.clip(std / (abs(mean) + eps), 0.0, 1.0))

        # fraction in Pareto front 0
        fronts = self.nsga2.fronts
        f0_frac = float(len(fronts[0]) / n) if fronts and n > 0 else 0.0

        # trend vs previous generation
        prev_best = float(self._prev_best)
        prev_med  = float(self._prev_median)
        d_best = (prev_best - best) if np.isfinite(prev_best) else 0.0
        d_med  = (prev_med  - med)  if np.isfinite(prev_med)  else 0.0

        # base 10 features
        feat = np.array([
            best,          # 1
            med,           # 2
            mean,          # 3
            std,           # 4
            rng,           # 5
            cv,            # 6
            f0_frac,       # 7
            d_best,        # 8
            d_med,         # 9
            float(n),      # 10
        ], dtype=np.float32)

        # add last action + stall_steps
        last_cx, last_mut = self._last_action
        stall_steps = float(self._stall_steps)

        extended = np.concatenate([
            feat,
            np.array([last_cx, last_mut, stall_steps], dtype=np.float32),
        ])

        extended = np.nan_to_num(extended, nan=0.0, posinf=1e9, neginf=-1e9)
        return torch.from_numpy(extended)
