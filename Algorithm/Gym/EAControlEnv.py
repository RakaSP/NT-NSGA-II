# Algorithm/Gym/EAControlEnv.py
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
    - Reward: best-change + median-shift + diversity-floor
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
        
        self.initial_nsga2 = copy.deepcopy(nsga2)

        # Reward weights
        rp = reward_params or {}
        self.rw_alpha: float = float(rp.get("alpha", 1.0))    # best-change gain
        self.rw_lambda: float = float(rp.get("lambda", 1.5))  # best-change loss
        self.rw_beta: float = float(rp.get("beta", 0.5))      # population shift
        self.rw_gamma: float = float(rp.get("gamma", 0.1))    # diversity floor bonus
        self.rw_div_floor: float = float(diversity_floor)
        self.rw_ema_tau: float = float(scale_ema_tau)

        # Progress anchors (env-only scalars)
        self._reward_scale_ema: float = 0.0
        self._prev_best: float = float("inf")
        self._prev_median: float = float("inf")
        self._last_action: Tuple[float, float] = (0.5, 0.5)  # neutral defaults

        # Spaces
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self._obs_dim: int = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        # Seed RNG (optional)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # --------------- Gym API ---------------

    def reset(self):
        # Initialize NSGA-II state on the nsga2 instance (single source of truth)
        self.nsga2 = copy.deepcopy(self.initial_nsga2)
        self.nsga2.start_run()
        n = int(self.nsga2.population_size)
        self.nsga2.population = [self.nsga2._initialize_individual() for _ in range(n)]
        self.nsga2.objectives = [self.nsga2._evaluate_multi_objective(ind) for ind in self.nsga2.population]
        self.nsga2.fronts = self.nsga2._fast_non_dominated_sort(self.nsga2.objectives)
        self.nsga2._crowding_distance_assignment(self.nsga2.fronts, self.nsga2.objectives)
        self.nsga2._update_pareto_front(self.nsga2.population, self.nsga2.objectives)
        
        # Reset anchors
        self._reward_scale_ema = 0.0
        self._prev_best, self._prev_median = self._best_and_median(self.nsga2.objectives)
        self._last_action = (0.5, 0.5)

        return self._extract_features()

    def step(self, action: np.ndarray):
        """
        action: np.array([cx, mut]) in [0,1]
        """
        assert len(self.nsga2.population) > 0 and len(self.nsga2.objectives) > 0, "Call reset() before step()."

        # Set hyperparams on the nsga2 instance
        cx = float(action[0]); mut = float(action[1])
        self.nsga2.crossover_rate = cx
        self.nsga2.mutation_rate = mut

        # Snapshot previous summary (for reward)
        prev_best, prev_med = self._best_and_median(self.nsga2.objectives)

        # Advance one generation inside NSGA-II
        self.nsga2.run_one_generation()

        # Reward from transition (Terms 1..3)
        reward = self._compute_reward_from_transition(
            prev_best=prev_best,
            prev_med=prev_med,
            curr_best=self._best_and_median(self.nsga2.objectives)[0],
            curr_med=self._best_and_median(self.nsga2.objectives)[1],
        )

        # Update anchors
        self._prev_best, self._prev_median = self._best_and_median(self.nsga2.objectives)
        self._last_action = (cx, mut)

        obs = self._extract_features()
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
    ) -> float:
        S_t = self._update_reward_scale(d_best=curr_best - prev_best, d_med=curr_med - prev_med)

        # Term 1: best-change (gain vs loss)
        gain = max(0.0, prev_best - curr_best)
        loss = max(0.0, curr_best - prev_best)
        t1 = (self.rw_alpha * gain - self.rw_lambda * loss) / max(S_t, float(self.nsga2.EPS))

        # Term 2: median shift
        improv = prev_med - curr_med
        t2 = self.rw_beta * (improv / max(S_t, float(self.nsga2.EPS)))

        # Term 3: diversity floor
        div01 = self._phenotypic_diversity01(self.nsga2.objectives)
        t3 = self.rw_gamma * max(0.0, div01 - self.rw_div_floor)

        return t1 + t2 + t3

    # --------------- 21-D observation extractor ---------------

    def _extract_features(self) -> torch.Tensor:
        """
        10-D robust feature vector:
        [best, median, mean, std, range, cv, f0_frac, delta_best, delta_median, pop_size]
        Returns torch.float32 with no NaNs/Infs.
        """
        eps = float(self.nsga2.EPS)
        objectives = self.nsga2.objectives
        scores = np.asarray(self._primary_scores(objectives), dtype=np.float64)

        n = int(len(scores))
        if n == 0 or not np.isfinite(scores).any():
            return torch.zeros(10, dtype=torch.float32)

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

        feat = np.nan_to_num(feat, nan=0.0, posinf=1e9, neginf=-1e9)
        return torch.from_numpy(feat)

