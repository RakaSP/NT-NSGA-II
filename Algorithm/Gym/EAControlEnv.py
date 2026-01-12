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

    Backward-compat mode (default):
      - reset() -> obs
      - step(a) -> (obs, reward)

    If gym_api=True:
      - reset() -> (obs, info)
      - step(a) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        nsga2,
        reward_params: Optional[Dict[str, float]] = None,
        diversity_floor: float = 0.15,
        scale_ema_tau: float = 0.2,
        seed: Optional[int] = None,
        *,
        gym_api: bool = False,
    ):
        super().__init__()
        self.nsga2 = nsga2
        self.initial_nsga2 = copy.deepcopy(nsga2)

        self.gym_api = bool(gym_api)

        rp = reward_params or {}
        self.rw_alpha: float = float(rp.get("alpha", 1.0))
        self.rw_lambda: float = float(rp.get("lambda", 1.5))
        self.rw_beta: float = float(rp.get("beta", 0.5))
        self.rw_gamma: float = float(rp.get("gamma", 0.1))
        self.rw_div_floor: float = float(diversity_floor)
        self.rw_ema_tau: float = float(scale_ema_tau)

        self.rw_zeta: float = float(rp.get("zeta", 0.05))
        self.rw_stall_threshold: int = int(rp.get("stall_threshold", 5))
        self.rw_target_mut: float = float(rp.get("target_mut", 0.3))
        self._no_improve_tol: float = float(rp.get("no_improve_tol", 1e-6))

        self._reward_scale_ema: float = 0.0
        self._prev_best: float = float("inf")
        self._prev_median: float = float("inf")

        self._best_so_far: float = float("inf")
        self._stall_steps: int = 0

        self._last_action: Tuple[float, float] = (0.5, 0.5)

        self._obs_dim: int = 13

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # --------------- Gym API ---------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Gym>=0.26 passes seed here; keep working even if older code calls reset() with no args
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.nsga2 = copy.deepcopy(self.initial_nsga2)
        self.nsga2.start_run()
        n = int(self.nsga2.population_size)

        self.nsga2.population = [self.nsga2._initialize_individual() for _ in range(n)]
        self.nsga2.objectives = [self.nsga2._evaluate_multi_objective(ind) for ind in self.nsga2.population]
        self.nsga2.fronts = self.nsga2._fast_non_dominated_sort(self.nsga2.objectives)
        self.nsga2._crowding_distance_assignment(self.nsga2.fronts, self.nsga2.objectives)
        self.nsga2._update_pareto_front(
            front0=self.nsga2.fronts[0],
            population=self.nsga2.population,
            objectives=self.nsga2.objectives,
        )

        self._reward_scale_ema = 0.0
        self._prev_best, self._prev_median = self._best_and_median(self.nsga2.objectives)
        self._best_so_far = self._prev_best
        self._stall_steps = 0
        self._last_action = (0.5, 0.5)

        obs = self._extract_features()
        if self.gym_api:
            return obs, {}
        return obs

    def step(self, action: np.ndarray):
        assert len(self.nsga2.population) > 0 and len(self.nsga2.objectives) > 0, "Call reset() before step()."

        # Defensive clamp (some policies may output slightly outside [0,1])
        a0 = float(action[0])
        a1 = float(action[1])
        cx = float(np.clip(a0, 0.0, 1.0))
        mut = float(np.clip(a1, 0.0, 1.0))

        self.nsga2.crossover_rate = cx
        self.nsga2.mutation_rate = mut

        prev_best, prev_med = self._best_and_median(self.nsga2.objectives)

        self.nsga2.run_one_generation()

        curr_best, curr_med = self._best_and_median(self.nsga2.objectives)

        tol = self._no_improve_tol
        if curr_best < self._best_so_far - tol:
            self._best_so_far = curr_best
            self._stall_steps = 0
        else:
            if (abs(curr_best - prev_best) < tol) and (abs(curr_med - prev_med) < tol):
                self._stall_steps += 1
            else:
                self._stall_steps = 0

        reward = self._compute_reward_from_transition(
            prev_best=prev_best,
            prev_med=prev_med,
            curr_best=curr_best,
            curr_med=curr_med,
            cx=cx,
            mut=mut,
            stall_steps=self._stall_steps,
        )

        self._prev_best, self._prev_median = curr_best, curr_med
        self._last_action = (cx, mut)

        obs = self._extract_features()

        if self.gym_api:
            terminated = False
            truncated = False
            info = {"stall_steps": int(self._stall_steps), "best": float(curr_best), "median": float(curr_med)}
            return obs, float(reward), terminated, truncated, info

        return obs, float(reward)

    def render(self):
        return

    def close(self):
        return

    # --------------- Reward helpers ---------------

    def _primary_scores(self, objectives: List[Tuple[float, float, float]]) -> List[float]:
        # OPTIMIZED: Use list comprehension for speed
        if self.nsga2.distance_only:
            return [obj[0] for obj in objectives]
        else:
            return [self.nsga2._get_primary_score(obj) for obj in objectives]

    def _best_and_median(self, objectives: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        if not objectives:
            return float("inf"), float("inf")
        
        # OPTIMIZED: Use numpy for faster statistics
        arr = np.array([obj[0] for obj in objectives], dtype=np.float64)
        if self.nsga2.distance_only:
            arr = np.array([obj[0] for obj in objectives], dtype=np.float64)
        else:
            arr = np.array([self.nsga2._get_primary_score(obj) for obj in objectives], dtype=np.float64)
        
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return float("inf"), float("inf")
            
        best = float(np.min(arr))
        median = float(np.median(arr))
        return best, median

    def _phenotypic_diversity01(self, objectives: List[Tuple[float, float, float]]) -> float:
        scores = self._primary_scores(objectives)
        n = len(scores)
        if n == 0:
            return 0.0
        
        # OPTIMIZED: Use numpy for faster statistics
        arr = np.array(scores, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return 0.0
            
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
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
        prev_med: float,
        curr_best: float,
        curr_med: float,
        cx: float,
        mut: float,
        stall_steps: int,
    ) -> float:
        S_t = self._update_reward_scale(d_best=curr_best - prev_best, d_med=curr_med - prev_med)

        gain = max(0.0, prev_best - curr_best)
        loss = max(0.0, curr_best - prev_best)
        t1 = (self.rw_alpha * gain - self.rw_lambda * loss) / max(S_t, float(self.nsga2.EPS))

        improv = prev_med - curr_med
        t2 = self.rw_beta * (improv / max(S_t, float(self.nsga2.EPS)))

        div01 = self._phenotypic_diversity01(self.nsga2.objectives)
        t3 = self.rw_gamma * max(0.0, div01 - self.rw_div_floor)

        base_reward = t1 + t2 + t3

        plateau_bonus = 0.0
        if stall_steps >= self.rw_stall_threshold:
            plateau_bonus = self.rw_zeta * float(stall_steps) * (mut - self.rw_target_mut)

        return base_reward + plateau_bonus

    # --------------- 13-D observation extractor ---------------

    def _extract_features(self) -> torch.Tensor:
        eps = float(self.nsga2.EPS)
        objectives = self.nsga2.objectives
        
        # OPTIMIZED: Early return if no objectives
        if not objectives:
            return torch.zeros(self._obs_dim, dtype=torch.float32)
        
        n = len(objectives)
        
        # OPTIMIZED: Extract scores efficiently using numpy
        if self.nsga2.distance_only:
            scores = np.array([obj[0] for obj in objectives], dtype=np.float64)
        else:
            scores = np.array([self.nsga2._get_primary_score(obj) for obj in objectives], dtype=np.float64)
        
        # Filter finite scores
        finite_mask = np.isfinite(scores)
        finite_scores = scores[finite_mask]
        
        if len(finite_scores) == 0:
            return torch.zeros(self._obs_dim, dtype=torch.float32)
        
        # Compute statistics efficiently with numpy
        best = float(np.min(finite_scores))
        med = float(np.median(finite_scores))
        mean = float(np.mean(finite_scores))
        std = float(np.std(finite_scores, ddof=1)) if len(finite_scores) > 1 else 0.0
        smax = float(np.max(finite_scores))
        rng = smax - best
        
        cv = float(np.clip(std / (abs(mean) + eps), 0.0, 1.0))
        
        # Front calculation
        fronts = self.nsga2.fronts
        f0_frac = float(len(fronts[0]) / n) if fronts and n > 0 else 0.0
        
        # Differences
        prev_best = float(self._prev_best)
        prev_med = float(self._prev_median)
        d_best = (prev_best - best) if np.isfinite(prev_best) else 0.0
        d_med = (prev_med - med) if np.isfinite(prev_med) else 0.0
        
        # Build features efficiently with numpy array
        last_cx, last_mut = self._last_action
        stall_steps = float(self._stall_steps)
        
        # Use list comprehension for faster array creation
        extended = np.array([
            best, med, mean, std, rng, cv, f0_frac, 
            d_best, d_med, float(n), last_cx, last_mut, stall_steps
        ], dtype=np.float32)
        
        # Handle NaNs
        extended = np.nan_to_num(extended, nan=0.0, posinf=1e9, neginf=-1e9)
        return torch.from_numpy(extended)