from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import copy
import numpy as np
import gym
from gym import spaces
import torch

from Algorithm.NSGA2 import NSGA2


class EAControlEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        nsga2: NSGA2,
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

        # Reward weights
        rp = reward_params or {}

        self.rw_best: float = float(rp.get("best", 2.0))
        self.rw_good: float = float(rp.get("good", 0.7))
        self.rw_diversity: float = float(rp.get("diversity", 0.3))
        self.rw_stall: float = float(rp.get("stall", 0.01))
        self.rw_good_tol: float = float(rp.get("good_tol", 0.05))
        self.rw_clip: float = float(rp.get("clip", 1.0))

        self._no_improve_tol: float = float(rp.get("no_improve_tol", 1e-6))

        self._prev_best: float = float("inf")
        self._prev_good_ratio: float = 0.0
        self._prev_good_diversity: float = 0.0

        self._best_so_far: float = float("inf")
        self._stall_steps: int = 0

        self._last_action: Tuple[float, float] = (0.5, 0.5)

        # Features:
        # best, mean, worst, cv, stall_steps, avg_hamming, unique_ratio
        self._obs_dim: int = 7

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.nsga2 = copy.deepcopy(self.initial_nsga2)
        self.nsga2.initialize_run_state()

        (
            self._prev_best,
            self._prev_good_ratio,
            self._prev_good_diversity,
        ) = self._good_population_stats(
            population=self.nsga2.population,
            objectives=self.nsga2.objectives,
        )

        self._best_so_far = self._prev_best
        self._stall_steps = 0
        self._last_action = (0.9, 0.2)

        obs = self._extract_features()

        if self.gym_api:
            return obs, {}

        return obs

    def step(self, action: np.ndarray):
        assert (
            len(self.nsga2.population) > 0
            and len(self.nsga2.objectives) > 0
        ), "Call reset() before step()."

        cx = float(np.clip(float(action[0]), 0.0, 1.0))
        mut = float(np.clip(float(action[1]), 0.0, 1.0))

        self.nsga2.crossover_rate = cx
        self.nsga2.mutation_rate = mut

        prev_best, prev_good_ratio, prev_good_diversity = self._good_population_stats(
            population=self.nsga2.population,
            objectives=self.nsga2.objectives,
        )

        self.nsga2.run_one_generation()

        curr_best, curr_good_ratio, curr_good_diversity = self._good_population_stats(
            population=self.nsga2.population,
            objectives=self.nsga2.objectives,
        )

        if curr_best < self._best_so_far - self._no_improve_tol:
            self._best_so_far = curr_best
            self._stall_steps = 0
        else:
            self._stall_steps += 1

        reward = self._compute_reward_from_transition(
            prev_best=prev_best,
            prev_good_ratio=prev_good_ratio,
            prev_good_diversity=prev_good_diversity,
            curr_best=curr_best,
            curr_good_ratio=curr_good_ratio,
            curr_good_diversity=curr_good_diversity,
            stall_steps=self._stall_steps,
        )

        self._prev_best = curr_best
        self._prev_good_ratio = curr_good_ratio
        self._prev_good_diversity = curr_good_diversity
        self._last_action = (cx, mut)

        obs = self._extract_features()

        if self.gym_api:
            terminated = False
            truncated = False
            info = {
                "stall_steps": int(self._stall_steps),
                "best": float(curr_best),
                "good_ratio": float(curr_good_ratio),
                "good_diversity": float(curr_good_diversity),
                "cx": float(cx),
                "mut": float(mut),
            }
            return obs, float(reward), terminated, truncated, info

        return obs, float(reward)

    def render(self):
        return

    def close(self):
        return


    def _primary_scores(
        self,
        objectives: List[Tuple[float, ...]],
    ) -> np.ndarray:
        if not objectives:
            return np.array([], dtype=np.float64)

        scores = np.array(
            [self.nsga2._primary(obj) for obj in objectives],
            dtype=np.float64,
        )

        return scores

    def _good_population_stats(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, ...]],
    ) -> Tuple[float, float, float]:
        eps = float(self.nsga2.EPS)

        if not population or not objectives:
            return float("inf"), 0.0, 0.0

        scores = self._primary_scores(objectives)

        if len(scores) == 0:
            return float("inf"), 0.0, 0.0

        finite_mask = np.isfinite(scores)

        if not np.any(finite_mask):
            return float("inf"), 0.0, 0.0

        valid_scores = scores[finite_mask]
        valid_population = [
            population[i]
            for i, ok in enumerate(finite_mask)
            if ok
        ]

        best = float(np.min(valid_scores))

        # A good solution means its score is within good_tol of current best.
        # Example: good_tol = 0.05 means within 5% of best.
        threshold = best + self.rw_good_tol * (abs(best) + eps)

        good_population = [
            valid_population[i]
            for i, score in enumerate(valid_scores)
            if score <= threshold
        ]

        if not good_population:
            return best, 0.0, 0.0

        total_valid = len(valid_population)

        # Many good solutions, but duplicates do not count twice.
        unique_good_ratio = float(
            len({tuple(ind) for ind in good_population}) / total_valid
        )

        # Diversity only among good solutions.
        if len(good_population) < 2:
            good_diversity = 0.0
        else:
            chroms = np.array(good_population, dtype=np.int64)
            sample_size = min(len(chroms), 50)

            idx = self.np_random.choice(
                len(chroms),
                size=sample_size,
                replace=False,
            )

            sample = chroms[idx]

            diffs = [
                np.mean(sample[i] != sample[j])
                for i in range(sample_size)
                for j in range(i + 1, sample_size)
            ]

            good_diversity = float(np.mean(diffs)) if diffs else 0.0

        good_diversity = float(np.clip(good_diversity, 0.0, 1.0))

        return best, unique_good_ratio, good_diversity

    def _compute_reward_from_transition(
        self,
        prev_best: float,
        prev_good_ratio: float,
        prev_good_diversity: float,
        curr_best: float,
        curr_good_ratio: float,
        curr_good_diversity: float,
        stall_steps: int,
    ) -> float:
        eps = float(self.nsga2.EPS)

        if not np.isfinite(prev_best) or not np.isfinite(curr_best):
            return 0.0

        best_gain = (prev_best - curr_best) / (abs(prev_best) + eps)

        good_gain = curr_good_ratio - prev_good_ratio
        diversity_gain = curr_good_diversity - prev_good_diversity

        reward = (
            self.rw_best * best_gain
            + self.rw_good * good_gain
            + self.rw_diversity * diversity_gain
        )

        if best_gain <= self._no_improve_tol and good_gain <= 0.0:
            reward -= self.rw_stall * min(float(stall_steps), 20.0)

        return float(np.clip(reward, -self.rw_clip, self.rw_clip))

    def _extract_features(self) -> torch.Tensor:
        eps = float(self.nsga2.EPS)
        objectives = self.nsga2.objectives
        population = self.nsga2.population

        if not objectives or not population:
            return torch.zeros(self._obs_dim, dtype=torch.float32)

        n = len(objectives)

        scores = self._primary_scores(objectives)
        finite_scores = scores[np.isfinite(scores)]

        if len(finite_scores) == 0:
            return torch.zeros(self._obs_dim, dtype=torch.float32)

        best = float(np.min(finite_scores))
        mean = float(np.mean(finite_scores))
        worst = float(np.max(finite_scores))

        std = (
            float(np.std(finite_scores, ddof=1))
            if len(finite_scores) > 1
            else 0.0
        )

        cv = float(np.clip(std / (abs(mean) + eps), 0.0, 1.0))

        stall_steps = float(self._stall_steps)

        chroms = np.array(population, dtype=np.int64)

        unique_ratio = float(len(np.unique(chroms, axis=0)) / n)

        sample_size = min(n, 50)

        idx = self.np_random.choice(
            n,
            size=sample_size,
            replace=False,
        )

        sample = chroms[idx]

        diffs = [
            np.mean(sample[i] != sample[j])
            for i in range(sample_size)
            for j in range(i + 1, sample_size)
        ]

        avg_hamming = float(np.mean(diffs)) if diffs else 0.0

        features = np.array(
            [
                best,
                mean,
                worst,
                cv,
                stall_steps,
                avg_hamming,
                unique_ratio,
            ],
            dtype=np.float32,
        )

        return torch.from_numpy(features)