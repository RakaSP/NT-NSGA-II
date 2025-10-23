# Algorithm/NT_NSGA2.py
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Utils.Logger import log_info


class PPOMemory:
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.best_scores = []
        self.mean_scores = []

    def store(self, action: Any, reward: float, best_score: float, mean_score: float):
        """Store iteration data"""
        self.actions.append(action)
        self.rewards.append(reward)
        self.best_scores.append(best_score)
        self.mean_scores.append(mean_score)

    def clear(self):
        """Clear memory for new epoch"""
        self.actions.clear()
        self.rewards.clear()
        self.best_scores.clear()
        self.mean_scores.clear()

    def print_avg_stats(self):
        """Print average action and reward after epoch finish"""
        if not self.actions:
            print("PPO Memory: No data collected")
            return

        # Calculate averages
        avg_action = np.mean(self.actions)
        avg_reward = np.mean(self.rewards)

        print(
            f"PPO Memory - Avg Action: {avg_action:.6f}, Avg Reward: {avg_reward:.6f}")


class FirstNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size // 2, 3)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        raw = self.head(h)

        population_size = self.softplus(raw[:, 0:1]) + 10.0
        crossover_rate = torch.sigmoid(raw[:, 1:2])
        mutation_rate = torch.sigmoid(raw[:, 2:3])

        return {
            "population_size": population_size.squeeze(-1),
            "crossover_rate": crossover_rate.squeeze(-1),
            "mutation_rate": mutation_rate.squeeze(-1)
        }


class SecondNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size // 2, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        raw = self.head(h)

        crossover_rate = torch.sigmoid(raw[:, 0:1])
        mutation_rate = torch.sigmoid(raw[:, 1:2])

        return {
            "crossover_rate": crossover_rate.squeeze(-1),
            "mutation_rate": mutation_rate.squeeze(-1)
        }


class NT_NSGA2(BaseAlgorithm):
    def __init__(self, vrp: Dict[str, Any], scorer: str, params: Dict[str, Any]):
        super().__init__(vrp=vrp, scorer=scorer)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        log_info(f"Using device: {self.device}")

        self.training_enabled = bool(params.get("training_enabled", False))
        self.epochs = int(params.get("epochs", 1)
                          ) if self.training_enabled else 1

        self.learning_rate = float(params.get("learning_rate", 3e-4))
        self.second_nn_learning_rate = float(
            params.get("second_nn_learning_rate", 3e-4))

        self.first_nn_input_size = 8
        self.second_nn_input_size = 15

        self.first_nn = FirstNN(self.first_nn_input_size).to(self.device)
        self.second_nn = SecondNN(self.second_nn_input_size).to(self.device)

        self.first_optimizer = optim.Adam(
            self.first_nn.parameters(), lr=self.learning_rate)
        self.second_optimizer = optim.Adam(
            self.second_nn.parameters(), lr=self.second_nn_learning_rate)

        self.training_data_first_nn = deque(maxlen=100)
        self.training_data_second_nn = deque(maxlen=1000)

        # Initialize PPO Memory
        self.ppo_memory = PPOMemory()

        self.population_size = 100.0
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.crossover_method = str(params.get("crossover_method", "ox"))
        self.mutation_method = str(params.get("mutation_method", "swap"))

        self.baseline_best_score = None
        self.best_score_so_far = None

        self.second_nn_batch_size = int(params.get("second_nn_batch_size", 32))
        self.second_nn_min_buffer_size = int(
            params.get("second_nn_min_buffer_size", 50))
        self.first_nn_batch_size = int(params.get("first_nn_batch_size", 16))
        self.first_nn_min_buffer_size = int(
            params.get("first_nn_min_buffer_size", 10))

        self.epoch_performance = []
        self.iteration_performances = []
        self.current_initial_params = {}

        valid_crossovers = ["ox", "pmx", "cx", "er"]
        if self.crossover_method not in valid_crossovers:
            raise ValueError(
                f"crossover_method must be one of {valid_crossovers}")

        valid_mutations = ["swap", "inversion", "scramble", "displacement"]
        if self.mutation_method not in valid_mutations:
            raise ValueError(
                f"mutation_method must be one of {valid_mutations}")

        self.population: List[List[int]] = []
        self.objectives: List[Tuple[float, float, float]] = []
        self.pareto_front: List[List[int]] = []
        self.pareto_scores: List[Tuple[float, float, float]] = []
        self._crowding_distance: Dict[int, float] = {}

        self._ema_iter_baseline: Optional[float] = None
        self._ema_epoch_baseline: Optional[float] = None

        log_info("NT-NSGA2 initialized: training_enabled=%s, epochs=%d, device=%s",
                 self.training_enabled, self.epochs, self.device)

    def _tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def _zscore(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        m, s = np.mean(x), np.std(x)
        return (x - m) / (s + eps)

    def _extract_vrp_features(self, vrp: Dict[str, Any]) -> np.ndarray:
        nodes = vrp["nodes"]
        vehicles = vrp["vehicles"]
        D = vrp["D"]
        T = vrp["T"]

        num_nodes = len(nodes) - 1
        num_vehicles = len(vehicles)

        demands = [node.demand for node in nodes if node.id != 0]
        total_demand = sum(demands) if demands else 0.0
        avg_demand = (total_demand / len(demands)) if demands else 0.0
        max_demand = max(demands) if demands else 0.0

        capacities = [v.max_capacity for v in vehicles] if vehicles else []
        total_capacity = sum(capacities) if capacities else 0.0
        avg_capacity = (total_capacity / len(capacities)
                        ) if capacities else 0.0

        flat_distances = D.flatten()
        flat_distances = flat_distances[flat_distances > 0]
        avg_distance = float(np.mean(flat_distances)
                             ) if flat_distances.size else 0.0

        flat_times = T.flatten()
        flat_times = flat_times[flat_times > 0]
        avg_time = float(np.mean(flat_times)) if flat_times.size else 0.0

        features = np.array([
            float(num_nodes),
            float(num_vehicles),
            float(total_demand),
            float(avg_demand),
            float(max_demand),
            float(avg_capacity),
            float(avg_distance),
            float(avg_time),
        ], dtype=np.float32)

        features = self._zscore(features)
        return features.astype(np.float32)

    def _extract_population_features(self, population: List[List[int]],
                                     objectives: List[Tuple[float, float, float]]) -> np.ndarray:
        if not population or not objectives:
            return np.zeros(self.second_nn_input_size, dtype=np.float32)

        costs = np.array([o[0] for o in objectives], dtype=np.float64)
        dists = np.array([o[1] for o in objectives], dtype=np.float64)
        times = np.array([o[2] for o in objectives], dtype=np.float64)

        diversity = self._calculate_population_diversity(population)
        convergence = self._calculate_convergence(objectives)
        pareto_ratio = len(self.pareto_front) / \
            len(population) if self.pareto_front else 0.0

        feats = np.array([
            float(np.mean(costs)), float(np.std(costs)), float(np.min(costs)),
            float(np.mean(dists)), float(np.std(dists)), float(np.min(dists)),
            float(np.mean(times)), float(np.std(times)), float(np.min(times)),
            float(diversity),
            float(convergence),
            float(pareto_ratio),
            float(self.crossover_rate),
            float(self.mutation_rate),
            float(len(population)),
        ], dtype=np.float32)

        feats = self._zscore(feats)
        return feats.astype(np.float32)

    def _calculate_population_diversity(self, population: List[List[int]]) -> float:
        if len(population) <= 1:
            return 0.0
        total = 0.0
        cnt = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = sum(1 for a, b in zip(
                    population[i], population[j]) if a != b)
                total += dist
                cnt += 1
        return total / cnt if cnt > 0 else 0.0

    def _calculate_convergence(self, objectives: List[Tuple[float, float, float]]) -> float:
        if len(objectives) <= 1:
            return 0.0
        arr = np.array(objectives, dtype=np.float64)
        cv = []
        for k in range(3):
            col = arr[:, k]
            mu = np.mean(col)
            sig = np.std(col)
            cv.append(float(sig / (abs(mu) + 1e-8)))
        return float(np.mean(cv))

    def _calculate_progressive_reward(self, current_best_score: float) -> float:
        if self.baseline_best_score is None:
            self.baseline_best_score = current_best_score
            self.best_score_so_far = current_best_score
            return 0.0

        improvement_from_baseline = (
            self.baseline_best_score - current_best_score)
        improvement_from_best = (self.best_score_so_far - current_best_score)

        baseline_norm = improvement_from_baseline / \
            (abs(self.baseline_best_score) + 1e-8)
        best_norm = improvement_from_best / \
            (abs(self.best_score_so_far) + 1e-8)

        new_best_achieved = False
        if current_best_score < self.best_score_so_far:
            self.best_score_so_far = current_best_score
            new_best_achieved = True

        baseline_reward = np.tanh(baseline_norm * 5)
        progressive_reward = np.tanh(best_norm * 10)

        reward = 0.4 * baseline_reward + 0.6 * progressive_reward

        if new_best_achieved:
            reward += 0.3
            if best_norm > 0.1:
                reward += 0.2 * best_norm

        if baseline_norm < -0.05:
            reward -= 0.2

        return float(reward)

    def _get_initial_parameters_from_nn(self):
        vrp_features = self._extract_vrp_features(self.vrp)
        x = torch.as_tensor(vrp_features, dtype=torch.float32,
                            device=self.device).unsqueeze(0)

        self.first_nn.train(False)
        with torch.no_grad():
            out = self.first_nn(x)

        self.population_size = float(out["population_size"].item())
        self.crossover_rate = float(out["crossover_rate"].item())
        self.mutation_rate = float(out["mutation_rate"].item())

        self.current_initial_params = {
            "vrp_features": vrp_features,
            "population_size": self.population_size,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
        }

        log_info("FirstNN -> population=%.6f, crossover=%.6f, mutation=%.6f",
                 self.population_size, self.crossover_rate, self.mutation_rate)

    def _get_next_parameters_from_nn(self, population: List[List[int]],
                                     objectives: List[Tuple[float, float, float]],
                                     iteration_index: int,
                                     current_performance: float):
        pop_feats = self._extract_population_features(population, objectives)
        x = torch.as_tensor(pop_feats, dtype=torch.float32,
                            device=self.device).unsqueeze(0)

        self.second_nn.train(False)
        with torch.no_grad():
            out = self.second_nn(x)

        self.crossover_rate = float(out["crossover_rate"].item())
        self.mutation_rate = float(out["mutation_rate"].item())

        current_best_score = min(
            [self._get_primary_score(obj) for obj in objectives])
        reward = self._calculate_progressive_reward(current_best_score)

        # Store in PPO memory
        # Simple average of parameters
        action_param = (self.crossover_rate + self.mutation_rate) / 2.0
        self.ppo_memory.store(
            action=action_param,
            reward=reward,
            best_score=current_best_score,
            mean_score=current_performance
        )

        self.training_data_second_nn.append({
            "features": pop_feats,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "reward": reward,
            "iter": iteration_index,
            "current_best": current_best_score,
            "best_so_far": self.best_score_so_far,
            "baseline": self.baseline_best_score,
        })

        log_info("SecondNN iter %d -> crossover=%.6f, mutation=%.6f, reward=%.6f, best=%.4f",
                 iteration_index, self.crossover_rate, self.mutation_rate, reward, current_best_score)

    def _store_first_nn_training_data(self, epoch_performance: float):
        improvement = 0.0
        if len(self.epoch_performance) > 1:
            improvement = self.epoch_performance[-2] - epoch_performance

        self.baseline_best_score = None
        self.best_score_so_far = None

        self.training_data_first_nn.append({
            "vrp_features": np.asarray(self.current_initial_params.get("vrp_features"), dtype=np.float32),
            "population_size": float(self.current_initial_params.get("population_size", 0.0)),
            "crossover_rate": float(self.current_initial_params.get("crossover_rate", 0.0)),
            "mutation_rate": float(self.current_initial_params.get("mutation_rate", 0.0)),
            "epoch_performance": float(epoch_performance),
            "improvement": float(improvement),
        })

    def _train_second_nn_online(self):
        if len(self.training_data_second_nn) < self.second_nn_min_buffer_size:
            return

        batch_size = min(self.second_nn_batch_size,
                         len(self.training_data_second_nn))
        idxs = self.rng.sample(
            range(len(self.training_data_second_nn)), batch_size)
        batch = [self.training_data_second_nn[i] for i in idxs]

        feats = np.stack([b["features"]
                         for b in batch], axis=0).astype(np.float32)
        rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
        crossover_rates = np.array([b["crossover_rate"]
                                   for b in batch], dtype=np.float32)
        mutation_rates = np.array([b["mutation_rate"]
                                  for b in batch], dtype=np.float32)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        mean_r = float(rewards.mean())
        if self._ema_iter_baseline is None:
            self._ema_iter_baseline = mean_r
        else:
            self._ema_iter_baseline = 0.95 * self._ema_iter_baseline + 0.05 * mean_r

        adv = rewards - self._ema_iter_baseline
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        feats_t = torch.as_tensor(feats, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)

        self.second_optimizer.zero_grad()

        out = self.second_nn(feats_t)
        pred_crossover = out["crossover_rate"]
        pred_mutation = out["mutation_rate"]

        target_crossover = torch.as_tensor(crossover_rates, device=self.device)
        target_mutation = torch.as_tensor(mutation_rates, device=self.device)

        crossover_loss = torch.mean(
            adv_t * (pred_crossover - target_crossover) ** 2)
        mutation_loss = torch.mean(
            adv_t * (pred_mutation - target_mutation) ** 2)

        loss = crossover_loss + mutation_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.second_nn.parameters(), max_norm=1.0)
        self.second_optimizer.step()

        log_info("SecondNN train | loss=%.6f crossover_loss=%.6f mutation_loss=%.6f adv_mean=%.4f",
                 float(loss.item()), float(
                     crossover_loss.item()), float(mutation_loss.item()),
                 float(adv.mean()))

    def _train_first_nn_offline(self):
        if len(self.training_data_first_nn) < self.first_nn_min_buffer_size:
            return

        batch_size = min(self.first_nn_batch_size,
                         len(self.training_data_first_nn))
        idxs = self.rng.sample(
            range(len(self.training_data_first_nn)), batch_size)
        batch = [self.training_data_first_nn[i] for i in idxs]

        feats = np.stack([b["vrp_features"]
                         for b in batch], axis=0).astype(np.float32)
        perfs = np.array([b["epoch_performance"]
                         for b in batch], dtype=np.float32)
        imps = np.array([b["improvement"] for b in batch], dtype=np.float32)
        pop_sizes = np.array([b["population_size"]
                             for b in batch], dtype=np.float32)
        cross_rates = np.array([b["crossover_rate"]
                               for b in batch], dtype=np.float32)
        mut_rates = np.array([b["mutation_rate"]
                             for b in batch], dtype=np.float32)

        prev_mean = float(perfs.mean()) if perfs.size else 0.0
        performance_reward = (prev_mean - perfs) / (abs(prev_mean) + 1e-8)
        improvement_reward = 0.1 * imps
        rewards = performance_reward + improvement_reward

        mean_r = float(rewards.mean())
        if self._ema_epoch_baseline is None:
            self._ema_epoch_baseline = mean_r
        else:
            self._ema_epoch_baseline = 0.9 * self._ema_epoch_baseline + 0.1 * mean_r

        adv = rewards - self._ema_epoch_baseline
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        feats_t = torch.as_tensor(feats, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)

        self.first_optimizer.zero_grad()

        out = self.first_nn(feats_t)
        pred_pop = out["population_size"]
        pred_cross = out["crossover_rate"]
        pred_mut = out["mutation_rate"]

        target_pop = torch.as_tensor(pop_sizes, device=self.device)
        target_cross = torch.as_tensor(cross_rates, device=self.device)
        target_mut = torch.as_tensor(mut_rates, device=self.device)

        pop_loss = torch.mean(adv_t * (pred_pop - target_pop) ** 2)
        cross_loss = torch.mean(adv_t * (pred_cross - target_cross) ** 2)
        mut_loss = torch.mean(adv_t * (pred_mut - target_mut) ** 2)

        loss = pop_loss + cross_loss + mut_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.first_nn.parameters(), max_norm=1.0)
        self.first_optimizer.step()

        log_info("FirstNN train | loss=%.6f pop_loss=%.6f cross_loss=%.6f mut_loss=%.6f adv=%.4f",
                 float(loss.item()), float(
                     pop_loss.item()), float(cross_loss.item()),
                 float(mut_loss.item()), float(adv.mean()))

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        if iters <= 0:
            raise ValueError("iters must be > 0")

        log_info("NT-NSGA2 starting: training_enabled=%s, epochs=%d, device=%s",
                 self.training_enabled, self.epochs, self.device)

        best_overall_perm = None
        best_overall_score = float('inf')
        best_overall_metrics = []
        total_runtime = 0.0

        for epoch in range(self.epochs):
            log_info("Starting epoch %d/%d", epoch + 1, self.epochs)
            self.iteration_performances = []

            # Clear PPO memory at the start of each epoch
            self.ppo_memory.clear()

            self.baseline_best_score = None
            self.best_score_so_far = None

            if epoch == 0 or self.training_enabled:
                self._get_initial_parameters_from_nn()

            self.start_run()

            pop_size_int = int(self.population_size)
            if pop_size_int < 2:
                pop_size_int = 2

            population = [self._initialize_individual()
                          for _ in range(pop_size_int)]
            objectives = [self._evaluate_multi_objective(
                ind) for ind in population]

            for iteration_index in range(1, iters + 1):
                offspring = self._create_offspring(population, objectives)
                offspring_objectives = [
                    self._evaluate_multi_objective(ind) for ind in offspring]

                combined_population = population + offspring
                combined_objectives = objectives + offspring_objectives

                fronts = self._fast_non_dominated_sort(combined_objectives)
                self._crowding_distance_assignment(fronts, combined_objectives)

                new_population, new_objectives = self._create_new_population(
                    fronts, combined_population, combined_objectives)

                population = new_population
                objectives = new_objectives

                self._update_pareto_front(population, objectives)

                current_perf = np.mean(
                    [self._get_primary_score(obj) for obj in objectives])
                self.iteration_performances.append(current_perf)

                if iteration_index < iters and self.training_enabled:
                    self._get_next_parameters_from_nn(
                        population, objectives, iteration_index, current_perf)
                    self._train_second_nn_online()

                primary_scores = [self._get_primary_score(
                    obj) for obj in objectives]
                self.record_iteration(iteration_index, primary_scores)
                self._update_global_best_from_pareto()

            runtime_seconds = self.finalize()
            total_runtime += runtime_seconds

            epoch_performance = np.mean(
                self.iteration_performances) if self.iteration_performances else float('inf')
            self.epoch_performance.append(epoch_performance)

            # Print PPO memory stats after epoch finishes
            self.ppo_memory.print_avg_stats()

            if self.training_enabled:
                self._store_first_nn_training_data(epoch_performance)
                self._train_first_nn_offline()

            if self.best_score < best_overall_score:
                best_overall_score = self.best_score
                best_overall_perm = self.best_perm
                best_overall_metrics = self.metrics.copy()

        self.best_perm = best_overall_perm
        self.best_score = best_overall_score
        self.metrics = best_overall_metrics

        return self.best_perm, self.best_score, self.metrics, total_runtime

    def _initialize_individual(self) -> List[int]:
        individual = self.customers[:]
        self.rng.shuffle(individual)
        return individual

    def _evaluate_multi_objective(self, perm: List[int]) -> Tuple[float, float, float]:
        self.check_constraints(perm)
        solution = self._solution_from_perm(perm)

        from Scorer.Distance import score_solution as sdist
        from Scorer.Cost import score_solution as scost

        distance = sdist(solution)
        cost = scost(solution, self.vrp["nodes"],
                     self.vrp["vehicles"], self.vrp["D"])
        time_val = self._calculate_total_time(solution)

        if not all(math.isfinite(val) for val in [distance, cost, time_val]):
            return float("inf"), float("inf"), float("inf")
        if any(val <= 0.0 for val in [distance, cost, time_val]):
            return float("inf"), float("inf"), float("inf")

        return float(cost), float(distance), float(time_val)

    def _calculate_total_time(self, solution: List[Dict[str, Any]]) -> float:
        total_time = 0.0
        T = self.vrp["T"]
        for route_data in solution:
            route = route_data["route"]
            for i in range(len(route) - 1):
                total_time += float(T[route[i], route[i + 1]])
        return total_time

    def _create_offspring(self, population: List[List[int]],
                          objectives: List[Tuple[float, float, float]]) -> List[List[int]]:
        offspring = []

        def binary_tournament() -> List[int]:
            idx1, idx2 = self.rng.sample(range(len(population)), 2)
            if self._dominates(objectives[idx1], objectives[idx2]):
                return population[idx1][:]
            elif self._dominates(objectives[idx2], objectives[idx1]):
                return population[idx2][:]
            else:
                if hasattr(self, '_crowding_distance'):
                    if self._crowding_distance.get(idx1, 0) > self._crowding_distance.get(idx2, 0):
                        return population[idx1][:]
                return population[self.rng.choice([idx1, idx2])][:]

        while len(offspring) < int(self.population_size):
            parent1 = binary_tournament()

            if self.rng.random() < self.crossover_rate:
                parent2 = binary_tournament()
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            if self.rng.random() < self.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _fast_non_dominated_sort(self, objectives: List[Tuple[float, float, float]]) -> List[List[int]]:
        n = len(objectives)
        S: List[List[int]] = [[] for _ in range(n)]
        n_count = [0] * n
        rank = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    S[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    n_count[i] += 1
            if n_count[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts[:-1]

    def _crowding_distance_assignment(self, fronts: List[List[int]],
                                      objectives: List[Tuple[float, float, float]]) -> None:
        self._crowding_distance = {}
        num_objectives = 3

        for front in fronts:
            if not front:
                continue

            for i in front:
                self._crowding_distance[i] = 0.0

            for m in range(num_objectives):
                front.sort(key=lambda i: objectives[i][m])
                self._crowding_distance[front[0]] = float('inf')
                self._crowding_distance[front[-1]] = float('inf')

                if len(front) > 2:
                    min_obj = objectives[front[0]][m]
                    max_obj = objectives[front[-1]][m]

                    if max_obj - min_obj > 1e-9:
                        for i in range(1, len(front) - 1):
                            self._crowding_distance[front[i]] += (
                                objectives[front[i + 1]][m] -
                                objectives[front[i - 1]][m]
                            ) / (max_obj - min_obj)

    def _create_new_population(self, fronts: List[List[int]],
                               combined_population: List[List[int]],
                               combined_objectives: List[Tuple[float, float, float]]):
        new_population = []
        new_objectives = []
        front_index = 0
        target = int(self.population_size)

        if target < 2:
            target = 2

        while front_index < len(fronts) and len(new_population) + len(fronts[front_index]) <= target:
            for idx in fronts[front_index]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])
            front_index += 1

        if len(new_population) < target and front_index < len(fronts):
            remaining = target - len(new_population)
            current_front = fronts[front_index]
            current_front.sort(
                key=lambda idx: self._crowding_distance.get(idx, 0.0), reverse=True)
            for i in range(min(remaining, len(current_front))):
                idx = current_front[i]
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

        return new_population, new_objectives

    def _dominates(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        better_in_any = False
        for i in range(3):
            if a[i] > b[i]:
                return False
            if a[i] < b[i]:
                better_in_any = True
        return better_in_any

    def _get_primary_score(self, objectives: Tuple[float, float, float]) -> float:
        if self.scorer == "cost":
            return objectives[0]
        elif self.scorer == "distance":
            return objectives[1]
        elif self.scorer == "time":
            return objectives[2]
        else:
            return objectives[0]

    def _update_pareto_front(self, population: List[List[int]],
                             objectives: List[Tuple[float, float, float]]) -> None:
        current_pareto = []
        current_scores = []

        for i, individual in enumerate(population):
            is_dominated = False
            for j, obj in enumerate(objectives):
                if i == j:
                    continue
                if self._dominates(obj, objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                current_pareto.append(individual)
                current_scores.append(objectives[i])

        self.pareto_front = current_pareto
        self.pareto_scores = current_scores

    def _update_global_best_from_pareto(self) -> None:
        if not self.pareto_front:
            return

        if self.scorer == "cost":
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][0])
            best_score = self.pareto_scores[best_idx][0]
        elif self.scorer == "distance":
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][1])
            best_score = self.pareto_scores[best_idx][1]
        elif self.scorer == "time":
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][2])
            best_score = self.pareto_scores[best_idx][2]
        else:
            best_idx = min(range(len(self.pareto_scores)),
                           key=lambda i: self.pareto_scores[i][0])
            best_score = self.pareto_scores[best_idx][0]

        self.update_global_best(self.pareto_front[best_idx], best_score)

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        if self.crossover_method == "ox":
            return self._order_crossover(parent1, parent2)
        elif self.crossover_method == "pmx":
            return self._partially_mapped_crossover(parent1, parent2)
        elif self.crossover_method == "cx":
            return self._cycle_crossover(parent1, parent2)
        elif self.crossover_method == "er":
            return self._edge_recombination_crossover(parent1, parent2)
        else:
            return self._order_crossover(parent1, parent2)

    def _mutate(self, individual: List[int]) -> List[int]:
        mutated = individual[:]
        if self.mutation_method == "swap":
            self._swap_mutation(mutated)
        elif self.mutation_method == "inversion":
            self._inversion_mutation(mutated)
        elif self.mutation_method == "scramble":
            self._scramble_mutation(mutated)
        elif self.mutation_method == "displacement":
            self._displacement_mutation(mutated)
        else:
            self._swap_mutation(mutated)
        return mutated

    def _order_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        length = len(first_parent)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))
        child: List[Optional[int]] = [None] * length
        child[start_index:end_index + 1] = first_parent[start_index:end_index + 1]
        remaining_genes = [gene for gene in second_parent if gene not in child]
        fill_pointer = 0
        for position in range(length):
            if child[position] is None:
                child[position] = remaining_genes[fill_pointer]
                fill_pointer += 1
        return [int(x) for x in child]

    def _partially_mapped_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        length = len(first_parent)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))
        child: List[Optional[int]] = [None] * length
        child[start_index:end_index + 1] = first_parent[start_index:end_index + 1]
        mapping = {}
        for i in range(start_index, end_index + 1):
            mapping[first_parent[i]] = second_parent[i]
        for i in range(length):
            if child[i] is None:
                candidate = second_parent[i]
                while candidate in mapping:
                    candidate = mapping[candidate]
                child[i] = candidate
        return [int(x) for x in child]

    def _cycle_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        length = len(first_parent)
        child: List[Optional[int]] = [None] * length
        cycles = []
        visited = [False] * length
        for i in range(length):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    value = first_parent[current]
                    current = second_parent.index(value)
                cycles.append(cycle)
        for i, cycle in enumerate(cycles):
            parent = first_parent if i % 2 == 0 else second_parent
            for idx in cycle:
                child[idx] = parent[idx]
        return [int(x) for x in child]

    def _edge_recombination_crossover(self, first_parent: List[int], second_parent: List[int]) -> List[int]:
        length = len(first_parent)
        edge_table: Dict[int, set] = {}
        for i in range(length):
            for node, neighbors in [(first_parent[i], [first_parent[(i-1) % length], first_parent[(i+1) % length]]),
                                    (second_parent[i], [second_parent[(i-1) % length], second_parent[(i+1) % length]])]:
                if node not in edge_table:
                    edge_table[node] = set()
                edge_table[node].update(neighbors)
        child = []
        current = self.rng.choice(first_parent)
        while len(child) < length:
            child.append(current)
            for neighbors in edge_table.values():
                neighbors.discard(current)
            if not edge_table[current]:
                remaining = [
                    node for node in first_parent if node not in child]
                if remaining:
                    current = self.rng.choice(remaining)
                else:
                    break
            else:
                current = min(edge_table[current],
                              key=lambda x: len(edge_table[x]))
        return child

    def _swap_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        i, j = self.rng.sample(range(len(permutation)), 2)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    def _inversion_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start_index, end_index = sorted(
            self.rng.sample(range(len(permutation)), 2))
        permutation[start_index:end_index +
                    1] = reversed(permutation[start_index:end_index + 1])

    def _scramble_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start_index, end_index = sorted(
            self.rng.sample(range(len(permutation)), 2))
        segment = permutation[start_index:end_index + 1]
        self.rng.shuffle(segment)
        permutation[start_index:end_index + 1] = segment

    def _displacement_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        length = len(permutation)
        start_index, end_index = sorted(self.rng.sample(range(length), 2))
        segment = permutation[start_index:end_index + 1]
        remaining = permutation[:start_index] + permutation[end_index + 1:]
        insert_pos = self.rng.randint(0, len(remaining))
        permutation.clear()
        permutation.extend(remaining[:insert_pos])
        permutation.extend(segment)
        permutation.extend(remaining[insert_pos:])

    def get_pareto_front(self) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        return self.pareto_front, self.pareto_scores

    def save_models(self, path: str):
        torch.save({
            'first_nn_state_dict': self.first_nn.state_dict(),
            'second_nn_state_dict': self.second_nn.state_dict(),
            'first_optimizer_state_dict': self.first_optimizer.state_dict(),
            'second_optimizer_state_dict': self.second_optimizer.state_dict(),
        }, path)

    def load_models(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.first_nn.load_state_dict(checkpoint['first_nn_state_dict'])
        self.second_nn.load_state_dict(checkpoint['second_nn_state_dict'])
        self.first_optimizer.load_state_dict(
            checkpoint['first_optimizer_state_dict'])
        self.second_optimizer.load_state_dict(
            checkpoint['second_optimizer_state_dict'])
        self.first_nn.to(self.device)
        self.second_nn.to(self.device)
