from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from Algorithm.BaseAlgorithm import BaseAlgorithm
from Algorithm.NSGA2 import NSGA2
from Utils.Logger import log_info, log_trace

from Algorithm.NN.Utils import extract_population_features
from Algorithm.NN.Second import SecondNN


class NTNSGA2(BaseAlgorithm):
    """
    NT-NSGA-II (NN2-only version)
    """

    EPS = 1e-9

    def __init__(self, vrp: Dict[str, Any], scorer: str, params: Dict[str, Any]):
        super().__init__(vrp=vrp, scorer=scorer)

        # ---- Base (reference) params ----
        self.base_population = int(params["population_size"])
        self.base_crossover_rate = float(params["crossover_rate"])
        self.base_mutation_rate = float(params["mutation_rate"])
        self.crossover_method = str(params.get("crossover_method", "cx"))
        self.mutation_method = str(params.get("mutation_method", "inversion"))
        self.base_iters = int(params.get("iters", 100))

        # ---- Training controls ----
        self.training_enabled = bool(params.get("training_enabled", False))
        self.epochs = int(params.get("epochs", 1))
        if not self.training_enabled:
            self.epochs = 1

        # ---- Training parameters ----
        self.batch_size_iters = int(params.get("batch_size_iters", 4))

        # ---- NN2 only ----
        self.second_nn = SecondNN(in_dim=5, hidden=int(
            params.get("second_hidden", 512)))
        lr_second = float(params.get("second_lr", 1e-3))
        self.opt_second = optim.Adam(self.second_nn.parameters(), lr=lr_second)

        # Batch storage - store features to recompute logps later
        self._batch_features: List[torch.Tensor] = []
        self._batch_logps: List[torch.Tensor] = []
        self._batch_rewards: List[torch.Tensor] = []

        self._log_cx: List = []
        self._log_mut: List = []

        # Reuse scorer callables for speed
        from Scorer.Distance import score_solution as sdist
        from Scorer.Cost import score_solution as scost
        self._sdist = sdist
        self._scost = scost

        log_info(
            "NTNSGA2 init (NN2-only): base_pop=%d, base_cx=%.3f, base_mut=%.3f, base_iters=%d, "
            "train=%s, epochs=%d, batch_size_iters=%d",
            self.base_population, self.base_crossover_rate, self.base_mutation_rate,
            self.base_iters, str(
                self.training_enabled), self.epochs, self.batch_size_iters
        )

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        self.start_run()

        # 1) Baseline reference
        log_info("[NTNSGA2] Baseline run starting: base_pop=%d base_iters=%d cx=%.4f mut=%.4f",
                 self.base_population, self.base_iters, self.base_crossover_rate, self.base_mutation_rate)
        baseline_best_score, baseline_metrics, baseline_runtime = self._run_baseline_nsga2()
        log_info("[NTNSGA2] Baseline run finished: best=%.6f runtime=%.3fs metrics_len=%d",
                 baseline_best_score, baseline_runtime, len(baseline_metrics))

        best_overall_perm: Optional[List[int]] = None
        best_overall_score = float("inf")

        for epoch in range(1, self.epochs + 1):
            # Clear batch storage at start of each epoch
            self._batch_features.clear()
            self._batch_rewards.clear()

            # --- initialize population ---
            pop_size = self.base_population
            iters_pred = self.base_iters
            population = [self._initialize_individual()
                          for _ in range(pop_size)]
            objectives = [self._evaluate_multi_objective(
                ind) for ind in population]

            fronts = self._fast_non_dominated_sort(objectives)
            self._crowding_distance_assignment(fronts, objectives)

            pareto_pop: List[List[int]] = []
            pareto_scores: List[Tuple[float, float, float]] = []
            self._update_pareto_archive(
                population, objectives, pareto_pop, pareto_scores)

            epoch_pred_t0 = time.perf_counter()

            # --- main iteration loop ---
            for it in range(1, iters_pred + 1):
                pop_feats = extract_population_features(objectives)
                cx_rate, mut_rate, logp_second, info_second = self.second_nn(
                    pop_feats)

                self._log_cx.append(cx_rate)
                self._log_mut.append(mut_rate)

                if self.training_enabled:
                    self._batch_features.append(pop_feats)
                    self._batch_logps.append(logp_second)

                log_trace("[NTNSGA2] Epoch %d Iter %d SECOND_NN: cx=%.6f mut=%.6f | logits=[%.3f, %.3f] logp=%.6f",
                          epoch, it, cx_rate, mut_rate,
                          float(info_second["logits"][0]),
                          float(info_second["logits"][1]),
                          float(logp_second))

                # Predicted step
                pred_next_pop, pred_next_obj = self._nsga2_one_step(
                    population, objectives, cx_rate, mut_rate
                )

                # Base step
                base_next_pop, base_next_obj = self._nsga2_one_step(
                    population, objectives, self.base_crossover_rate, self.base_mutation_rate
                )

                # Calculate reward
                pred_best = min(self._get_primary_score(o)
                                for o in pred_next_obj)
                base_best = min(self._get_primary_score(o)
                                for o in base_next_obj)
                pred_mean = sum(self._get_primary_score(o)
                                for o in pred_next_obj) / len(pred_next_obj)
                base_mean = sum(self._get_primary_score(o)
                                for o in base_next_obj) / len(base_next_obj)

                r_best = (base_best - pred_best) / (base_best)
                r_mean = (base_mean - pred_mean) / (base_mean)
                reward2 = torch.tensor(
                    (r_best), dtype=torch.float32)

                if self.training_enabled:
                    self._batch_rewards.append(reward2)

                log_trace(
                    "[NTNSGA2] Epoch %d Iter %d STEP: pred_best=%.6f base_best=%.6f pred_mean=%.6f base_mean=%.6f reward2=%.6f",
                    epoch, it, pred_best, base_best, pred_mean, base_mean, reward2
                )

                # BATCH LEARNING - Learn every batch_size_iters iterations
                if (self.training_enabled and
                        len(self._batch_rewards) >= self.batch_size_iters):
                    self._learn_batch()
                    # Clear batch after learning
                    self._batch_features.clear()
                    self._batch_logps.clear()
                    self._batch_rewards.clear()

                # Advance with predicted trajectory
                population, objectives = pred_next_pop, pred_next_obj
                fronts = self._fast_non_dominated_sort(objectives)
                self._crowding_distance_assignment(fronts, objectives)
                self._update_pareto_archive(
                    population, objectives, pareto_pop, pareto_scores)

            # Learn on remaining experiences at end of epoch
            if (self.training_enabled and
                    len(self._batch_rewards) > 0):
                self._learn_batch()
                self._batch_features.clear()
                self._batch_rewards.clear()

            epoch_pred_runtime = time.perf_counter() - epoch_pred_t0
            log_info("[NTNSGA2] Epoch %d predicted trajectory runtime: %.3fs predicted CX mean: %.3fs predicted Mut mean: %.3fs",
                     epoch, epoch_pred_runtime, sum(self._log_cx)/len(self._log_cx), sum(self._log_mut)/len(self._log_mut))

            self._log_cx.clear()
            self._log_mut.clear()

            # Pick epoch best from Pareto archive
            if pareto_pop:
                if self.scorer == "cost":
                    idx = min(range(len(pareto_scores)),
                              key=lambda i: pareto_scores[i][0])
                    best_score_epoch = pareto_scores[idx][0]
                elif self.scorer == "distance":
                    idx = min(range(len(pareto_scores)),
                              key=lambda i: pareto_scores[i][1])
                    best_score_epoch = pareto_scores[idx][1]
                else:
                    idx = min(range(len(pareto_scores)),
                              key=lambda i: pareto_scores[i][2])
                    best_score_epoch = pareto_scores[idx][2]
                best_perm_epoch = pareto_pop[idx]
            else:
                best_perm_epoch, best_score_epoch = None, float("inf")

            # Update global best
            if best_perm_epoch is not None and best_score_epoch < best_overall_score:
                best_overall_score = best_score_epoch
                best_overall_perm = best_perm_epoch

            log_info("[NTNSGA2] Epoch %d/%d SUMMARY: pop=%d iters=%d | best_epoch=%.6f best_overall=%.6f",
                     epoch, self.epochs, self.base_population, self.base_iters, best_score_epoch, best_overall_score)

        runtime_seconds = self.finalize()

        # Fallback to baseline if no improvement
        if best_overall_perm is None:
            log_info(
                "[NTNSGA2] No improvement found over epochs; returning baseline best.")
            nsga2 = NSGA2(self.vrp, self.scorer, dict(
                population_size=self.base_population,
                crossover_rate=self.base_crossover_rate,
                mutation_rate=self.base_mutation_rate,
                crossover_method=self.crossover_method,
                mutation_method=self.mutation_method,
            ))
            best_perm, best_score, _, _ = nsga2.solve(self.base_iters)
            self.update_global_best(best_perm, best_score)
            return best_perm, best_score, self.metrics, runtime_seconds

        self.update_global_best(best_overall_perm, best_overall_score)
        log_info("[NTNSGA2] DONE: best_overall=%.6f runtime=%.3fs",
                 best_overall_score, runtime_seconds)
        return best_overall_perm, best_overall_score, self.metrics, runtime_seconds

    def _learn_batch(self):
        """Simple policy gradient learning based on rewards"""

        # Convert to tensors
        rewards = torch.stack(self._batch_rewards)

        # RECOMPUTE log probabilities with the stored features
        logps = []
        for features in self._batch_features:
            # This creates a fresh computation graph
            _, _, logp_second, _ = self.second_nn(features)
            logps.append(logp_second)
        logps = torch.stack(logps)

        policy_loss = -(logps * rewards).mean()

        # before backward
        param_snapshot = [p.data.clone()
                          for p in self.second_nn.parameters() if p.requires_grad]

        self.opt_second.zero_grad()
        policy_loss.backward()

        # grad norm
        gn = 0.0
        for p in self.second_nn.parameters():
            if p.grad is not None:
                gn += p.grad.detach().norm().item() ** 2
        gn = gn ** 0.5
        log_trace("[NTNSGA2] grad_norm=%.6f loss=%.6f", gn, policy_loss.item())

        self.opt_second.step()

        # param delta
        delta = 0.0
        for p, old in zip(self.second_nn.parameters(), param_snapshot):
            delta += (p.data - old).norm().item() ** 2
        delta = delta ** 0.5
        log_trace("[NTNSGA2] param_delta=%.6f", delta)

        # log_info("[NTNSGA2] Batch update: batch_size=%d, mean_reward=%.4f",
        #          len(self._batch_rewards), rewards.mean().item())

    def _run_baseline_nsga2(self) -> Tuple[float, List[dict], float]:
        nsga2 = NSGA2(self.vrp, self.scorer, dict(
            population_size=self.base_population,
            crossover_rate=self.base_crossover_rate,
            mutation_rate=self.base_mutation_rate,
            crossover_method=self.crossover_method,
            mutation_method=self.mutation_method,
        ))
        best_perm, best_score, metrics, runtime = nsga2.solve(self.base_iters)
        self.update_global_best(best_perm, best_score)
        return best_score, metrics, runtime

    def _nsga2_one_step(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
        crossover_rate: float,
        mutation_rate: float,
    ) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:
        """
        Standard NSGA-II generation step:
        1. Generate offspring from parent population
        2. Combine parent + offspring
        3. Non-dominated sort + crowding distance
        4. Select best N individuals for next generation
        """
        n = len(population)

        # 1. Generate offspring through random selection + variation
        offspring: List[List[int]] = []
        while len(offspring) < n:
            p1_idx = self.rng.randrange(n)
            p2_idx = self.rng.randrange(n)

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            if self.rng.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            if self.rng.random() < mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        # 2. Evaluate offspring
        off_obj = [self._evaluate_multi_objective(ind) for ind in offspring]

        # 3. Combine parent + offspring (2N individuals)
        combined_population = population + offspring
        combined_objectives = objectives + off_obj

        # 4. Non-dominated sort on combined population
        fronts = self._fast_non_dominated_sort(combined_objectives)
        crowd = self._crowding_distance_assignment(
            fronts, combined_objectives, return_map=True
        )

        # 5. Select best N individuals for next generation
        new_population: List[List[int]] = []
        new_objectives: List[Tuple[float, float, float]] = []
        front_index = 0

        # Fill with complete fronts first
        while (front_index < len(fronts) and
               len(new_population) + len(fronts[front_index]) <= n):
            for idx in fronts[front_index]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])
            front_index += 1

        # Fill remaining slots with best individuals from next front
        # (sorted by crowding distance descending)
        if len(new_population) < n and front_index < len(fronts):
            remaining = n - len(new_population)
            current_front = list(fronts[front_index])
            current_front.sort(key=lambda idx: -crowd.get(idx, 0.0))

            for idx in current_front[:remaining]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

        return new_population, new_objectives

    def _initialize_individual(self) -> List[int]:
        individual = self.customers[:]
        self.rng.shuffle(individual)
        return individual

    def _evaluate_multi_objective(self, perm: List[int]) -> Tuple[float, float, float]:
        solution = self._solution_from_perm(perm)
        distance = self._sdist(solution)
        cost = self._scost(
            solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])

        total_time = 0.0
        T = self.vrp["T"]
        for route_data in solution:
            route = route_data["route"]
            for i in range(len(route) - 1):
                total_time += float(T[route[i], route[i + 1]])

        if not (math.isfinite(distance) and math.isfinite(cost) and math.isfinite(total_time)):
            return (float("inf"),) * 3
        if distance < 0.0 or cost < 0.0 or total_time < 0.0:
            return (float("inf"),) * 3
        return float(cost), float(distance), float(total_time)

    def _get_primary_score(self, objectives: Tuple[float, float, float]) -> float:
        if self.scorer == "cost":
            return objectives[0]
        elif self.scorer == "distance":
            return objectives[1]
        elif self.scorer == "time":
            return objectives[2]
        else:
            return objectives[0]

    def _dominates(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        better_in_any = False
        for i in range(3):
            if a[i] > b[i] + self.EPS:
                return False
            if a[i] < b[i] - self.EPS:
                better_in_any = True
        return better_in_any

    def _fast_non_dominated_sort(self, objectives: List[Tuple[float, float, float]]) -> List[List[int]]:
        n = len(objectives)
        S: List[List[int]] = [[] for _ in range(n)]
        n_count = [0] * n
        rank = [0] * n
        fronts: List[List[int]] = [[]]

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
            Q: List[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts[:-1]

    def _crowding_distance_assignment(
        self,
        fronts: List[List[int]],
        objectives: List[Tuple[float, float, float]],
        return_map: bool = False,
    ) -> Optional[Dict[int, float]]:
        crowd: Dict[int, float] = {}
        num_objectives = 3
        for front in fronts:
            if not front:
                continue
            for idx in front:
                crowd[idx] = 0.0
            for m in range(num_objectives):
                sorted_idx = sorted(front, key=lambda idx: objectives[idx][m])
                crowd[sorted_idx[0]] = float("inf")
                crowd[sorted_idx[-1]] = float("inf")
                if len(sorted_idx) > 2:
                    min_obj = objectives[sorted_idx[0]][m]
                    max_obj = objectives[sorted_idx[-1]][m]
                    denom = max_obj - min_obj
                    if denom > self.EPS:
                        for k in range(1, len(sorted_idx) - 1):
                            left = objectives[sorted_idx[k - 1]][m]
                            right = objectives[sorted_idx[k + 1]][m]
                            crowd[sorted_idx[k]] += (right - left) / denom
        if return_map:
            return crowd
        return None

    def _tournament_pick(self, i: int, j: int, rank: Dict[int, int], crowd: Dict[int, float]) -> int:
        ri, rj = rank.get(i, math.inf), rank.get(j, math.inf)
        if ri < rj:
            return i
        if rj < ri:
            return j
        ci = crowd.get(i)
        cj = crowd.get(j)
        if ci > cj:
            return i
        if cj > ci:
            return j
        return i if self.rng.random() < 0.5 else j

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        method = getattr(self, "crossover_method", "erx").lower()

        return self._edge_recombination_crossover(parent1, parent2)

    def _mutate(self, individual: List[int]) -> List[int]:
        mutated = individual[:]
        self._inversion_mutation(mutated)
        return mutated

    def _edge_recombination_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        adj: Dict[int, set] = {g: set() for g in p1}

        def add_edges(p: List[int]) -> None:
            for i in range(n):
                a = p[i]
                adj[a].add(p[(i - 1) % n])
                adj[a].add(p[(i + 1) % n])
        add_edges(p1)
        add_edges(p2)
        remaining = set(p1)
        current = self.rng.choice(p1)
        child: List[int] = []
        while remaining:
            child.append(current)
            remaining.remove(current)
            for s in adj.values():
                s.discard(current)
            neigh = [v for v in adj[current] if v in remaining]
            if neigh:
                min_deg = min(len(adj[v]) for v in neigh)
                candidates = [v for v in neigh if len(adj[v]) == min_deg]
                current = self.rng.choice(candidates)
            else:
                if not remaining:
                    break
                current = self.rng.choice(list(remaining))
        return child

    def _inversion_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        permutation[start: end +
                    1] = list(reversed(permutation[start: end + 1]))

    def _update_pareto_archive(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
        pareto_pop: List[List[int]],
        pareto_scores: List[Tuple[float, float, float]],
    ) -> None:
        cand_pop = pareto_pop + population
        cand_obj = pareto_scores + objectives
        new_pop: List[List[int]] = []
        new_obj: List[Tuple[float, float, float]] = []
        seen = set()
        for i, (ind, obj) in enumerate(zip(cand_pop, cand_obj)):
            dom = False
            for j, oth in enumerate(cand_obj):
                if i == j:
                    continue
                if self._dominates(oth, obj):
                    dom = True
                    break
            if not dom:
                key = tuple(ind)
                if key not in seen:
                    seen.add(key)
                    new_pop.append(ind)
                    new_obj.append(obj)
        pareto_pop[:] = new_pop
        pareto_scores[:] = new_obj
