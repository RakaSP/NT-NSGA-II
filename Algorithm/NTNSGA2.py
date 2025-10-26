# Algorithm/NTNSGA2.py
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from Algorithm.BaseAlgorithm import BaseAlgorithm
# used only as a baseline black box (unmodified)
from Algorithm.NSGA2 import NSGA2
from Utils.Logger import log_info, log_trace

from Algorithm.NN.Utils import extract_population_features
from Algorithm.NN.Second import SecondNN


class NTNSGA2(BaseAlgorithm):
    """
    NT-NSGA-II (NN2-only version)

    What this file does:
      • Disables/removes NN1 entirely.
      • Uses ONLY SecondNN to predict (crossover_rate, mutation_rate) per iteration.
      • At each iteration, from the SAME current population:
          - Run one NSGA-II step with BASE params (cx_base, mut_base)
          - Run one NSGA-II step with PREDICTED params (cx_pred, mut_pred)
        Compute reward r = 0.5 * [ (base_best - pred_best)/|base_best| + (base_mean - pred_mean)/|base_mean| ]
        Then ADVANCE the trajectory using the PREDICTED step only.
      • Keeps only CX crossover + inversion mutation in this file.
      • Baseline full run is done with external NSGA2 (unmodified).
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
        self.base_iters = int(params.get("iters", 500))

        # ---- Training controls ----
        self.training_enabled = bool(params.get("training_enabled", False))
        self.epochs = int(params.get("epochs", 1))
        if not self.training_enabled:
            # If not training, just run once
            self.epochs = 1

        # ---- NN2 only ----
        s_in = extract_population_features([], [], 0, 1).numel()
        self.second_nn = SecondNN(in_dim=s_in, hidden=int(params.get("second_hidden", 128)))
        lr_second = float(params.get("second_lr", 1e-3))
        self.opt_second = optim.Adam(self.second_nn.parameters(), lr=lr_second)

        # Training storage for NN2
        self._second_logps: List[torch.Tensor] = []
        self._second_rewards: List[float] = []

        # Reuse scorer callables for speed
        from Scorer.Distance import score_solution as sdist  # type: ignore
        from Scorer.Cost import score_solution as scost      # type: ignore
        self._sdist = sdist
        self._scost = scost

        log_info(
            "NTNSGA2 init (NN2-only): base_pop=%d, base_cx=%.3f, base_mut=%.3f, base_iters=%d, "
            "train=%s, epochs=%d",
            self.base_population, self.base_crossover_rate, self.base_mutation_rate,
            self.base_iters, str(self.training_enabled), self.epochs
        )

    # -----------------------------
    # Public entry
    # -----------------------------

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        """
        Entry point. `iters` is ignored; we use base_iters as the number of inner steps.
        """
        self.start_run()

        # 1) Baseline reference (single full run with NSGA-II, unmodified)
        log_info("[NTNSGA2] Baseline run starting: base_pop=%d base_iters=%d cx=%.4f mut=%.4f",
                 self.base_population, self.base_iters, self.base_crossover_rate, self.base_mutation_rate)
        baseline_best_score, baseline_metrics, baseline_runtime = self._run_baseline_nsga2()
        log_info("[NTNSGA2] Baseline run finished: best=%.6f runtime=%.3fs metrics_len=%d",
                 baseline_best_score, baseline_runtime, len(baseline_metrics))

        best_overall_perm: Optional[List[int]] = None
        best_overall_score = float("inf")

        for epoch in range(1, self.epochs + 1):
            self._second_logps.clear()
            self._second_rewards.clear()

            # --- initialize population using base population size ---
            pop_size = self.base_population
            iters_pred = self.base_iters  # number of per-iteration steps
            population = [self._initialize_individual() for _ in range(pop_size)]
            objectives = [self._evaluate_multi_objective(ind) for ind in population]

            # init fronts/crowding
            fronts = self._fast_non_dominated_sort(objectives)
            self._crowding_distance_assignment(fronts, objectives)

            # Pareto archive for the predicted trajectory (what we advance)
            pareto_pop: List[List[int]] = []
            pareto_scores: List[Tuple[float, float, float]] = []
            self._update_pareto_archive(population, objectives, pareto_pop, pareto_scores)

            epoch_pred_t0 = time.perf_counter()

            # --- iterate: predict (cx, mut) via NN2; compare base vs pred from SAME current pop ---
            for it in range(1, iters_pred + 1):
                pop_feats = extract_population_features(population, objectives, it - 1, iters_pred)
                cx_rate, mut_rate, logp_second, info_second = self.second_nn.sample_actions(pop_feats)

                if self.training_enabled:
                    self._second_logps.append(logp_second)

                log_trace("[NTNSGA2] Epoch %d Iter %d SECOND_NN: cx=%.6f mut=%.6f | logits=[%.3f, %.3f] logp=%.6f",
                          epoch, it, cx_rate, mut_rate,
                          float(info_second["logits"][0]),
                          float(info_second["logits"][1]),
                          float(logp_second))

                # Predicted step (from current population)
                pred_next_pop, pred_next_obj = self._nsga2_one_step(
                    population, objectives, cx_rate, mut_rate
                )

                # Base step (from the same current population)
                base_next_pop, base_next_obj = self._nsga2_one_step(
                    population, objectives, self.base_crossover_rate, self.base_mutation_rate
                )

                # Diagnostics and reward
                pred_best = min(self._get_primary_score(o) for o in pred_next_obj)
                base_best = min(self._get_primary_score(o) for o in base_next_obj)
                pred_mean = sum(self._get_primary_score(o) for o in pred_next_obj) / len(pred_next_obj)
                base_mean = sum(self._get_primary_score(o) for o in base_next_obj) / len(base_next_obj)

                eps = 1e-9
                r_best = (base_best - pred_best) / (abs(base_best) + eps)
                r_mean = (base_mean - pred_mean) / (abs(base_mean) + eps)
                reward2 = 0.5 * (r_best + r_mean)

                if self.training_enabled:
                    self._second_rewards.append(float(reward2))

                log_trace(
                    "[NTNSGA2] Epoch %d Iter %d STEP: pred_best=%.6f base_best=%.6f pred_mean=%.6f base_mean=%.6f reward2=%.6f",
                    epoch, it, pred_best, base_best, pred_mean, base_mean, reward2
                )

                # Advance ONLY with predicted trajectory
                population, objectives = pred_next_pop, pred_next_obj
                fronts = self._fast_non_dominated_sort(objectives)
                self._crowding_distance_assignment(fronts, objectives)
                self._update_pareto_archive(population, objectives, pareto_pop, pareto_scores)

            epoch_pred_runtime = time.perf_counter() - epoch_pred_t0
            log_info("[NTNSGA2] Epoch %d predicted trajectory runtime: %.3fs", epoch, epoch_pred_runtime)

            # pick epoch best from Pareto archive according to scorer
            if pareto_pop:
                if self.scorer == "cost":
                    idx = min(range(len(pareto_scores)), key=lambda i: pareto_scores[i][0])
                    best_score_epoch = pareto_scores[idx][0]
                elif self.scorer == "distance":
                    idx = min(range(len(pareto_scores)), key=lambda i: pareto_scores[i][1])
                    best_score_epoch = pareto_scores[idx][1]
                else:  # time or default
                    idx = min(range(len(pareto_scores)), key=lambda i: pareto_scores[i][2])
                    best_score_epoch = pareto_scores[idx][2]
                best_perm_epoch = pareto_pop[idx]
            else:
                best_perm_epoch, best_score_epoch = None, float("inf")

            # ---- Train ONLY NN2 (advantages) ----
            if self.training_enabled and self._second_logps and self._second_rewards:
                self.opt_second.zero_grad()
                advantages = self._calculate_advantages(self._second_rewards)
                second_loss = 0.0
                for logp_second, advantage in zip(self._second_logps, advantages):
                    second_loss += -(logp_second * torch.tensor(advantage, dtype=torch.float32))
                second_loss = second_loss / len(self._second_logps)
                log_info("[NTNSGA2] Epoch %d SECOND_NN train: loss=%.6f mean_adv=%.6f steps=%d",
                         epoch,
                         float(second_loss.detach().cpu()),
                         (sum(advantages)/len(advantages)) if advantages else 0.0,
                         len(self._second_logps))
                second_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.second_nn.parameters(), max_norm=1.0)
                self.opt_second.step()

            # Clear for next epoch
            self._second_logps.clear()
            self._second_rewards.clear()

            # Update global best trackers
            if best_perm_epoch is not None and best_score_epoch < best_overall_score:
                best_overall_score = best_score_epoch
                best_overall_perm = best_perm_epoch

            log_info("[NTNSGA2] Epoch %d/%d SUMMARY: pop=%d iters=%d | best_epoch=%.6f best_overall=%.6f",
                     epoch, self.epochs, self.base_population, self.base_iters, best_score_epoch, best_overall_score)

        runtime_seconds = self.finalize()

        # Fallback: if nothing better found, return baseline best (via black-box NSGA-II)
        if best_overall_perm is None:
            log_info("[NTNSGA2] No improvement found over epochs; returning baseline best.")
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
        log_info("[NTNSGA2] DONE: best_overall=%.6f runtime=%.3fs", best_overall_score, runtime_seconds)
        return best_overall_perm, best_overall_score, self.metrics, runtime_seconds

    # -----------------------------
    # Training Utilities
    # -----------------------------

    def _calculate_advantages(self, rewards: List[float]) -> List[float]:
        """Baseline-subtracted, normalized advantages for REINFORCE."""
        if not rewards:
            return []
        baseline = sum(rewards) / len(rewards)
        advantages = [r - baseline for r in rewards]
        if len(advantages) > 1:
            adv_std = (sum(a * a for a in advantages) / len(advantages)) ** 0.5
            if adv_std > 1e-8:
                advantages = [a / adv_std for a in advantages]
        return advantages

    # -----------------------------
    # Baseline (black-box NSGA-II)
    # -----------------------------

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

    # -----------------------------
    # One NSGA-II step (local copy)
    # -----------------------------

    def _nsga2_one_step(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
        crossover_rate: float,
        mutation_rate: float,
    ) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:

        # 1) fronts + crowding
        fronts = self._fast_non_dominated_sort(objectives)
        crowd = self._crowding_distance_assignment(fronts, objectives, return_map=True)

        # 2) offspring via binary tournaments on (rank, -crowding)
        rank_map: Dict[int, int] = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank_map[idx] = r

        offspring: List[List[int]] = []
        n = len(population)
        while len(offspring) < len(population):
            a, b = self.rng.randrange(n), self.rng.randrange(n)
            c, d = self.rng.randrange(n), self.rng.randrange(n)
            p1 = self._tournament_pick(a, b, rank_map, crowd)
            p2 = self._tournament_pick(c, d, rank_map, crowd)

            parent1 = population[p1]
            parent2 = population[p2]

            # crossover (CX only, probabilistic)
            if self.rng.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # mutation (inversion only, probabilistic)
            if self.rng.random() < mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        # 3) evaluate offspring
        off_obj = [self._evaluate_multi_objective(ind) for ind in offspring]

        # 4) survival selection (elitist)
        combined_population = population + offspring
        combined_objectives = objectives + off_obj
        fronts2 = self._fast_non_dominated_sort(combined_objectives)
        crowd2 = self._crowding_distance_assignment(fronts2, combined_objectives, return_map=True)

        new_population: List[List[int]] = []
        new_objectives: List[Tuple[float, float, float]] = []
        front_index = 0
        P = len(population)

        while front_index < len(fronts2) and len(new_population) + len(fronts2[front_index]) <= P:
            for idx in fronts2[front_index]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])
            front_index += 1

        if len(new_population) < P and front_index < len(fronts2):
            remaining = P - len(new_population)
            current_front = list(fronts2[front_index])
            current_front.sort(
                key=lambda idx: (-(crowd2.get(idx, 0.0)),
                                 self._get_primary_score(combined_objectives[idx]))
            )
            for idx in current_front[:remaining]:
                new_population.append(combined_population[idx])
                new_objectives.append(combined_objectives[idx])

        return new_population, new_objectives

    # -----------------------------
    # NSGA-II subroutines (local)
    # -----------------------------

    def _initialize_individual(self) -> List[int]:
        individual = self.customers[:]
        self.rng.shuffle(individual)
        return individual

    def _evaluate_multi_objective(self, perm: List[int]) -> Tuple[float, float, float]:
        try:
            self.check_constraints(perm)
        except Exception:
            return (float("inf"), float("inf"), float("inf"))

        solution = self._solution_from_perm(perm)
        distance = self._sdist(solution)
        cost = self._scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])

        # time from T
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
        ci = crowd.get(i, 0.0)
        cj = crowd.get(j, 0.0)
        if ci > cj:
            return i
        if cj > ci:
            return j
        return i if self.rng.random() < 0.5 else j

    # ---- Variation operators: CX crossover + inversion mutation ONLY ----

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Force cycle crossover (CX) only."""
        return self._cycle_crossover(parent1, parent2)

    def _mutate(self, individual: List[int]) -> List[int]:
        """Force inversion mutation only."""
        mutated = individual[:]
        self._inversion_mutation(mutated)
        return mutated

    def _cycle_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        child: List[Optional[int]] = [None] * n  # type: ignore
        pos2 = {v: i for i, v in enumerate(p2)}
        visited = [False] * n
        cycle_id = 0
        for start in range(n):
            if visited[start]:
                continue
            i = start
            cycle = []
            while not visited[i]:
                visited[i] = True
                cycle.append(i)
                i = pos2[p1[i]]
            parent = p1 if (cycle_id % 2 == 0) else p2
            for idx in cycle:
                child[idx] = parent[idx]
            cycle_id += 1
        # type: ignore: we ensured all positions filled
        return [int(x) for x in child]  # type: ignore

    def _inversion_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        permutation[start: end + 1] = list(reversed(permutation[start: end + 1]))

    # -----------------------------
    # Pareto archive helper (local)
    # -----------------------------

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
