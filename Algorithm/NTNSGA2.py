# Algorithm/NTNSGA2.py
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Poisson

from Algorithm.BaseAlgorithm import BaseAlgorithm
# used only as a baseline black box (unmodified)
from Algorithm.NSGA2 import NSGA2
from Utils.Logger import log_info, log_trace


# -----------------------------
# Utility helpers
# -----------------------------

def _flatten(lst_of_lsts: List[List[int]]) -> List[int]:
    return [x for lst in lst_of_lsts for x in lst]


# -----------------------------
# Feature Extraction
# -----------------------------

def extract_vrp_features(vrp: Dict[str, Any]) -> torch.Tensor:
    """
    Handcrafted global features from a VRP dict to feed FirstNN.
    Shape: [F1]
    """
    nodes = vrp["nodes"]
    vehicles = vrp["vehicles"]
    D = vrp["D"]
    T = vrp["T"]

    N = len(nodes)
    V = len(vehicles)

    demands = [float(getattr(n, "demand", 0.0)) for n in nodes]
    cap = [float(getattr(v, "max_capacity", 0.0)) for v in vehicles]

    mean_dem = float(sum(demands) / max(1, len(demands)))
    sq_dem = sum((d - mean_dem) ** 2 for d in demands)
    std_dem = float(math.sqrt(sq_dem / max(1, len(demands))))

    mean_cap = float(sum(cap) / max(1, len(cap)))
    sq_cap = sum((c - mean_cap) ** 2 for c in cap)
    std_cap = float(math.sqrt(sq_cap / max(1, len(cap))))

    # distance/time matrix stats (assume dense 2D arrays with numeric types)
    # skip diagonals
    dij = []
    tij = []
    nD0 = len(D)
    for i in range(nD0):
        for j in range(nD0):
            if i == j:
                continue
            dij.append(float(D[i, j]))
            tij.append(float(T[i, j]))
    if dij:
        d_mean = float(sum(dij) / len(dij))
        d_min = float(min(dij))
        d_max = float(max(dij))
        d_sq = sum((x - d_mean) ** 2 for x in dij)
        d_std = float(math.sqrt(d_sq / len(dij)))
    else:
        d_mean = d_min = d_max = d_std = 0.0

    if tij:
        t_mean = float(sum(tij) / len(tij))
        t_min = float(min(tij))
        t_max = float(max(tij))
        t_sq = sum((x - t_mean) ** 2 for x in tij)
        t_std = float(math.sqrt(t_sq / len(tij)))
    else:
        t_mean = t_min = t_max = t_std = 0.0

    feats = torch.tensor(
        [
            float(N),
            float(V),
            mean_dem,
            std_dem,
            mean_cap,
            std_cap,
            d_mean,
            d_std,
            d_min,
            d_max,
            t_mean,
            t_std,
            t_min,
            t_max,
        ],
        dtype=torch.float32,
    )
    return feats


def extract_population_features(
    population: List[List[int]],
    objectives: List[Tuple[float, float, float]],
    iter_idx: int,
    total_iters: int,
) -> torch.Tensor:
    """
    Per-iteration features for SecondNN.
    Includes basic population stats (size, iteration progress) and
    objective distribution stats (mean/std/min/max for each objective).
    Also includes a cheap diversity proxy (#unique genes at first position / pop).
    Shape: [F2]
    """
    P = len(population)
    progress = 0.0 if total_iters <= 1 else float(
        iter_idx) / float(total_iters - 1)

    if objectives:
        # per objective (cost, dist, time)
        cols = list(zip(*objectives))
        stats = []
        for col in cols:
            arr = list(col)
            mean = float(sum(arr) / len(arr))
            var = sum((x - mean) ** 2 for x in arr) / len(arr)
            std = float(math.sqrt(var))
            stats.extend([mean, std, float(min(arr)), float(max(arr))])
    else:
        stats = [0.0] * 12  # 3 objectives * 4 stats

    # cheap diversity: distinct head-genes proportion
    if P > 0 and len(population[0]) > 0:
        head = [ind[0] for ind in population]
        div_head_ratio = float(len(set(head))) / float(P)
    else:
        div_head_ratio = 0.0

    feats = torch.tensor(
        [float(P), progress, div_head_ratio] + stats,
        dtype=torch.float32,
    )
    return feats


# -----------------------------
# Neural Networks
# -----------------------------

class FirstNN(nn.Module):
    """
    Predicts integer (iters, population) with constant Poisson lambdas:
    - Network outputs probabilities p_pop, p_iter in (0,1) via sigmoid
    - Constant lambdas: lambda_pop=260, lambda_iter=1600 (so p=0.5 gives ~130 pop and ~800 iters)
    - Sample from Poisson and add offsets (+2 for pop, +1 for iters)
    """
    # CONSTANT lambdas - calculated to give desired K values at p=0.5
    LAMBDA_POP = 260.0    # Poisson(260) mean=260, p=0.5 -> sample~130
    LAMBDA_ITER = 1600.0  # Poisson(1600) mean=1600, p=0.5 -> sample~800

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.prob_pop_head = nn.Linear(hidden, 1)
        self.prob_iter_head = nn.Linear(hidden, 1)

        # Initialize to output ~0.5 probabilities
        nn.init.constant_(self.prob_pop_head.bias, 0.0)
        nn.init.constant_(self.prob_iter_head.bias, 0.0)

    def _forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        logit_pop = self.prob_pop_head(h).squeeze(-1)
        logit_iter = self.prob_iter_head(h).squeeze(-1)
        return h, logit_pop, logit_iter

    def forward_probs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, logit_pop, logit_iter = self._forward_raw(x)
        p_pop = torch.sigmoid(logit_pop)   # in (0,1)
        p_iter = torch.sigmoid(logit_iter)  # in (0,1)
        return {"p_pop": p_pop, "p_iter": p_iter, "logit_pop": logit_pop, "logit_iter": logit_iter}

    def sample_actions(self, x: torch.Tensor, min_pop: int = 2, min_iters: int = 1):
        probs = self.forward_probs(x)

        # Use probabilities as sampling weights with CONSTANT lambdas
        p_pop = probs["p_pop"]
        p_iter = probs["p_iter"]

        # Scale the Poisson samples by probabilities
        # p=1.0 -> full Poisson sample, p=0.0 -> sample near 0
        pop_sample = Poisson(self.LAMBDA_POP * p_pop).sample()
        iter_sample = Poisson(self.LAMBDA_ITER * p_iter).sample()

        # Convert to integers with offsets
        population_size = (pop_sample.to(torch.long) + min_pop).clamp(min=2)
        iters = (iter_sample.to(torch.long) + min_iters).clamp(min=1)

        # Log probabilities - use the actual distributions we sampled from
        log_prob_pop = Poisson(self.LAMBDA_POP * p_pop).log_prob(pop_sample)
        log_prob_iter = Poisson(
            self.LAMBDA_ITER * p_iter).log_prob(iter_sample)
        log_prob = log_prob_pop + log_prob_iter

        return (
            population_size.item(),
            iters.item(),
            log_prob,
            {
                "p_pop": probs["p_pop"].detach(),
                "p_iter": probs["p_iter"].detach(),
                "logit_pop": probs["logit_pop"].detach(),
                "logit_iter": probs["logit_iter"].detach(),
                "pop_sample": pop_sample.detach(),
                "iter_sample": iter_sample.detach(),
                "scaled_lambda_pop": (self.LAMBDA_POP * p_pop).detach(),
                "scaled_lambda_iter": (self.LAMBDA_ITER * p_iter).detach(),
            },
        )


class SecondNN(nn.Module):
    """
    Predicts *rates* (crossover_rate, mutation_rate) in (0,1) by architecture:
      - Head outputs logits -> sigmoid -> (0,1). Deterministic; used *as is*.
      - No exploration noise; logp = 0 (we only train FirstNN with REINFORCE per your instruction).
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.logits_head = nn.Linear(hidden, 2)  # [logit_cx, logit_mut]

    def rates(self, x: torch.Tensor) -> Tuple[float, float]:
        logits = self.logits_head(self.net(x))
        cx, mut = torch.sigmoid(logits[0]).item(
        ), torch.sigmoid(logits[1]).item()
        return cx, mut

    def sample_actions(self, x: torch.Tensor) -> Tuple[float, float, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.logits_head(self.net(x))
        rates = torch.sigmoid(logits)
        cx, mut = rates[0].item(), rates[1].item()
        logp = torch.tensor(0.0)  # deterministic
        info = {"logits": logits.detach(), "rates": rates.detach()}
        return cx, mut, logp, info


# -----------------------------
# NT-NSGA-II
# -----------------------------

class NTNSGA2(BaseAlgorithm):
    """
    Neural-Tuned NSGA-II (NT-NSGA-II):

    - FirstNN: chooses (iters, population_size) once per epoch from VRP features.
      Reward: improvement over baseline *minus* runtime penalty; trained with REINFORCE.

    - SecondNN: chooses (crossover_rate, mutation_rate) every iteration from population+objective features.
      Used deterministically *as predicted* (no sampling).

    Notes:
    * We do NOT modify Algorithm.NSGA2. It is used only for the baseline full run.
    * For per-iteration comparisons, we reimplement one NSGA-II step here (same logic).
    * No clamping: Poisson(+offset) for integers; Sigmoid for rates -> valid ranges by architecture.
    """

    EPS = 1e-9

    def __init__(self, vrp: Dict[str, Any], scorer: str, params: Dict[str, Any]):
        super().__init__(vrp=vrp, scorer=scorer)

        # ---- Base (reference) params ----
        self.base_population = int(params["population_size"])
        self.base_crossover_rate = float(params["crossover_rate"])
        self.base_mutation_rate = float(params["mutation_rate"])
        self.crossover_method = str(params["crossover_method"])
        self.mutation_method = str(params["mutation_method"])
        self.base_iters = int(params.get("iters", 500))

        # ---- Training controls ----
        self.training_enabled = bool(params.get("training_enabled", False))
        self.epochs = int(params.get("epochs", 1))
        if not self.training_enabled:
            self.epochs = 1

        # runtime penalty coefficient for FirstNN
        self.runtime_penalty = float(params.get("runtime_penalty", 0.1))

        # Neural nets + optimizers
        f_in = extract_vrp_features(vrp).numel()
        s_in = extract_population_features([], [], 0, 1).numel()

        self.first_nn = FirstNN(in_dim=f_in, hidden=int(
            params.get("first_hidden", 128)))
        self.second_nn = SecondNN(in_dim=s_in, hidden=int(
            params.get("second_hidden", 128)))

        lr_first = float(params.get("first_lr", 1e-3))
        lr_second = float(params.get("second_lr", 1e-3))
        self.opt_first = optim.Adam(self.first_nn.parameters(), lr=lr_first)
        # kept for future fine-tuning hooks
        self.opt_second = optim.Adam(self.second_nn.parameters(), lr=lr_second)

        # Reuse scorer callables for speed
        from Scorer.Distance import score_solution as sdist  # type: ignore
        from Scorer.Cost import score_solution as scost      # type: ignore
        self._sdist = sdist
        self._scost = scost

        log_info(
            "NTNSGA2 init: base_pop=%d, base_cx=%.3f, base_mut=%.3f, base_iters=%d, "
            "train=%s, epochs=%d",
            self.base_population, self.base_crossover_rate, self.base_mutation_rate,
            self.base_iters, str(self.training_enabled), self.epochs
        )

    # -----------------------------
    # Public entry
    # -----------------------------

    def solve(self, iters: int) -> Tuple[List[int], float, List[dict], float]:
        """
        Here `iters` is ignored for NT-NSGA-II (we let FirstNN decide per epoch).
        Kept only to match BaseAlgorithm signature.
        """
        self.start_run()

        # 1) Baseline reference (single full run with NSGA-II, unmodified)
        log_info("[NTNSGA2] Baseline run starting: base_pop=%d base_iters=%d cx=%.4f mut=%.4f",
                 self.base_population, self.base_iters, self.base_crossover_rate, self.base_mutation_rate)
        baseline_best_score, baseline_metrics, baseline_runtime = self._run_baseline_nsga2()
        log_info("[NTNSGA2] Baseline run finished: best=%.6f runtime=%.3fs metrics_len=%d",
                 baseline_best_score, baseline_runtime, len(baseline_metrics))

        # 2) Train / evaluate across epochs with parameter predictors
        best_overall_perm = None
        best_overall_score = float("inf")

        for epoch in range(1, self.epochs + 1):
            # containers for per-iteration diagnostics
            cx_hist: List[float] = []
            mut_hist: List[float] = []
            reward_hist: List[float] = []
            pred_best_hist: List[float] = []
            base_best_hist: List[float] = []
            pred_mean_hist: List[float] = []
            base_mean_hist: List[float] = []

            # ---- FIRST NN: pick iters, population from VRP features ----
            vrp_feats = extract_vrp_features(self.vrp)
            pop_pred, iters_pred, logp_first, info_first = self.first_nn.sample_actions(
                vrp_feats)

            # visibility of FirstNN internals (now includes raw logits + logp)
            try:
                log_info("[NTNSGA2] Epoch %d/%d FIRST_NN: pop_pred=%d iters_pred=%d",
                         epoch, self.epochs, pop_pred, iters_pred)
                log_info("[NTNSGA2] Epoch %d/%d FIRST_NN DETAIL: logits=[pop %.3f, iter %.3f] "
                         "probs=[pop %.4f, iter %.4f] lambdas=[pop %.2f, iter %.2f] logp=%.6f",
                         epoch, self.epochs,
                         float(info_first["logit_pop"]), float(
                             info_first["logit_iter"]),
                         float(info_first["p_pop"]), float(
                             info_first["p_iter"]),
                         float(info_first["lam_pop"]), float(
                             info_first["lam_iter"]),
                         float(logp_first.detach().cpu()))
                log_trace("[NTNSGA2] Epoch %d/%d FIRST_NN probs: p_pop=%.4f p_iter=%.4f | lambdas: lam_pop=%.2f lam_iter=%.2f",
                          epoch, self.epochs,
                          float(info_first["p_pop"]), float(
                              info_first["p_iter"]),
                          float(info_first["lam_pop"]), float(info_first["lam_iter"]))
            except Exception:
                pass

            # Generate a fresh random initial population of size pop_pred
            population = [self._initialize_individual()
                          for _ in range(pop_pred)]
            objectives = [self._evaluate_multi_objective(
                ind) for ind in population]
            # quick stats
            uniq = len(set(tuple(ind) for ind in population))
            cur_best = min(self._get_primary_score(o)
                           for o in objectives) if objectives else float("inf")
            cur_mean = (sum(self._get_primary_score(o) for o in objectives) /
                        len(objectives)) if objectives else float("inf")
            log_info("[NTNSGA2] Epoch %d init: pop=%d uniq=%d best=%.6f mean=%.6f",
                     epoch, len(population), uniq, cur_best, cur_mean)

            fronts = self._fast_non_dominated_sort(objectives)
            self._crowding_distance_assignment(fronts, objectives)

            # Track the epoch's best (predicted) like NSGA-II does
            pareto_pop = []
            pareto_scores = []
            self._update_pareto_archive(
                population, objectives, pareto_pop, pareto_scores)
            log_trace("[NTNSGA2] Epoch %d Pareto init: size=%d",
                      epoch, len(pareto_pop))

            # ---- SECOND NN: iterate, adapt (cx, mut) each step ----
            for it in range(1, iters_pred + 1):
                # Build SecondNN features from current population state
                pop_feats = extract_population_features(
                    population, objectives, it - 1, iters_pred)

                # SECOND NN: deterministic rates used as-is
                cx_rate, mut_rate, logp_second, info_second = self.second_nn.sample_actions(
                    pop_feats)
                log_trace("[NTNSGA2] Epoch %d Iter %d SECOND_NN: cx=%.6f mut=%.6f | logits=[%.3f, %.3f] logp=%.6f",
                          epoch, it, cx_rate, mut_rate,
                          float(info_second["logits"][0]), float(
                              info_second["logits"][1]),
                          float(logp_second))
                log_trace("[NTNSGA2] Epoch %d Iter %d SECOND_NN TRACE: logits=[%.3f, %.3f] rates=[%.6f, %.6f]",
                          epoch, it,
                          float(info_second["logits"][0]), float(
                              info_second["logits"][1]),
                          float(info_second["rates"][0]), float(info_second["rates"][1]))

                # ---- Predicted step ----
                pred_next_pop, pred_next_obj = self._nsga2_one_step(
                    population, objectives, cx_rate, mut_rate
                )

                # ---- Base step (same start pop) ----
                base_next_pop, base_next_obj = self._nsga2_one_step(
                    population, objectives, self.base_crossover_rate, self.base_mutation_rate
                )

                # ---- Reward for Second NN at this iteration (diagnostics only; deterministic policy) ----
                pred_best = min(self._get_primary_score(o)
                                for o in pred_next_obj)
                base_best = min(self._get_primary_score(o)
                                for o in base_next_obj)
                pred_mean = sum(self._get_primary_score(o)
                                for o in pred_next_obj) / len(pred_next_obj)
                base_mean = sum(self._get_primary_score(o)
                                for o in base_next_obj) / len(base_next_obj)

                # overlap diagnostics
                pred_keys = set(tuple(ind) for ind in pred_next_pop)
                base_keys = set(tuple(ind) for ind in base_next_pop)
                jacc = (len(pred_keys & base_keys) / max(1, len(pred_keys |
                        base_keys))) if (pred_keys or base_keys) else 1.0

                # Normalize by |base| to keep scales sane (still useful to log)
                norm = abs(base_best) + 1e-9
                reward2 = ((base_best - pred_best) / norm) + 0.25 * \
                    ((base_mean - pred_mean) / (abs(base_mean) + 1e-9))

                log_trace(
                    "[NTNSGA2] Epoch %d Iter %d STEP: pred_best=%.6f base_best=%.6f pred_mean=%.6f base_mean=%.6f "
                    "reward2=%.6f jaccard=%.3f",
                    epoch, it, pred_best, base_best, pred_mean, base_mean, reward2, jacc
                )

                # ---- store diagnostics ----
                cx_hist.append(cx_rate)
                mut_hist.append(mut_rate)
                reward_hist.append(float(reward2))
                pred_best_hist.append(float(pred_best))
                base_best_hist.append(float(base_best))
                pred_mean_hist.append(float(pred_mean))
                base_mean_hist.append(float(base_mean))

                # Move to predicted_next for the next iteration
                population, objectives = pred_next_pop, pred_next_obj
                fronts = self._fast_non_dominated_sort(objectives)
                self._crowding_distance_assignment(fronts, objectives)
                self._update_pareto_archive(
                    population, objectives, pareto_pop, pareto_scores)

                # log population health after moving
                uniq_after = len(set(tuple(ind) for ind in population))
                cur_best = min(self._get_primary_score(o)
                               for o in objectives) if objectives else float("inf")
                cur_mean = (sum(self._get_primary_score(
                    o) for o in objectives) / len(objectives)) if objectives else float("inf")
                log_trace("[NTNSGA2] Epoch %d Iter %d AFTER: pop=%d uniq=%d best=%.6f mean=%.6f pareto=%d",
                          epoch, it, len(population), uniq_after, cur_best, cur_mean, len(pareto_pop))

            # ----- End of epoch: evaluate epoch result -----
            # pick best from pareto by current scorer
            if pareto_pop:
                if self.scorer == "cost":
                    idx = min(range(len(pareto_scores)),
                              key=lambda i: pareto_scores[i][0])
                    best_score_epoch = pareto_scores[idx][0]
                elif self.scorer == "distance":
                    idx = min(range(len(pareto_scores)),
                              key=lambda i: pareto_scores[i][1])
                    best_score_epoch = pareto_scores[idx][1]
                else:  # time or default
                    idx = min(range(len(pareto_scores)),
                              key=lambda i: pareto_scores[i][2])
                    best_score_epoch = pareto_scores[idx][2]
                best_perm_epoch = pareto_pop[idx]
            else:
                best_perm_epoch, best_score_epoch = None, float("inf")

            log_info("[NTNSGA2] Epoch %d END: pareto_size=%d best_epoch=%.6f",
                     epoch, len(pareto_pop), best_score_epoch)

            # ---- per-epoch means of stored diagnostics ----
            def _m(v: List[float]) -> float:
                return float(sum(v) / len(v)) if v else float("nan")

            log_info("[NTNSGA2] Epoch %d MEANS: cx=%.6f mut=%.6f reward=%.6f "
                     "pred_best=%.6f base_best=%.6f pred_mean=%.6f base_mean=%.6f",
                     epoch, _m(cx_hist), _m(mut_hist), _m(reward_hist),
                     _m(pred_best_hist), _m(base_best_hist), _m(pred_mean_hist), _m(base_mean_hist))

            # ---- FIRST NN reward (isolated; use base rates to remove SecondNN confound) ----
            if pop_pred > 1:
                pop_shadow = [self._initialize_individual()
                              for _ in range(pop_pred)]
                obj_shadow = [self._evaluate_multi_objective(
                    ind) for ind in pop_shadow]
                for it in range(iters_pred):
                    pop_shadow, obj_shadow = self._nsga2_one_step(
                        pop_shadow, obj_shadow, self.base_crossover_rate, self.base_mutation_rate
                    )
                shadow_best = min(self._get_primary_score(o)
                                  for o in obj_shadow)
            else:
                shadow_best = float("inf")

            normB = abs(baseline_best_score) + 1e-9
            imp = (baseline_best_score - shadow_best) / normB

            # runtime penalty ~ relative compute vs baseline
            rel_compute = (pop_pred * iters_pred) / \
                float(self.base_population * self.base_iters)
            penalty = self.runtime_penalty * max(0.0, rel_compute - 1.0)
            reward1 = imp - penalty

            if self.training_enabled:
                self.opt_first.zero_grad()
                loss1 = -(logp_first *
                          torch.tensor(reward1, dtype=torch.float32))
                log_info("[NTNSGA2] Epoch %d FIRST_NN train: loss=%.6f logp=%.6f imp=%.6f penalty=%.6f reward1=%.6f "
                         "shadow_best=%.6f baseline_best=%.6f rel_compute=%.3f",
                         epoch, float(loss1.detach().cpu()), float(
                             logp_first.detach().cpu()),
                         imp, penalty, reward1, shadow_best, baseline_best_score, rel_compute)
                loss1.backward()
                self.opt_first.step()
            else:
                log_info("[NTNSGA2] Epoch %d FIRST_NN eval: imp=%.6f penalty=%.6f reward1=%.6f "
                         "shadow_best=%.6f baseline_best=%.6f rel_compute=%.3f",
                         epoch, imp, penalty, reward1, shadow_best, baseline_best_score, rel_compute)

            # Update global best trackers
            if best_perm_epoch is not None and best_score_epoch < best_overall_score:
                best_overall_score = best_score_epoch
                best_overall_perm = best_perm_epoch

            log_info(
                "[NTNSGA2] Epoch %d/%d SUMMARY: pop=%d iters=%d | best_epoch=%.6f best_overall=%.6f",
                epoch, self.epochs, pop_pred, iters_pred, best_score_epoch, best_overall_score
            )

        runtime_seconds = self.finalize()

        # Fallback: if nothing better found, return baseline best (via black-box NSGA-II)
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
    # One NSGA-II step (same logic, local copy)
    # -----------------------------

    def _nsga2_one_step(
        self,
        population: List[List[int]],
        objectives: List[Tuple[float, float, float]],
        crossover_rate: float,
        mutation_rate: float,
    ) -> Tuple[List[List[int]], List[Tuple[float, float, float]]]:

        # quick pre-step stats
        if objectives:
            pre_best = min(self._get_primary_score(o) for o in objectives)
            pre_mean = sum(self._get_primary_score(o)
                           for o in objectives) / len(objectives)
        else:
            pre_best = float("inf")
            pre_mean = float("inf")
        log_trace("[NTNSGA2] one_step: start pop=%d cx=%.4f mut=%.4f | best=%.6f mean=%.6f",
                  len(population), crossover_rate, mutation_rate, pre_best, pre_mean)

        # 1) fronts + crowding
        fronts = self._fast_non_dominated_sort(objectives)
        crowd = self._crowding_distance_assignment(
            fronts, objectives, return_map=True)
        log_trace("[NTNSGA2] one_step: fronts=%s",
                  str([len(f) for f in fronts]))

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

            # crossover
            if self.rng.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # mutation
            if self.rng.random() < mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        # 3) evaluate offspring
        off_obj = [self._evaluate_multi_objective(ind) for ind in offspring]
        off_best = min(self._get_primary_score(o) for o in off_obj)
        off_mean = sum(self._get_primary_score(o)
                       for o in off_obj) / len(off_obj)
        log_trace("[NTNSGA2] one_step: offspring_ready size=%d | best=%.6f mean=%.6f",
                  len(offspring), off_best, off_mean)

        # 4) survival selection (elitist)
        combined_population = population + offspring
        combined_objectives = objectives + off_obj
        fronts2 = self._fast_non_dominated_sort(combined_objectives)
        crowd2 = self._crowding_distance_assignment(
            fronts2, combined_objectives, return_map=True)
        log_trace("[NTNSGA2] one_step: combined fronts=%s",
                  str([len(f) for f in fronts2]))

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

        # post-step stats & stagnation flag
        if new_objectives:
            post_best = min(self._get_primary_score(o) for o in new_objectives)
            post_mean = sum(self._get_primary_score(o)
                            for o in new_objectives) / len(new_objectives)
        else:
            post_best = float("inf")
            post_mean = float("inf")

        same = (set(tuple(ind) for ind in new_population)
                == set(tuple(ind) for ind in population))
        log_trace("[NTNSGA2] one_step: new_pop=%d | best=%.6f mean=%.6f | no_change=%s",
                  len(new_population), post_best, post_mean, str(same))

        return new_population, new_objectives

    # -----------------------------
    # NSGA-II subroutines (copied logic; not modifying original class)
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
        cost = self._scost(
            solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"])

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
            log_trace(
                "[NTNSGA2] crowding: computed for %d individuals", len(crowd))
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

    # ---- Variation operators (same logic as your NSGA-II; pasted here) ----

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

    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child: List[Optional[int]] = [None] * n
        child[a:b+1] = p1[a:b+1]
        used = set(p1[a:b+1])
        fill_idx = 0
        for i in range(n):
            if child[i] is None:
                while p2[fill_idx] in used:
                    fill_idx += 1
                child[i] = p2[fill_idx]
                used.add(p2[fill_idx])
                fill_idx += 1
        return [int(x) for x in child]  # type: ignore

    def _partially_mapped_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child: List[Optional[int]] = [None] * n
        child[a:b+1] = p1[a:b+1]
        mapping: Dict[int, int] = {}
        for i in range(a, b+1):
            mapping[p2[i]] = p1[i]
        in_child = set(p1[a:b+1])
        for i in range(n):
            if child[i] is not None:
                continue
            val = p2[i]
            while val in mapping and val in in_child:
                val = mapping[val]
            child[i] = val
            in_child.add(val)
        if any(v is None for v in child):
            used = set(v for v in child if v is not None)
            tail = [v for v in p2 if v not in used]
            for i in range(n):
                if child[i] is None:
                    child[i] = tail.pop(0)
        return [int(x) for x in child]  # type: ignore

    def _cycle_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        child: List[Optional[int]] = [None] * n
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
        return [int(x) for x in child]  # type: ignore

    def _edge_recombination_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        edge_table: Dict[int, set] = {}
        for i in range(n):
            a = p1[i]
            b = p2[i]
            edge_table.setdefault(a, set()).update(
                [p1[(i - 1) % n], p1[(i + 1) % n]])
            edge_table.setdefault(b, set()).update(
                [p2[(i - 1) % n], p2[(i + 1) % n]])
        child: List[int] = []
        current = self.rng.choice(p1)
        while len(child) < n:
            child.append(current)
            for neighbors in edge_table.values():
                neighbors.discard(current)
            neighbors = edge_table.get(current, set())
            candidates = [v for v in neighbors if v not in child]
            if candidates:
                min_deg = min(len(edge_table.get(v, ())) for v in candidates)
                ties = [v for v in candidates if len(
                    edge_table.get(v, ())) == min_deg]
                current = self.rng.choice(ties)
            else:
                remaining = [node for node in p1 if node not in child]
                if remaining:
                    current = self.rng.choice(remaining)
                else:
                    break
        if len(child) != n:
            missing = [g for g in p1 if g not in child]
            child.extend(missing)
        return child

    def _swap_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        i, j = self.rng.sample(range(len(permutation)), 2)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    def _inversion_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        permutation[start: end +
                    1] = list(reversed(permutation[start: end + 1]))

    def _scramble_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        start, end = sorted(self.rng.sample(range(len(permutation)), 2))
        segment = permutation[start: end + 1]
        self.rng.shuffle(segment)
        permutation[start: end + 1] = segment

    def _displacement_mutation(self, permutation: List[int]) -> None:
        if len(permutation) < 2:
            return
        n = len(permutation)
        start, end = sorted(self.rng.sample(range(n), 2))
        segment = permutation[start: end + 1]
        remaining = permutation[:start] + permutation[end + 1:]
        insert_pos = self.rng.randrange(len(remaining) + 1)
        permutation.clear()
        permutation.extend(remaining[:insert_pos])
        permutation.extend(segment)
        permutation.extend(remaining[insert_pos:])

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
        log_trace("[NTNSGA2] pareto_update: size=%d", len(pareto_pop))
