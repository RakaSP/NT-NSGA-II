from __future__ import annotations

import json
import os
import time
import threading
import math
from typing import Any, Dict, List, Optional, Mapping, Tuple
import numpy as np

import pandas as pd
import torch as T

from Algorithm.Gym.EAControlEnv import EAControlEnv
from Algorithm.NN.Second import SecondNN
from Utils.Logger import set_log_level, log_info, log_trace
from vrp_core import (
    decode_split_equal,
    decode_minimize,
    write_routes_json,
    write_summary_csv,
    write_metadata_json,
    ensure_dir,
)


# ==============================
# Engine
# ==============================
class VRPSolverEngine:
    # Keep other defaults if you want; but TIME LIMIT is NOT allowed to default.
    DEFAULT_MAX_ITERS = 99999999999999

    def __init__(self, config: Dict[str, Any]):
        self.config = config  # <-- MINIMAL: keep config for metadata output

        set_log_level(str(config.get("logger", "INFO")).upper())

        self.output_dir = config.get("output_dir", "results")
        ensure_dir(self.output_dir)
        self.routes_name = config.get("routes_out", "solution_routes.json")
        self.summary_name = config.get("summary_out", "solution_summary.csv")
        self.decode_mode = str(config.get("decode_mode", "minimize")).lower()

        self.algo_name = str(config.get("algorithm")).lower()
        self.scorer_name = str(config.get("scorer")).lower()
        self.params: Dict[str, Any] = config["algo_params"][self.algo_name]

        self.iters: int = int(self.DEFAULT_MAX_ITERS)

        # --------------------------
        # REQUIRED: time_limit from config.yaml
        # --------------------------
        if "time_limit" not in config:
            raise KeyError("Missing required config key: 'time_limit' (seconds)")
        try:
            self.time_limit: float = float(config["time_limit"])
        except (TypeError, ValueError) as e:
            raise ValueError("Config 'time_limit' must be a number (seconds)") from e
        if not math.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("Config 'time_limit' must be finite and > 0 (seconds)")

        self.epochs: int = int(config.get("epochs", 1))
        self.batch_size: int = int(config.get("batch_size", 1))

        self.rl_enabled: bool = bool(config.get("rl_enabled"))
        self.training_enabled: bool = bool(config.get("training_enabled", True))

        self.secondNN_checkpoint_path: Optional[str] = config.get("second_nn_path")

        # Now store lists for multiple clusters
        self.vrps: Optional[List[Dict[str, Any]]] = None
        self.algos: Optional[List[Any]] = None
        self.second_nn: Optional[SecondNN] = None
        self.opt_second: Optional[T.optim.Optimizer] = None

        self.entropy_coef = float(config.get("entropy_coef", 1e-3))
        self.std_reg_coef = float(config.get("std_reg_coef", 0.0))
        self.std_target = float(config.get("std_target", 0.25))
        self.adv_clip = float(config.get("adv_clip", 5.0))

    # ==============================
    # Factories
    # ==============================
    def _make_scorer(self):
        if self.scorer_name == "cost":
            from vrp_core import score_cost as score_from_perm
        elif self.scorer_name == "distance":
            from vrp_core import score_distance as score_from_perm
        else:
            raise ValueError("Unknown scorer: use 'cost' or 'distance'")
        return score_from_perm

    def _make_algorithm(self):
        if self.algo_name == "ga":
            from Algorithm.GA import GA as Algo
        elif self.algo_name == "pso":
            from Algorithm.PSO import PSO as Algo
        elif self.algo_name == "aco":
            from Algorithm.ACO import ACO as Algo
        elif self.algo_name == "nsga2":
            from Algorithm.NSGA2 import NSGA2 as Algo
        else:
            raise ValueError("Unknown algorithm: use 'ga', 'pso', 'aco', or 'nsga2'")
        return Algo

    # ==============================
    # Setup
    # ==============================
    def prepare(self, vrps: List[Dict[str, Any]]) -> None:
        """
        Prepare the engine for ALL VRP clusters at once.
        Creates one algorithm instance per cluster.
        """
        self.vrps = vrps
        self.algos = []
        Algo = self._make_algorithm()

        for vrp in vrps:
            algo = Algo(vrp=vrp, scorer=self.scorer_name, params=self.params, seed=vrp["clustering_seed"])
            self.algos.append(algo)

        log_info("Prepared %d clusters for processing", len(vrps))

    def build_agent(self, in_dim: int = 10, hidden: int = 512, lr: float = 3e-4) -> None:
        """
        Build (or load) the Gaussian policy ('second_nn') and its optimizer.
        """
        def _new_model() -> SecondNN:
            return SecondNN(in_dim=in_dim, hidden=hidden)

        ckpt_path = self.secondNN_checkpoint_path
        if ckpt_path:
            candidate = ckpt_path if os.path.isfile(ckpt_path) else os.path.join(ckpt_path, "secondNN.pt")
            if os.path.isfile(candidate):
                try:
                    state = T.load(candidate, map_location="cpu")
                    model = _new_model()
                    model.load_state_dict(state, strict=True)
                    self.second_nn = model
                    log_info("Loaded SecondNN weights from: %s", candidate)
                except Exception as e:
                    log_info("SecondNN load failed (%s). Using fresh model.", str(e))
                    raise
            else:
                log_info("SecondNN checkpoint not found at '%s'. Using fresh model.", candidate)
                self.second_nn = _new_model()
        else:
            self.second_nn = _new_model()
            log_info("No SecondNN checkpoint provided. Using fresh model (in_dim=%d, hidden=%d).", in_dim, hidden)

        self.opt_second = T.optim.Adam(self.second_nn.parameters(), lr=lr)

    # ==============================
    # Route totals (distance/time/cost)
    # ==============================
    @staticmethod
    def _finite(x: float, default: float = 0.0) -> float:
        try:
            x = float(x)
        except Exception:
            return float(default)
        return x if math.isfinite(x) else float(default)

    def _totals_from_routes(self, routes: List[List[int]], cluster_idx: int) -> Tuple[float, float, float]:
        """
        Compute total_distance, total_time, total_cost from decoded routes and vehicle costs.
        This is REPORTING, independent of what the optimizer is minimizing.
        """
        assert self.vrps is not None
        vrp = self.vrps[cluster_idx]

        D: Mapping[int, Mapping[int, float]] = vrp["D"]
        Tm: Optional[Mapping[int, Mapping[int, float]]] = vrp.get("T", None)
        vehicles = vrp["vehicles"]

        total_distance = 0.0
        total_time = 0.0
        total_cost = 0.0

        for r, veh in zip(routes, vehicles):
            dist_m = 0.0
            time_s = 0.0

            # sum legs
            for a, b in zip(r[:-1], r[1:]):
                try:
                    dist_m += float(D[int(a)][int(b)])
                except Exception:
                    dist_m += 0.0

                if Tm is not None:
                    try:
                        time_s += float(Tm[int(a)][int(b)])
                    except Exception:
                        time_s += 0.0

            ic = self._finite(getattr(veh, "initial_cost", 0.0), 0.0)
            dc = self._finite(getattr(veh, "distance_cost", 0.0), 0.0)

            total_distance += dist_m
            total_time += time_s
            total_cost += (ic + dc * dist_m)

        return (
            self._finite(total_distance, 0.0),
            self._finite(total_time, 0.0),
            self._finite(total_cost, 0.0),
        )

    # ==============================
    # RL: Update step
    # ==============================
    def update_second_nn(self, batch_obs: List[T.Tensor], batch_rewards: List[T.Tensor]) -> None:
        """REINFORCE with centered advantages, entropy bonus, and optional std regularization."""
        assert self.second_nn is not None and self.opt_second is not None

        rewards = T.stack(batch_rewards)
        adv = rewards - rewards.mean()
        if rewards.numel() > 1:
            std = rewards.std()
            if float(std.item()) > 1e-8:
                adv = adv / (std + 1e-8)

        if self.adv_clip is not None and self.adv_clip > 0:
            adv = adv.clamp(min=-self.adv_clip, max=self.adv_clip)

        logps, entropies = [], []
        for features in batch_obs:
            _, _, logp_second, info = self.second_nn(features)
            logps.append(logp_second)
            entropies.append(info["entropy_u"])

        logps = T.stack(logps).squeeze()
        entropies = T.stack(entropies).squeeze()

        policy_loss = -(logps * adv.detach()).mean()
        if self.entropy_coef > 0:
            policy_loss = policy_loss - self.entropy_coef * entropies.mean()

        if self.std_reg_coef > 0:
            std_cx = self.second_nn.log_std_cx.exp()
            std_mut = self.second_nn.log_std_mut.exp()
            std_reg = ((std_cx - self.std_target) ** 2 + (std_mut - self.std_target) ** 2)
            policy_loss = policy_loss + self.std_reg_coef * std_reg

        param_snapshot = [p.data.clone() for p in self.second_nn.parameters() if p.requires_grad]

        self.opt_second.zero_grad()
        policy_loss.backward()
        T.nn.utils.clip_grad_norm_(self.second_nn.parameters(), max_norm=1.0)
        self.opt_second.step()

        gn = (sum((p.grad.detach().norm().item() ** 2 for p in self.second_nn.parameters() if p.grad is not None))) ** 0.5
        delta = (sum(((p.data - old).norm().item() ** 2) for p, old in zip(self.second_nn.parameters(), param_snapshot))) ** 0.5
        log_trace(
            "[NTNSGA2] Batch update: size=%d loss=%.6f grad_norm=%.6f param_delta=%.6f mean_adv=%.6f std_adv=%.6f",
            len(batch_rewards),
            float(policy_loss.item()),
            gn,
            delta,
            float(adv.mean().item()),
            float(adv.std().item()) if adv.numel() > 1 else 0.0,
        )

    # ==============================
    # Run (public)
    # ==============================
    def run(self) -> Dict[str, Any]:
        assert self.vrps is not None and self.algos is not None
        if self.rl_enabled:
            return self._run_rl()
        else:
            return self._run_non_rl()

    # ==============================
    # RL path – Process all clusters epoch by epoch
    # ==============================
    def _run_rl(self) -> Dict[str, Any]:
        envs = [EAControlEnv(algo) for algo in self.algos]
        for env in envs:
            env.reset()
            if hasattr(env.nsga2, "start_run"):
                env.nsga2.start_run()

        self.build_agent(13)

        best_solution = float("inf")

        if not self.training_enabled:
            self.epochs = 1
            self.batch_size = 1

        total_solving_time_s = 0.0

        for curr_epoch in range(self.epochs):
            epoch_start = time.time()
            log_info(f"Starting epoch {curr_epoch} for all {len(envs)} clusters")

            cluster_solving_times = []

            for cluster_idx, env in enumerate(envs):
                log_info(f"[Epoch {curr_epoch}] Processing cluster {cluster_idx}")

                obs = env.reset()
                curr_iter = 0
                epoch_cx, epoch_mut = [], []
                solving_start = time.time()
                
                # Clear metrics at the start of each cluster
                env.nsga2.metrics = []

                while True:
                    if time.time() - solving_start >= self.time_limit:
                        log_info(f"[Epoch {curr_epoch}] Cluster {cluster_idx} time limit reached")
                        break

                    obs_memory: List[T.Tensor] = []
                    obs_reward: List[T.Tensor] = []

                    for _ in range(self.batch_size):
                        if time.time() - solving_start >= self.time_limit:
                            break

                        action = self.second_nn(obs)
                        obs, reward = env.step(action)

                        epoch_cx.append(float(env.nsga2.crossover_rate))
                        epoch_mut.append(float(env.nsga2.mutation_rate))

                        obs_memory.append(obs)
                        obs_reward.append(T.as_tensor(reward, dtype=T.float32))
                        curr_iter += 1
                        
                        # Calculate route time for current best permutation
                        primary_scores = [env.nsga2._get_primary_score(obj) for obj in env.nsga2.objectives]

                        route_times = []
                        for perm in env.nsga2.population:
                            details = env.nsga2._compute_details(perm)
                            total_time = details.get("summary", {}).get("total_time", float("inf"))
                            route_times.append(total_time)

                        best_score = float(np.min(primary_scores)) if primary_scores else env.nsga2.best_score
                        mean_score = float(np.mean(primary_scores)) if primary_scores else env.nsga2.best_score
                        best_route_time = float(np.min(route_times)) if route_times else float("inf")
                        mean_route_time = float(np.mean(route_times)) if route_times else float("inf")

                        env.nsga2.metrics.append({
                            'iter': curr_iter,
                            'best': best_score,
                            'mean': mean_score,
                            'best_route_time': best_route_time,
                            'mean_route_time': mean_route_time
                        })

                    if self.training_enabled and obs_memory:
                        self.update_second_nn(obs_memory, obs_reward)

                    if time.time() - solving_start >= self.time_limit:
                        break

                solving_time = time.time() - solving_start
                cluster_solving_times.append(solving_time)
                total_solving_time_s += solving_time

                if self.training_enabled:
                    output_dir_cluster = os.path.join(self.output_dir, f"epoch_{curr_epoch}", f"cluster_{cluster_idx}")
                else:
                    output_dir_cluster = os.path.join(self.output_dir, f"cluster_{cluster_idx}")

                ensure_dir(output_dir_cluster)

                if env.nsga2.best_score < best_solution:
                    best_solution = env.nsga2.best_score

                perm_e = env.nsga2.best_perm
                routes_e = self._decode_for_cluster(perm_e, cluster_idx)

                # SAVE (NOT included in solving_time)
                self._save_routes_and_summary(output_dir_cluster, routes_e, cluster_idx)
                self._save_metrics_csv(output_dir_cluster, env.nsga2.metrics, suffix=self.algo_name)

                # MINIMAL: write metadata
                write_metadata_json(
                    output_dir_cluster,
                    "metadata.json",
                    config=self.config,
                    vrp=self.vrps[cluster_idx],
                    runtime_s=solving_time,
                    iterations=curr_iter,
                    algorithm=self.algo_name,
                    scorer=self.scorer_name,
                    algo_params=self.params,
                    cluster_id=cluster_idx,
                    epoch=curr_epoch,
                )

                # REPORT totals from routes (NOT NaN ever)
                total_distance_rt, total_time_rt, total_cost_rt = self._totals_from_routes(routes_e, cluster_idx)

                best_info = getattr(env.nsga2, "best_info", {}) or {}
                summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
                score_e = float(summary_info.get("score", env.nsga2.best_score))

                # distance value depends on scorer; cost/time always reported
                if self.scorer_name == "distance":
                    final_distance_e = self._finite(score_e, total_distance_rt)
                    final_cost_e = total_cost_rt
                elif self.scorer_name == "cost":
                    final_cost_e = self._finite(score_e, total_cost_rt)
                    final_distance_e = total_distance_rt
                else:
                    final_distance_e = total_distance_rt
                    final_cost_e = total_cost_rt

                cluster_summary = {
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "cluster_id": cluster_idx,
                    "final_distance": float(final_distance_e),
                    "final_cost": float(final_cost_e),
                    "final_time": float(total_time_rt),
                    "score": float(score_e),
                    "solving_time_s": float(solving_time),
                    "iterations": int(curr_iter),
                    "epoch": int(curr_epoch),
                }
                self._write_json(os.path.join(output_dir_cluster, "cluster_summary.json"), cluster_summary)

                if epoch_cx and epoch_mut:
                    mean_cx = sum(epoch_cx) / len(epoch_cx)
                    mean_mut = sum(epoch_mut) / len(epoch_mut)
                else:
                    mean_cx = mean_mut = float("nan")

                log_info(
                    f"[Epoch {curr_epoch}] Cluster {cluster_idx} complete: "
                    f"mean_cx={mean_cx:.4f} mean_mut={mean_mut:.4f} "
                    f"iterations={curr_iter} solving_time={solving_time:.2f}s score={score_e:.6f}"
                )

            if self.training_enabled:
                epoch_dir = os.path.join(self.output_dir, f"epoch_{curr_epoch}")
                ensure_dir(epoch_dir)
                # Save checkpoint (NOT included in solving time)
                T.save(self.second_nn.state_dict(), os.path.join(epoch_dir, "secondNN.pt"))
                log_info(f"Saved checkpoint for epoch {curr_epoch}")

                epoch_cluster_solving_time = sum(cluster_solving_times)

                epoch_agg = {
                    "epoch": curr_epoch,
                    "clusters": len(envs),
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "sum_final_distance": 0.0,
                    "sum_final_cost": 0.0,
                    "sum_final_time": 0.0,
                    "epoch_runtime_s": float(time.time() - epoch_start),
                    "total_cluster_solving_time_s": float(epoch_cluster_solving_time),
                }

                for cluster_idx, env in enumerate(envs):
                    perm_e = env.nsga2.best_perm
                    routes_e = self._decode_for_cluster(perm_e, cluster_idx)
                    d_rt, t_rt, c_rt = self._totals_from_routes(routes_e, cluster_idx)

                    best_info = getattr(env.nsga2, "best_info", {}) or {}
                    summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
                    score_e = float(summary_info.get("score", env.nsga2.best_score))

                    if self.scorer_name == "distance":
                        epoch_agg["sum_final_distance"] += self._finite(score_e, d_rt)
                    elif self.scorer_name == "cost":
                        epoch_agg["sum_final_distance"] += d_rt
                    else:
                        epoch_agg["sum_final_distance"] += d_rt

                    # ALWAYS aggregate cost/time
                    epoch_agg["sum_final_cost"] += c_rt
                    epoch_agg["sum_final_time"] += t_rt

                self._write_json(os.path.join(epoch_dir, "run_summary_aggregated.json"), epoch_agg)
                log_info(
                    f"Epoch {curr_epoch} aggregated: distance={epoch_agg['sum_final_distance']:.2f} "
                    f"cost={epoch_agg['sum_final_cost']:.2f} time={epoch_agg['sum_final_time']:.2f} "
                    f"cluster_solving_time={epoch_agg['total_cluster_solving_time_s']:.2f}s"
                )

            epoch_total_time = time.time() - epoch_start
            log_info(f"Epoch {curr_epoch} complete for all clusters. Total epoch time: {epoch_total_time:.2f}s")

        log_info("Finalizing all clusters...")
        # Don't time finalize
        for cluster_idx, env in enumerate(envs):
            env.nsga2.finalize()
            
            if self.training_enabled:
                output_dir_final = os.path.join(self.output_dir, f"cluster_{cluster_idx}")
                ensure_dir(output_dir_final)

                perm_final = env.nsga2.best_perm
                routes_final = self._decode_for_cluster(perm_final, cluster_idx)

                # Final saving (NOT included in solving time)
                self._save_routes_and_summary(output_dir_final, routes_final, cluster_idx)
                self._save_metrics_csv(output_dir_final, env.nsga2.metrics, suffix=self.algo_name)

                # MINIMAL: write metadata (final)
                write_metadata_json(
                    output_dir_final,
                    "metadata.json",
                    config=self.config,
                    vrp=self.vrps[cluster_idx],
                    runtime_s=cluster_solving_times[cluster_idx],
                    iterations=None,
                    algorithm=self.algo_name,
                    scorer=self.scorer_name,
                    algo_params=self.params,
                    cluster_id=cluster_idx,
                )

                # totals from routes (no NaN)
                total_distance_rt, total_time_rt, total_cost_rt = self._totals_from_routes(routes_final, cluster_idx)

                best_info = getattr(env.nsga2, "best_info", {}) or {}
                summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
                score_f = float(summary_info.get("score", env.nsga2.best_score))

                if self.scorer_name == "distance":
                    final_distance = self._finite(score_f, total_distance_rt)
                    final_cost = total_cost_rt
                elif self.scorer_name == "cost":
                    final_cost = self._finite(score_f, total_cost_rt)
                    final_distance = total_distance_rt
                else:
                    final_distance = total_distance_rt
                    final_cost = total_cost_rt

                final_summary = {
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "cluster_id": cluster_idx,
                    "final_distance": float(final_distance),
                    "final_cost": float(final_cost),
                    "final_time": float(total_time_rt),
                    "score": float(score_f),
                    "solving_time_s": float(cluster_solving_times[cluster_idx]),
                }
                self._write_json(os.path.join(output_dir_final, "final_summary.json"), final_summary)

        aggregated = {
            "clusters": len(envs),
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "epochs": self.epochs,
            "sum_final_distance": 0.0,
            "sum_final_cost": 0.0,
            "sum_final_time": 0.0,
            "total_solving_time_s": float(total_solving_time_s),
        }

        for cluster_idx, env in enumerate(envs):
            perm_f = env.nsga2.best_perm
            routes_f = self._decode_for_cluster(perm_f, cluster_idx)
            d_rt, t_rt, c_rt = self._totals_from_routes(routes_f, cluster_idx)

            best_info = getattr(env.nsga2, "best_info", {}) or {}
            summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
            score_f = float(summary_info.get("score", env.nsga2.best_score))

            if self.scorer_name == "distance":
                aggregated["sum_final_distance"] += self._finite(score_f, d_rt)
            else:
                aggregated["sum_final_distance"] += d_rt

            aggregated["sum_final_cost"] += c_rt
            aggregated["sum_final_time"] += t_rt

        return aggregated

    # ==============================
    # Non-RL path
    # ==============================
    def _run_non_rl(self) -> Dict[str, Any]:
        log_info("Algorithm: %s | Scorer: %s", self.algo_name.upper(), self.scorer_name.upper())
        log_info("Output dir: %s", self.output_dir)
        log_info("Processing %d clusters", len(self.algos))

        cluster_summaries = []
        total_solving_time_s = 0.0

        for cluster_idx, algo in enumerate(self.algos):
            log_info(f"Processing cluster {cluster_idx}")

            perm, score, metrics, solving_time_s = self._solve_with_timeout(
                algo, iters=self.iters, time_limit=self.time_limit
            )
            total_solving_time_s += solving_time_s

            output_dir_cluster = os.path.join(self.output_dir, f"cluster_{cluster_idx}")
            ensure_dir(output_dir_cluster)

            routes = self._decode_for_cluster(perm, cluster_idx)
            
            self._save_routes_and_summary(output_dir_cluster, routes, cluster_idx)
            self._save_metrics_csv(output_dir_cluster, metrics, suffix=self.algo_name)

            # MINIMAL: write metadata
            write_metadata_json(
                output_dir_cluster,
                "metadata.json",
                config=self.config,
                vrp=self.vrps[cluster_idx],
                runtime_s=solving_time_s,
                iterations=(int(metrics[-1]["iter"]) if metrics else None),
                algorithm=self.algo_name,
                scorer=self.scorer_name,
                algo_params=self.params,
                cluster_id=cluster_idx,
            )

            # totals from routes (no NaN)
            total_distance_rt, total_time_rt, total_cost_rt = self._totals_from_routes(routes, cluster_idx)

            best_info = getattr(algo, "best_info", {}) or {}
            summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
            score_final = float(summary_info.get("score", score))

            if self.scorer_name == "distance":
                final_distance = self._finite(score_final, total_distance_rt)
                final_cost = total_cost_rt
            elif self.scorer_name == "cost":
                final_cost = self._finite(score_final, total_cost_rt)
                final_distance = total_distance_rt
            else:
                final_distance = total_distance_rt
                final_cost = total_cost_rt

            cluster_summary = {
                "algorithm": self.algo_name,
                "scorer": self.scorer_name,
                "cluster_id": cluster_idx,
                "final_distance": float(final_distance),
                "final_cost": float(final_cost),
                "final_time": float(total_time_rt),
                "score": float(score_final),
                "solving_time_s": float(solving_time_s),
                "iterations": int(metrics[-1]["iter"]) if metrics else 0,
            }
            cluster_summaries.append(cluster_summary)

            self._write_json(os.path.join(output_dir_cluster, "cluster_summary.json"), cluster_summary)
            log_info(f"Cluster {cluster_idx} complete: score={score_final:.6f}")

        aggregated = {
            "clusters": len(self.algos),
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "sum_final_distance": float(sum(float(s["final_distance"]) for s in cluster_summaries)),
            "sum_final_cost": float(sum(float(s["final_cost"]) for s in cluster_summaries)),
            "sum_final_time": float(sum(float(s["final_time"]) for s in cluster_summaries)),
            "total_solving_time_s": float(total_solving_time_s),
        }

        return aggregated

    # ==============================
    # Cooperative solve with timeout
    # ==============================
    def _solve_with_timeout(self, algo, *, iters: int, time_limit: float):
        stop_event = threading.Event()
        result: Dict[str, Any] = {}

        def _worker():
            try:
                out = algo.solve(iters=iters, stop_event=stop_event)
                perm, score, metrics, solving_time_s = out
                result.update(dict(perm=perm, score=score, metrics=metrics, solving_time_s=solving_time_s))
            except Exception as e:
                result["error"] = e

        t0 = time.time()
        th = threading.Thread(target=_worker, daemon=False)
        th.start()
        th.join(timeout=time_limit)

        if th.is_alive():
            stop_event.set()
            th.join()

            if not result:
                if hasattr(algo, "best_perm") and hasattr(algo, "best_score"):
                    result["perm"] = algo.best_perm
                    result["score"] = algo.best_score
                    result["metrics"] = getattr(algo, "metrics", [])
                    result["solving_time_s"] = time.time() - t0
                else:
                    raise RuntimeError("Algorithm didn't stop cooperatively and has no best_* exposed.")
        else:
            if "error" in result:
                raise result["error"]

        result.setdefault("metrics", [])
        result.setdefault("solving_time_s", time.time() - t0)
        return result["perm"], result["score"], result["metrics"], result["solving_time_s"]

    # ==============================
    # Decode helper
    # ==============================
    def _decode_for_cluster(self, perm: List[int], cluster_idx: int) -> List[List[int]]:
        vrp = self.vrps[cluster_idx]
        if self.decode_mode == "split_equal":
            return decode_split_equal(perm, vrp)
        return decode_minimize(perm, vrp)

    # ==============================
    # I/O helpers
    # ==============================
    def _save_routes_and_summary(self, out_dir: str, routes: List[List[int]], cluster_idx: int) -> None:
        ensure_dir(out_dir)
        vrp = self.vrps[cluster_idx]
        write_routes_json(out_dir, self.routes_name, routes, vrp)
        write_summary_csv(out_dir, self.summary_name, routes, vrp)

    def _save_metrics_csv(self, out_dir: str, metrics: List[Dict[str, Any]], *, suffix: str) -> None:
        df = pd.DataFrame(metrics)
        if not df.empty:
            columns = ['iter', 'best', 'mean', 'best_route_time', 'mean_route_time']
            for col in df.columns:
                if col not in columns:
                    columns.append(col)
            
            columns = [col for col in columns if col in df.columns]
            df = df[columns]
            
            path = os.path.join(out_dir, "metrics.csv")
            df.to_csv(path, index=False)
            log_info("Metrics saved: %s", path)

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)