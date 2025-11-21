#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch as T

from Algorithm.Gym.EAControlEnv import EAControlEnv
from Algorithm.NN.Second import SecondNN
from Utils.Logger import set_log_level, log_info, log_trace
from vrp_core import (
    decode_split_equal,
    decode_minimize,
    validate_capacity,
    write_routes_json,
    write_summary_csv,
    ensure_dir,
)


# ==============================
# Engine
# ==============================
class VRPSolverEngine:
    # ---- defaults / constants ----
    DEFAULT_TIME_LIMIT_S = 1.0
    DEFAULT_MAX_ITERS = 99999999999999

    def __init__(self, config: Dict[str, Any]):
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
        self.time_limit: float = float(config.get("time_limit", self.DEFAULT_TIME_LIMIT_S))
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
            algo = Algo(vrp=vrp, scorer=self.scorer_name, params=self.params)
            self.algos.append(algo)
        
        log_info("Prepared %d clusters for processing", len(vrps))

    def build_agent(self, in_dim: int = 10, hidden: int = 128, lr: float = 3e-4) -> None:
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
                    self.second_nn = _new_model()
            else:
                log_info("SecondNN checkpoint not found at '%s'. Using fresh model.", candidate)
                self.second_nn = _new_model()
        else:
            self.second_nn = _new_model()
            log_info("No SecondNN checkpoint provided. Using fresh model (in_dim=%d, hidden=%d).", in_dim, hidden)

        self.opt_second = T.optim.Adam(self.second_nn.parameters(), lr=lr)

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
        # Create one environment per cluster
        envs = [EAControlEnv(algo) for algo in self.algos]
        for env in envs:
            env.reset()
            if hasattr(env.nsga2, "start_run"):
                env.nsga2.start_run()
        
        self.build_agent(10)

        # Track best solution globally
        best_solution = float("inf")
        
        if not self.training_enabled:
            self.epochs = 1
            self.batch_size = 1

        # Track total runtime across all clusters
        total_runtime_s = 0.0
        
        # Process all clusters epoch by epoch
        for curr_epoch in range(self.epochs):
            epoch_start = time.time()
            log_info(f"Starting epoch {curr_epoch} for all {len(envs)} clusters")
            
            # Track epoch runtime per cluster
            cluster_runtimes = []
            
            # Process each cluster in this epoch
            for cluster_idx, env in enumerate(envs):
                log_info(f"[Epoch {curr_epoch}] Processing cluster {cluster_idx}")
                
                obs = env.reset()
                curr_iter = 0
                epoch_cx, epoch_mut = [], []
                cluster_start = time.time()

                # Run iterations for this cluster
                while True:
                    if time.time() - cluster_start >= self.time_limit:
                        log_info(f"[Epoch {curr_epoch}] Cluster {cluster_idx} time limit reached")
                        break

                    obs_memory: List[T.Tensor] = []
                    obs_reward: List[T.Tensor] = []

                    for _ in range(self.batch_size):
                        if time.time() - cluster_start >= self.time_limit:
                            break
                        
                        action = self.second_nn(obs)
                        obs, reward = env.step(action)

                        epoch_cx.append(float(env.nsga2.crossover_rate))
                        epoch_mut.append(float(env.nsga2.mutation_rate))

                        obs_memory.append(obs)
                        obs_reward.append(T.as_tensor(reward, dtype=T.float32))
                        curr_iter += 1

                    if self.training_enabled and obs_memory:
                        self.update_second_nn(obs_memory, obs_reward)

                    if time.time() - cluster_start >= self.time_limit:
                        break

                # Calculate cluster runtime
                cluster_runtime = time.time() - cluster_start
                cluster_runtimes.append(cluster_runtime)
                total_runtime_s += cluster_runtime

                # Save this cluster's results for this epoch
                if self.training_enabled:
                    output_dir_cluster = os.path.join(self.output_dir, f"epoch_{curr_epoch}", f"cluster_{cluster_idx}")
                else:
                    output_dir_cluster = os.path.join(self.output_dir, f"cluster_{cluster_idx}")
                
                ensure_dir(output_dir_cluster)

                # Track best for checkpoint
                if env.nsga2.best_score < best_solution:
                    best_solution = env.nsga2.best_score

                # Decode and save cluster results
                perm_e = env.nsga2.best_perm
                routes_e = self._decode_for_cluster(perm_e, cluster_idx)
                validate_capacity(routes_e, self.vrps[cluster_idx])
                self._save_routes_and_summary(output_dir_cluster, routes_e, cluster_idx)
                self._save_metrics_csv(output_dir_cluster, env.nsga2.metrics, suffix=self.algo_name)

                best_info = getattr(env.nsga2, "best_info", {}) or {}
                summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}

                score_e = float(summary_info.get("score", env.nsga2.best_score))
                total_distance_e = float(summary_info.get("total_distance", float("nan")))
                total_time_e = float(summary_info.get("total_time", 0.0))

                if self.scorer_name == "distance":
                    final_distance_e = score_e
                    final_cost_e = float("nan")
                elif self.scorer_name == "cost":
                    final_cost_e = score_e
                    final_distance_e = total_distance_e
                else:
                    final_distance_e = total_distance_e
                    final_cost_e = score_e

                cluster_summary = {
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "cluster_id": cluster_idx,
                    "final_distance": final_distance_e,
                    "final_cost": final_cost_e,
                    "final_time": float(total_time_e),
                    "score": score_e,
                    "cluster_runtime_s": float(cluster_runtime),
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
                    f"iterations={curr_iter} runtime={cluster_runtime:.2f}s score={score_e:.6f}"
                )

            # Save ONE checkpoint per epoch after all clusters are done
            if self.training_enabled:
                epoch_dir = os.path.join(self.output_dir, f"epoch_{curr_epoch}")
                ensure_dir(epoch_dir)
                T.save(self.second_nn.state_dict(), os.path.join(epoch_dir, "secondNN.pt"))
                log_info(f"Saved checkpoint for epoch {curr_epoch}")
                
                # Calculate total cluster runtime for this epoch
                epoch_cluster_runtime = sum(cluster_runtimes)
                
                # Create aggregated summary for this epoch
                epoch_agg = {
                    "epoch": curr_epoch,
                    "clusters": len(envs),
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "sum_final_distance": 0.0,
                    "sum_final_cost": 0.0,
                    "sum_final_time": 0.0,
                    "epoch_runtime_s": float(time.time() - epoch_start),
                    "total_cluster_runtime_s": float(epoch_cluster_runtime),  # Add total cluster runtime
                }
                
                # Aggregate all cluster results for this epoch
                for cluster_idx, env in enumerate(envs):
                    best_info = getattr(env.nsga2, "best_info", {}) or {}
                    summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
                    
                    score_e = float(summary_info.get("score", env.nsga2.best_score))
                    total_distance_e = float(summary_info.get("total_distance", 0.0))
                    total_time_e = float(summary_info.get("total_time", 0.0))
                    
                    if self.scorer_name == "distance":
                        epoch_agg["sum_final_distance"] += score_e
                    elif self.scorer_name == "cost":
                        epoch_agg["sum_final_cost"] += score_e
                        epoch_agg["sum_final_distance"] += total_distance_e
                    else:
                        epoch_agg["sum_final_distance"] += total_distance_e
                    
                    epoch_agg["sum_final_time"] += total_time_e
                
                # Save epoch aggregated summary
                self._write_json(os.path.join(epoch_dir, "run_summary_aggregated.json"), epoch_agg)
                log_info(f"Epoch {curr_epoch} aggregated: distance={epoch_agg['sum_final_distance']:.2f} cost={epoch_agg['sum_final_cost']:.2f} cluster_runtime={epoch_agg['total_cluster_runtime_s']:.2f}s")

            epoch_runtime = time.time() - epoch_start
            log_info(f"Epoch {curr_epoch} complete for all clusters. Total epoch time: {epoch_runtime:.2f}s")

        # Finalize all clusters
        log_info("Finalizing all clusters...")
        final_cluster_runtimes = []
        for cluster_idx, env in enumerate(envs):
            runtime_s = env.nsga2.finalize()
            final_cluster_runtimes.append(runtime_s)
            total_runtime_s += runtime_s
            
            if self.training_enabled:
                # Save final artifacts per cluster
                output_dir_final = os.path.join(self.output_dir, f"cluster_{cluster_idx}")
                ensure_dir(output_dir_final)
                
                perm_final = env.nsga2.best_perm
                routes_final = self._decode_for_cluster(perm_final, cluster_idx)
                validate_capacity(routes_final, self.vrps[cluster_idx])
                self._save_routes_and_summary(output_dir_final, routes_final, cluster_idx)
                self._save_metrics_csv(output_dir_final, env.nsga2.metrics, suffix=self.algo_name)

                best_info = getattr(env.nsga2, "best_info", {}) or {}
                summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}

                score_f = float(summary_info.get("score", env.nsga2.best_score))
                total_distance_f = float(summary_info.get("total_distance", float("nan")))
                total_time_f = float(summary_info.get("total_time", 0.0))

                if self.scorer_name == "distance":
                    final_distance = score_f
                    final_cost = float("nan")
                elif self.scorer_name == "cost":
                    final_cost = score_f
                    final_distance = total_distance_f
                else:
                    final_distance = total_distance_f
                    final_cost = score_f

                final_summary = {
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "cluster_id": cluster_idx,
                    "final_distance": final_distance,
                    "final_cost": final_cost,
                    "final_time": float(total_time_f),
                    "score": score_f,
                    "runtime_s": float(runtime_s),
                }
                self._write_json(os.path.join(output_dir_final, "final_summary.json"), final_summary)

        # Calculate total final cluster runtime
        total_final_cluster_runtime = sum(final_cluster_runtimes)
        
        # Create aggregated summary
        aggregated = {
            "clusters": len(envs),
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "epochs": self.epochs,
            "sum_final_distance": 0.0,
            "sum_final_cost": 0.0,
            "sum_final_time": 0.0,
            "total_runtime_s": float(total_runtime_s),  # Add total runtime across all clusters
            "total_cluster_runtime_s": float(total_final_cluster_runtime),  # Add total cluster runtime
        }

        for cluster_idx, env in enumerate(envs):
            best_info = getattr(env.nsga2, "best_info", {}) or {}
            summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}
            
            score_f = float(summary_info.get("score", env.nsga2.best_score))
            total_distance_f = float(summary_info.get("total_distance", 0.0))
            total_time_f = float(summary_info.get("total_time", 0.0))
            
            if self.scorer_name == "distance":
                aggregated["sum_final_distance"] += score_f
            elif self.scorer_name == "cost":
                aggregated["sum_final_cost"] += score_f
                aggregated["sum_final_distance"] += total_distance_f
            else:
                aggregated["sum_final_distance"] += total_distance_f
            
            aggregated["sum_final_time"] += total_time_f

        return aggregated

    # ==============================
    # Non-RL path
    # ==============================
    def _run_non_rl(self) -> Dict[str, Any]:
        log_info("Algorithm: %s | Scorer: %s", self.algo_name.upper(), self.scorer_name.upper())
        log_info("Output dir: %s", self.output_dir)
        log_info("Processing %d clusters", len(self.algos))

        cluster_summaries = []
        total_runtime_s = 0.0

        for cluster_idx, algo in enumerate(self.algos):
            log_info(f"Processing cluster {cluster_idx}")
            
            perm, score, metrics, runtime_s = self._solve_with_timeout(
                algo, iters=self.iters, time_limit=self.time_limit
            )
            total_runtime_s += runtime_s

            output_dir_cluster = os.path.join(self.output_dir, f"cluster_{cluster_idx}")
            ensure_dir(output_dir_cluster)

            routes = self._decode_for_cluster(perm, cluster_idx)
            validate_capacity(routes, self.vrps[cluster_idx])
            self._save_routes_and_summary(output_dir_cluster, routes, cluster_idx)
            self._save_metrics_csv(output_dir_cluster, metrics, suffix=self.algo_name)

            best_info = getattr(algo, "best_info", {}) or {}
            summary_info = best_info.get("summary", {}) if isinstance(best_info, dict) else {}

            score_final = float(summary_info.get("score", score))
            total_distance = float(summary_info.get("total_distance", float("nan")))
            total_time = float(summary_info.get("total_time", 0.0))

            if self.scorer_name == "distance":
                final_distance = score_final
                final_cost = float("nan")
            elif self.scorer_name == "cost":
                final_cost = score_final
                final_distance = total_distance
            else:
                final_distance = total_distance
                final_cost = score_final

            cluster_summary = {
                "algorithm": self.algo_name,
                "scorer": self.scorer_name,
                "cluster_id": cluster_idx,
                "final_distance": final_distance,
                "final_cost": final_cost,
                "final_time": float(total_time),
                "score": score_final,
                "runtime_s": float(runtime_s),
                "iterations": int(metrics[-1]["iter"]) if metrics else 0,
            }
            cluster_summaries.append(cluster_summary)

            self._write_json(os.path.join(output_dir_cluster, "cluster_summary.json"), cluster_summary)
            log_info(f"Cluster {cluster_idx} complete: score={score_final:.6f}")

        # Aggregate all clusters
        aggregated = {
            "clusters": len(self.algos),
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "sum_final_distance": float(sum(s.get("final_distance", 0.0) for s in cluster_summaries if not pd.isna(s.get("final_distance", float("nan"))))),
            "sum_final_cost": float(sum(s.get("final_cost", 0.0) for s in cluster_summaries if not pd.isna(s.get("final_cost", float("nan"))))),
            "sum_final_time": float(sum(s.get("final_time", 0.0) for s in cluster_summaries)),
            "total_runtime_s": float(total_runtime_s),  # Add total runtime across all clusters
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
                perm, score, metrics, runtime_s = out
                result.update(dict(perm=perm, score=score, metrics=metrics, runtime_s=runtime_s))
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
                    result["runtime_s"] = time.time() - t0
                else:
                    raise RuntimeError("Algorithm didn't stop cooperatively and has no best_* exposed.")
        else:
            if "error" in result:
                raise result["error"]

        result.setdefault("metrics", [])
        result.setdefault("runtime_s", time.time() - t0)
        return result["perm"], result["score"], result["metrics"], result["runtime_s"]

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
            path = os.path.join(out_dir, f"metrics_{suffix}.csv")
            df.to_csv(path, index=False)
            log_info("Metrics saved: %s", path)

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)