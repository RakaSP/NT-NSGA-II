#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import torch as T

from Algorithm.Gym.EAControlEnv import EAControlEnv
from Algorithm.NN.Second import SecondNN
from Utils.Logger import set_log_level, log_info, log_trace
from Utils.Utils import (
    load_problem_from_config,
    decode_routes,
    validate_capacity,
    eval_routes_cost,
    write_routes_json,
    write_summary_csv,
    ensure_dir,
)


# ==============================
# Engine
# ==============================
class VRPSolverEngine:
    # ---- defaults / constants ----
    DEFAULT_TIME_LIMIT_S = 60.0  # per epoch for RL, whole run for non-RL
    DEFAULT_MAX_ITERS = 99999999999999  # effectively "unlimited"

    def __init__(self, config: Dict[str, Any]):
        # logging
        set_log_level(str(config.get("logger", "INFO")).upper())

        # io
        self.nodes_path = config.get("nodes", "vrp_nodes.csv")
        self.vehicles_path = config.get("vehicles", "vrp_vehicles.csv")
        self.output_dir = config.get("output_dir", "results")
        ensure_dir(self.output_dir)
        self.routes_name = config.get("routes_out", "solution_routes.json")
        self.summary_name = config.get("summary_out", "solution_summary.csv")

        # choices
        self.algo_name = str(config.get("algorithm")).lower()
        self.scorer_name = str(config.get("scorer")).lower()
        self.params: Dict[str, Any] = config["algo_params"][self.algo_name]

        # limits / training
        self.iters: int = int(self.DEFAULT_MAX_ITERS)
        self.time_limit: float = float(config.get("time_limit", self.DEFAULT_TIME_LIMIT_S))
        self.epochs: int = int(config.get("epochs", 1))
        self.batch_size: int = int(config.get("batch_size", 1))

        # toggles
        self.rl_enabled: bool = bool(config.get("rl_enabled"))
        self.training_enabled: bool = bool(config.get("training_enabled", True))

        # optional checkpoint
        self.secondNN_checkpoint_path: Optional[str] = config.get("second_nn_path")

        # state
        self.vrp: Optional[Dict[str, Any]] = None
        self.algo = None
        self.second_nn: Optional[SecondNN] = None
        self.opt_second: Optional[T.optim.Optimizer] = None

        # REINFORCE knobs
        self.entropy_coef = 1e-3
        self.std_reg_coef = 0.0
        self.std_target = 0.25
        self.adv_clip = 5.0

    # ==============================
    # Factories
    # ==============================
    def _make_scorer(self):
        if self.scorer_name == "cost":
            from Scorer.Cost import score_from_perm
        elif self.scorer_name == "distance":
            from Scorer.Distance import score_from_perm
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
    def prepare(self, cfg: Dict[str, Any]) -> None:
        self.vrp = load_problem_from_config(cfg)
        Algo = self._make_algorithm()
        self.algo = Algo(vrp=self.vrp, scorer=self.scorer_name, params=self.params)

    def build_agent(self, in_dim: int = 10, hidden: int = 128, lr: float = 3e-4) -> None:
        """
        Build (or load) the Gaussian policy ('second_nn') and its optimizer.

        Behavior:
        - If self.secondNN_checkpoint_path is None: fresh SecondNN(in_dim=10).
        - Else: attempt to load state_dict from file (or <dir>/secondNN.pt).
          On mismatch, fall back to fresh model.
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
        """REINFORCE with centered advantages and entropy bonus."""
        assert self.second_nn is not None and self.opt_second is not None

        rewards = T.stack(batch_rewards)  # (B,)
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

        # logging extras
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
        assert self.vrp is not None and self.algo is not None

        if self.rl_enabled:
            return self._run_rl()
        else:
            return self._run_non_rl()

    # ==============================
    # RL path
    # ==============================
    def _run_rl(self) -> Dict[str, Any]:
        env = EAControlEnv(self.algo)
        obs = env.reset()
        self.build_agent(10)  # creates self.second_nn

        best_solution = float("inf")
        if hasattr(env.nsga2, "start_run"):
            env.nsga2.start_run()

        run_summary: Dict[str, Any] = {}
        curr_epoch = 0

        # turn off training? then 1 epoch and batch_size=1
        if not self.training_enabled:
            self.epochs = 1
            self.batch_size = 1

        while curr_epoch < self.epochs:
            epoch_start_time = time.time()
            obs = env.reset()
            curr_iter = 0
            epoch_cx_rates: List[float] = []
            epoch_mut_rates: List[float] = []

            # epoch loop with time limit
            while True:
                if time.time() - epoch_start_time >= self.time_limit:
                    log_info(f"[epoch {curr_epoch}] Time limit reached, stopping iteration")
                    break

                obs_memory: List[T.Tensor] = []
                obs_reward: List[T.Tensor] = []

                for _ in range(self.batch_size):
                    if time.time() - epoch_start_time >= self.time_limit:
                        break
                    action = self.second_nn(obs)  # type: ignore[arg-type]
                    obs, reward = env.step(action)

                    # collect actual hyper-parameters if available
                    epoch_cx_rates.append(float(env.nsga2.crossover_rate))
                    epoch_mut_rates.append(float(env.nsga2.mutation_rate))

                    obs_memory.append(obs)
                    obs_reward.append(T.as_tensor(reward, dtype=T.float32))
                    curr_iter += 1

                if self.training_enabled and obs_memory:
                    self.update_second_nn(obs_memory, obs_reward)

                if time.time() - epoch_start_time >= self.time_limit:
                    break

            # epoch artifacts directory
            epoch_dir = os.path.join(self.output_dir, f"epoch_{curr_epoch}")
            ensure_dir(epoch_dir)

            # checkpoint if improved
            if env.nsga2.best_score < best_solution:
                best_solution = env.nsga2.best_score
                T.save(self.second_nn.state_dict(), os.path.join(epoch_dir, "secondNN.pt"))  # type: ignore[arg-type]

            # epoch best routes + metrics
            perm_e = env.nsga2.best_perm
            routes_e = decode_routes(perm_e, self.vrp)
            validate_capacity(routes_e, self.vrp)
            total_cost_e, total_distance_e, total_time_e = eval_routes_cost(routes_e, self.vrp)

            # write artifacts inside epoch dir
            self._save_routes_and_summary(epoch_dir, routes_e)
            self._save_metrics_csv(epoch_dir, env.nsga2.metrics, suffix=self.algo_name)

            epoch_runtime = time.time() - epoch_start_time
            run_summary = {
                "algorithm": self.algo_name,
                "scorer": self.scorer_name,
                "final_distance": float(total_distance_e),
                "final_cost": float(total_cost_e),
                "final_time": float(total_time_e),
                "epoch_runtime_s": float(epoch_runtime),
                "iterations": int(curr_iter),
                "epoch": int(curr_epoch),
            }
            self._write_json(os.path.join(epoch_dir, "run_summary.json"), run_summary)

            # epoch hx summary
            if epoch_cx_rates and epoch_mut_rates:
                mean_cx = sum(epoch_cx_rates) / len(epoch_cx_rates)
                mean_mut = sum(epoch_mut_rates) / len(epoch_mut_rates)
            else:
                mean_cx, mean_mut = float("nan"), float("nan")
            log_info(f"[epoch {curr_epoch}] mean_cx={mean_cx:.4f} mean_mut={mean_mut:.4f} "
                     f"iterations={curr_iter} runtime={epoch_runtime:.2f}s")

            curr_epoch += 1
            log_info(f"FINAL RESULT: {env.nsga2.best_perm}, Score: {env.nsga2.best_score}")

        # finalize and write final artifacts to base output_dir
        runtime_s = env.nsga2.finalize()
        perm_final = env.nsga2.best_perm
        routes_final = decode_routes(perm_final, self.vrp)
        validate_capacity(routes_final, self.vrp)
        total_cost, total_distance, total_time = eval_routes_cost(routes_final, self.vrp)

        self._save_routes_and_summary(self.output_dir, routes_final)
        self._save_metrics_csv(self.output_dir, env.nsga2.metrics, suffix=self.algo_name)

        final_summary = {
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "final_distance": float(total_distance),
            "final_cost": float(total_cost),
            "final_time": float(total_time),
            "runtime_s": float(runtime_s),
            # last epoch's iteration count (curr_iter resets each epoch; here we surface last)
            "total_iterations": int(curr_iter),
        }
        self._write_json(os.path.join(self.output_dir, "run_summary.json"), final_summary)
        return final_summary

    # ==============================
    # Non-RL path
    # ==============================
    def _run_non_rl(self) -> Dict[str, Any]:
        log_info("Algorithm: %s | Scorer: %s", self.algo_name.upper(), self.scorer_name.upper())
        log_info("Output dir: %s", self.output_dir)

        # Worker to call solve()
        result_holder: Dict[str, Any] = {}

        def _worker():
            try:
                perm, score, metrics, runtime_s = self.algo.solve(iters=self.iters)
                result_holder["perm"] = perm
                result_holder["score"] = score
                result_holder["metrics"] = metrics
                result_holder["runtime_s"] = runtime_s
            except Exception as e:
                result_holder["error"] = e

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout=self.time_limit)

        if th.is_alive():
            log_info("Time limit reached, using best solution found so far")
            # We can't kill the thread safely; we just proceed with best so far.
            if hasattr(self.algo, "best_perm") and hasattr(self.algo, "best_score"):
                perm = self.algo.best_perm
                score = self.algo.best_score
                metrics = getattr(self.algo, "metrics", [])
                runtime_s = self.time_limit
            else:
                raise RuntimeError("Algorithm doesn't expose best solution after timeout")
        else:
            if "error" in result_holder:
                raise result_holder["error"]
            perm = result_holder["perm"]
            score = result_holder["score"]
            metrics = result_holder["metrics"]
            runtime_s = result_holder["runtime_s"]

        routes = decode_routes(perm, self.vrp)
        validate_capacity(routes, self.vrp)
        total_cost, total_distance, total_time = eval_routes_cost(routes, self.vrp)

        self._save_routes_and_summary(self.output_dir, routes)
        self._save_metrics_csv(self.output_dir, metrics, suffix=self.algo_name)

        run_summary: Dict[str, Any] = {
            "algorithm": self.algo_name,
            "scorer": self.scorer_name,
            "final_distance": float(total_distance),
            "final_cost": float(total_cost),
            "final_time": float(total_time),
            "runtime_s": float(runtime_s),
            "iterations": int(metrics[-1]["iter"]) if metrics else 0,
        }

        # NSGA2 extras: Pareto front
        if self.algo_name == "nsga2" and hasattr(self.algo, "get_pareto_front"):
            pareto_solutions, pareto_objectives = self.algo.get_pareto_front()
            pareto_data = []
            for i, (sol, obj) in enumerate(zip(pareto_solutions, pareto_objectives)):
                cost, distance, time_val = obj
                pareto_data.append({"solution_id": i, "cost": float(cost), "distance": float(distance), "time": float(time_val)})
            pareto_path = os.path.join(self.output_dir, "pareto_front.json")
            self._write_json(pareto_path, pareto_data)
            log_info("Pareto front saved: %s", pareto_path)
            run_summary["pareto_front_size"] = len(pareto_solutions)

        self._write_json(os.path.join(self.output_dir, "run_summary.json"), run_summary)
        log_info("Final distance: %.6f | Final cost: %.6f | Final time: %.6f", total_distance, total_cost, total_time)
        return run_summary

    # ==============================
    # I/O helpers
    # ==============================
    def _save_routes_and_summary(self, out_dir: str, routes: List[List[int]]) -> None:
        ensure_dir(out_dir)
        write_routes_json(out_dir, self.routes_name, routes, self.vrp)
        write_summary_csv(out_dir, self.summary_name, routes, self.vrp)

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
