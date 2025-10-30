#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd
from Algorithm.Gym.EAControlEnv import EAControlEnv
from Utils.Logger import set_log_level, log_info, log_trace
from Utils.Utils import load_problem_from_config, decode_routes, validate_capacity, eval_routes_cost, write_routes_json, write_summary_csv, ensure_dir
import torch as T

from Algorithm.NN.Second import SecondNN


class VRPSolverEngine:
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
        self.iters: int = int(self.params.get("iters"))
        self.epochs: int = int(config.get("epochs"))
        self.batch_size: int = int(config.get("batch_size"))

        # late
        self.vrp: Dict[str, Any] | None = None
        self.score_fn = None
        self.algo = None
        self.rl_enabled = True
        self.training_enabled = True
        self.secondNN_checkpoint_path = None

    # --- factories ---
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
            raise ValueError(
                "Unknown algorithm: use 'ga', 'pso', 'aco', 'nsga2', or 'ntnsga2'")
        return Algo

    def update_second_nn(self, batch_obs, batch_rewards):
        """REINFORCE with centered advantages, entropy bonus, and std regularization"""

        # 1) Stack rewards -> advantages (center/normalize safely)
        rewards = T.stack(batch_rewards)  # (B,)
        adv = rewards

        if adv.numel() > 1:
            mean = adv.mean()
            std = adv.std()
            if float(std.item()) > 1e-8:
                adv = (adv - mean) / (std + 1e-8)
            else:
                # all rewards equal: just zero-mean to remove sign bias
                adv = adv - mean
        else:
            adv = adv - adv.mean()

        # optional clipping for stability
        if self.adv_clip is not None and self.adv_clip > 0:
            adv = adv.clamp(min=-self.adv_clip, max=self.adv_clip)

        # 2) Recompute log-probs and entropies from stored features
        logps = []
        entropies = []
        for features in batch_obs:
            _, _, logp_second, info = self.second_nn(features)
            logps.append(logp_second)                 # shape (1,) or (Bf,)
            entropies.append(info["entropy_u"])       # same shape

        logps = T.stack(logps).squeeze()          # (B,)
        entropies = T.stack(entropies).squeeze()  # (B,)

        # 3) Policy loss: maximize E[adv * logp] + entropy
        policy_loss = -(logps * adv.detach()).mean()

        # entropy bonus (keeps std>0, discourages action collapse)
        if self.entropy_coef > 0:
            policy_loss = policy_loss - self.entropy_coef * entropies.mean()

        # std regularization toward a target in unsquashed space
        if self.std_reg_coef > 0:
            std_cx = self.second_nn.log_std_cx.exp()
            std_mut = self.second_nn.log_std_mut.exp()
            std_reg = ((std_cx - self.std_target) ** 2 +
                       (std_mut - self.std_target) ** 2)
            policy_loss = policy_loss + self.std_reg_coef * std_reg

        # 4) Optimize
        param_snapshot = [p.data.clone()
                          for p in self.second_nn.parameters() if p.requires_grad]

        self.opt_second.zero_grad()
        policy_loss.backward()

        # clip gradients
        T.nn.utils.clip_grad_norm_(
            self.second_nn.parameters(), max_norm=1.0)

        # gradient norm (for logging)
        gn = 0.0
        for p in self.second_nn.parameters():
            if p.grad is not None:
                gn += p.grad.detach().norm().item() ** 2
        gn = gn ** 0.5

        self.opt_second.step()

        # parameter delta (for logging)
        delta = 0.0
        for p, old in zip(self.second_nn.parameters(), param_snapshot):
            delta += (p.data - old).norm().item() ** 2
        delta = delta ** 0.5

        log_trace(
            "[NTNSGA2] Batch update: size=%d loss=%.6f grad_norm=%.6f param_delta=%.6f mean_adv=%.6f std_adv=%.6f",
            len(batch_rewards),
            float(policy_loss.item()),
            gn,
            delta,
            float(adv.mean().item()),
            float(adv.std().item()) if adv.numel() > 1 else 0.0
        )
    # --- lifecycle ---

    def prepare(self, cfg) -> None:
        self.vrp = load_problem_from_config(cfg)
        Algo = self._make_algorithm()
        self.algo = Algo(vrp=self.vrp, scorer=self.scorer_name,
                         params=self.params)

    def build_agent(self, in_dim: int, hidden: int = 128, lr: float = 3e-4) -> None:
        """
        Build the Gaussian policy ('second_nn') and its optimizer.
        - in_dim: observation feature dimension
        - hidden: hidden width for SecondNN
        - lr: learning rate for Adam

        Also sets the few attributes your update_second_nn() expects.
        """
        self.second_nn = SecondNN(in_dim=in_dim, hidden=hidden)

        self.opt_second = T.optim.Adam(self.second_nn.parameters(), lr=lr)

        # Minimal knobs used by update_second_nn (kept as simple constants)
        self.entropy_coef = 1e-3
        self.std_reg_coef = 0.0
        self.std_target = 0.25
        self.adv_clip = 5.0

    def run(self) -> Dict[str, Any]:
        assert self.vrp is not None and self.algo is not None

        if self.rl_enabled:
            env = EAControlEnv(self.algo)
            obs = env.reset()
            self.build_agent(10)  # this build agent to self.second_nn
            best_solution = 100000000000000000

            # start timing inside NSGA2 so finalize() returns runtime
            if hasattr(env.nsga2, "start_run"):
                env.nsga2.start_run()

            # last built summary to return
            run_summary: Dict[str, Any] = {}

            if self.training_enabled:
                curr_epoch = 0

                while curr_epoch < self.epochs:
                    obs = env.reset()
                    curr_iter = 0

                    # NEW: per-epoch trackers for mean cx/mut
                    epoch_cx_rates = []
                    epoch_mut_rates = []

                    while curr_iter < self.iters:
                        obs_memory = []
                        obs_reward = []
                        for k in range(self.batch_size):
                            action = self.second_nn(obs)
                            obs, reward = env.step(action)

                            # NEW: collect the actual rates applied this step
                            epoch_cx_rates.append(
                                float(env.nsga2.crossover_rate))
                            epoch_mut_rates.append(
                                float(env.nsga2.mutation_rate))

                            obs_memory.append(obs)
                            obs_reward.append(T.as_tensor(
                                reward, dtype=T.float32))
                            curr_iter += 1
                            if curr_iter >= self.iters:
                                break
                        self.update_second_nn(obs_memory, obs_reward)

                    # per-epoch output dir
                    epoch_dir = os.path.join(
                        self.output_dir, f"epoch_{curr_epoch}")
                    ensure_dir(epoch_dir)

                    # checkpoint on improvement (save inside epoch dir)
                    if env.nsga2.best_score < best_solution:
                        best_solution = env.nsga2.best_score
                        T.save(self.second_nn.state_dict(),
                               os.path.join(epoch_dir, "secondNN.pt"))

                    # epoch artifacts
                    perm_e = env.nsga2.best_perm
                    if perm_e is not None:
                        routes_e = decode_routes(perm_e, self.vrp)
                        validate_capacity(routes_e, self.vrp)
                        total_cost_e, total_distance_e, total_time_e = eval_routes_cost(
                            routes_e, self.vrp)

                        # routes + summary in epoch dir
                        write_routes_json(
                            epoch_dir, "solution_routes.json", routes_e, self.vrp)
                        write_summary_csv(
                            epoch_dir, "solution_summary.csv", routes_e, self.vrp)

                        # metrics in epoch dir (cumulative so far)
                        metrics_df_e = pd.DataFrame(env.nsga2.metrics)
                        if not metrics_df_e.empty:
                            metrics_df_e.to_csv(os.path.join(
                                epoch_dir, f"metrics_{self.algo_name}.csv"), index=False)

                        # run summary in epoch dir (no finalize yet → runtime_s=None)
                        run_summary = {
                            "algorithm": self.algo_name,
                            "scorer": self.scorer_name,
                            "final_distance": float(total_distance_e),
                            "final_cost": float(total_cost_e),
                            "final_time": float(total_time_e),
                            "runtime_s": env.nsga2.finalize(),
                            "iterations": int(self.iters),
                            "epoch": int(curr_epoch),
                        }
                    else:
                        # still write metrics to epoch dir even if perm missing
                        metrics_df_e = pd.DataFrame(env.nsga2.metrics)
                        if not metrics_df_e.empty:
                            metrics_df_e.to_csv(os.path.join(
                                epoch_dir, f"metrics_{self.algo_name}.csv"), index=False)

                        run_summary = {
                            "algorithm": self.algo_name,
                            "scorer": self.scorer_name,
                            "final_distance": float("nan"),
                            "final_cost": float("nan"),
                            "final_time": float("nan"),
                            "runtime_s": env.nsga2.finalize(),
                            "iterations": int(self.iters),
                            "epoch": int(curr_epoch),
                        }

                    with open(os.path.join(epoch_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                        json.dump(run_summary, f, indent=2)

                    # NEW: print per-epoch mean cx/mut (use f-string: single-arg logging)
                    if epoch_cx_rates and epoch_mut_rates:
                        mean_cx = sum(epoch_cx_rates) / len(epoch_cx_rates)
                        mean_mut = sum(epoch_mut_rates) / len(epoch_mut_rates)
                    else:
                        mean_cx, mean_mut = float("nan"), float("nan")
                    log_info(
                        f"[epoch {curr_epoch}] mean_cx={mean_cx:.4f} mean_mut={mean_mut:.4f}")

                    curr_epoch += 1
                    log_info(
                        f"FINAL RESULT: {env.nsga2.best_perm}, Score: {env.nsga2.best_score}")

                # finalize once after all epochs, write final artifacts in base output_dir
                runtime_s = env.nsga2.finalize()

                perm_final = env.nsga2.best_perm
                if perm_final is not None:
                    routes_final = decode_routes(perm_final, self.vrp)
                    validate_capacity(routes_final, self.vrp)
                    total_cost, total_distance, total_time = eval_routes_cost(
                        routes_final, self.vrp)

                    write_routes_json(
                        self.output_dir, self.routes_name, routes_final, self.vrp)
                    write_summary_csv(
                        self.output_dir, self.summary_name, routes_final, self.vrp)

                    metrics_df = pd.DataFrame(env.nsga2.metrics)
                    if not metrics_df.empty:
                        metrics_df.to_csv(os.path.join(
                            self.output_dir, f"metrics_{self.algo_name}.csv"), index=False)

                    run_summary = {
                        "algorithm": self.algo_name,
                        "scorer": self.scorer_name,
                        "final_distance": float(total_distance),
                        "final_cost": float(total_cost),
                        "final_time": float(total_time),
                        "runtime_s": float(runtime_s),
                        "iterations": int(self.iters),
                    }
                else:
                    metrics_df = pd.DataFrame(env.nsga2.metrics)
                    if not metrics_df.empty:
                        metrics_df.to_csv(os.path.join(
                            self.output_dir, f"metrics_{self.algo_name}.csv"), index=False)

                    run_summary = {
                        "algorithm": self.algo_name,
                        "scorer": self.scorer_name,
                        "final_distance": float("nan"),
                        "final_cost": float("nan"),
                        "final_time": float("nan"),
                        "runtime_s": float(runtime_s),
                        "iterations": int(self.iters),
                    }

                with open(os.path.join(self.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(run_summary, f, indent=2)

                return run_summary

            else:
                # -------- Non-RL path (unchanged) --------
                log_info("Algorithm: %s | Scorer: %s",
                         self.algo_name.upper(), self.scorer_name.upper())
                log_info("Output dir: %s", self.output_dir)

                perm, score, metrics, runtime_s = self.algo.solve(
                    iters=self.iters)

                routes = decode_routes(perm, self.vrp)
                validate_capacity(routes, self.vrp)
                total_cost, total_distance, total_time = eval_routes_cost(
                    routes, self.vrp)

                write_routes_json(
                    self.output_dir, self.routes_name, routes, self.vrp)
                write_summary_csv(
                    self.output_dir, self.summary_name, routes, self.vrp)

                metrics_df = pd.DataFrame(metrics)
                metrics_path = os.path.join(
                    self.output_dir, f"metrics_{self.algo_name}.csv")
                metrics_df.to_csv(metrics_path, index=False)
                log_info("Metrics saved: %s", metrics_path)

                run_summary = {
                    "algorithm": self.algo_name,
                    "scorer": self.scorer_name,
                    "final_distance": float(total_distance),
                    "final_cost": float(total_cost),
                    "final_time": float(total_time),
                    "runtime_s": float(runtime_s),
                    "iterations": int(metrics[-1]["iter"]) if metrics else 0,
                }

                if self.algo_name == "nsga2" and hasattr(self.algo, 'get_pareto_front'):
                    pareto_solutions, pareto_objectives = self.algo.get_pareto_front()
                    pareto_data = []
                    for i, (sol, obj) in enumerate(zip(pareto_solutions, pareto_objectives)):
                        cost, distance, time_val = obj
                        pareto_data.append({
                            "solution_id": i,
                            "cost": float(cost),
                            "distance": float(distance),
                            "time": float(time_val)
                        })

                    pareto_path = os.path.join(
                        self.output_dir, "pareto_front.json")
                    with open(pareto_path, 'w', encoding='utf-8') as f:
                        json.dump(pareto_data, f, indent=2)
                    log_info("Pareto front saved: %s", pareto_path)

                    run_summary["pareto_front_size"] = len(pareto_solutions)

                with open(os.path.join(self.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(run_summary, f, indent=2)

                log_info("Final distance: %.6f | Final cost: %.6f | Final time: %.6f",
                         total_distance, total_cost, total_time)
                return run_summary
