# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import os
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from loguru import logger
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from holomotion.src.modules.agent_modules import PPOActor, PPOCritic


class PPO:
    def __init__(
        self,
        env,
        config,
        log_dir=None,
        device="cpu",
    ):
        self.config = config
        self.use_accelerate = config.use_accelerate
        if self.use_accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                self.is_main_process = torch.distributed.get_rank() == 0
                self.process_rank = torch.distributed.get_rank()
            else:
                self.is_main_process = self.accelerator.is_main_process
                self.process_rank = self.accelerator.process_index
        else:
            self.device = device
            self.is_main_process = True
            self.process_rank = 0

        self.env = env
        self.log_dir = log_dir

        # Only initialize TensorBoard on the main process
        if self.is_main_process:
            # Initialize TensorBoard SummaryWriter
            if self.log_dir:
                self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
                logger.info(f"TensorBoard logging enabled at: {self.log_dir}")
            else:
                self.tensorboard_writer = None
                logger.warning(
                    "No log directory provided, TensorBoard logging disabled"
                )
        else:
            self.tensorboard_writer = None

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        # Manual episode tracking (RSL-RL style)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        # Always use scalar value function
        logger.info("Using scalar value function")

        if getattr(self.env, "obs_serializer", None) is not None:
            self.obs_serializer = self.env.obs_serializer
        else:
            self.obs_serializer = None

        if getattr(self.env, "critic_obs_serializer", None) is not None:
            self.critic_obs_serializer = self.env.critic_obs_serializer
        else:
            self.critic_obs_serializer = None

        self.actor_type = self.config.module_dict.get("actor", {}).get(
            "type", "MLP"
        )
        self.critic_type = self.config.module_dict.get("critic", {}).get(
            "type", "MLP"
        )

        logger.info(f"Actor type: {self.actor_type}")
        logger.info(f"Critic type: {self.critic_type}")

        self.save_interval = self.config.save_interval
        self.log_interval = self.config.log_interval
        self.num_steps_per_env = self.config.num_steps_per_env
        self.load_optimizer = self.config.load_optimizer
        self.num_learning_iterations = self.config.num_learning_iterations

        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss

        # Optimizer configuration
        self.optimizer_type = self.config.get(
            "optimizer_type", "adamw"
        ).lower()
        if self.optimizer_type not in ["adam", "adamw"]:
            logger.warning(
                f"Invalid optimizer_type '{self.optimizer_type}', defaulting to 'adamw'"
            )
            self.optimizer_type = "adamw"
        logger.info(f"Using optimizer: {self.optimizer_type.upper()}")

    def setup(self):
        self._setup_models_and_optimizer()
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        self.actor = PPOActor(
            obs_dim_dict=self.obs_serializer,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
        ).to(self.device)

        critic_config = self.config.module_dict.critic.copy()
        self.critic = PPOCritic(
            obs_dim_dict=self.critic_obs_serializer,
            module_config_dict=critic_config,
        ).to(self.device)

        logger.info("Actor:\n" + str(self.actor))
        logger.info("Critic:\n" + str(self.critic))

        optimizer_class = (
            optim.AdamW if self.optimizer_type == "adamw" else optim.Adam
        )

        self.actor_optimizer = optimizer_class(
            self.actor.parameters(), lr=self.actor_learning_rate
        )

        self.critic_optimizer = optimizer_class(
            self.critic.parameters(), lr=self.critic_learning_rate
        )

        self._log_model_summary(self.actor, "Actor")
        self._log_model_summary(self.critic, "Critic")

    def _setup_storage(self):
        self.storage = RolloutStorage(
            self.env.num_envs, self.num_steps_per_env, device=self.device
        )
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(
                obs_key, shape=(obs_dim,), dtype=torch.float
            )

        ## Register others
        self.storage.register_key(
            "actions", shape=(self.num_act,), dtype=torch.float
        )
        self.storage.register_key("rewards", shape=(1,), dtype=torch.float)
        self.storage.register_key("dones", shape=(1,), dtype=torch.bool)

        # Register scalar value function storage
        self.storage.register_key("values", shape=(1,), dtype=torch.float)
        self.storage.register_key("returns", shape=(1,), dtype=torch.float)
        self.storage.register_key("advantages", shape=(1,), dtype=torch.float)

        logger.info("Registered scalar value function storage")
        self.storage.register_key(
            "actions_log_prob", shape=(1,), dtype=torch.float
        )
        self.storage.register_key(
            "action_mean", shape=(self.num_act,), dtype=torch.float
        )
        self.storage.register_key(
            "action_sigma", shape=(self.num_act,), dtype=torch.float
        )

    def _eval_mode(self):
        # Handle both DDP-wrapped and normal models
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.eval()
        critic = (
            self.critic.module
            if hasattr(self.critic, "module")
            else self.critic
        )
        critic.eval()

    def _train_mode(self):
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.train()
        critic = (
            self.critic.module
            if hasattr(self.critic, "module")
            else self.critic
        )
        critic.train()

    @staticmethod
    def _clean_state_dict(state_dict):
        """Remove the '_orig_mod.' prefix from keys if it exists."""
        cleaned_dict = {}
        prefix = "_orig_mod."
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k.startswith(prefix):
                cleaned_dict[k[prefix_len:]] = v
            else:
                cleaned_dict[k] = v
        return cleaned_dict

    def load(self, ckpt_path):
        self.ckpt_path = ckpt_path

        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)

            # Clean the state dicts to remove potential compilation prefixes
            cleaned_actor_state_dict = self._clean_state_dict(
                loaded_dict["actor_model_state_dict"]
            )

            if "critic_model_state_dict" in loaded_dict:
                cleaned_critic_state_dict = self._clean_state_dict(
                    loaded_dict["critic_model_state_dict"]
                )
            else:
                logger.warning(
                    "critic_model_state_dict not found in checkpoint. Skipping critic model loading !"
                )

            if self.use_accelerate and hasattr(self, "accelerator"):
                self.accelerator.unwrap_model(self.actor).load_state_dict(
                    cleaned_actor_state_dict, strict=True
                )
                if "critic_model_state_dict" in loaded_dict:
                    self.accelerator.unwrap_model(self.critic).load_state_dict(
                        cleaned_critic_state_dict, strict=True
                    )
                logger.info(
                    "Strict loading of actor and critic state dicts successful !"
                )
            else:
                self.actor.load_state_dict(
                    cleaned_actor_state_dict, strict=True
                )
                if "critic_model_state_dict" in loaded_dict:
                    self.critic.load_state_dict(
                        cleaned_critic_state_dict, strict=True
                    )
                    logger.info(
                        "Strict loading of actor and critic state dicts successful."
                    )

            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(
                    loaded_dict["actor_optimizer_state_dict"]
                )
                if "critic_optimizer_state_dict" in loaded_dict:
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["critic_optimizer_state_dict"]
                    )

                self.actor_learning_rate = loaded_dict[
                    "actor_optimizer_state_dict"
                ]["param_groups"][0]["lr"]

                logger.info("Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(
                    f"Critic Learning rate: {self.critic_learning_rate}"
                )

            self.current_learning_iteration = loaded_dict["iter"]

            return loaded_dict.get("infos", None)

    def save(self, path, infos=None):
        if not self.is_main_process:
            return

        logger.info(f"Saving checkpoint to {path}")

        # Get unwrapped model state dict if using accelerate
        actor_state = (
            self.accelerator.unwrap_model(self.actor).state_dict()
            if (self.use_accelerate and hasattr(self, "accelerator"))
            else self.actor.state_dict()
        )
        critic_state = (
            self.accelerator.unwrap_model(self.critic).state_dict()
            if (self.use_accelerate and hasattr(self, "accelerator"))
            else self.critic.state_dict()
        )

        save_dict = {
            "actor_model_state_dict": actor_state,
            "critic_model_state_dict": critic_state,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        torch.save(save_dict, path)

    def learn(self):
        obs_dict = self.env.reset_all()[
            0
        ]  # isaaclab obs is a tuple of length 2 : ({'actor':...,'critic':...}, {'log':{'episode_rew':..., 'episode_len':...}})
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # Initialize distributed barrier for synchronized training
        if self.use_accelerate and hasattr(self, "accelerator"):
            self.accelerator.wait_for_everyone()

        if self.is_main_process:
            logger.info(
                f"Starting training for {num_learning_iterations} iterations "
                f"from iteration {self.current_learning_iteration}"
            )

        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time()

            obs_dict = self._rollout_step(obs_dict)

            end_of_rollout_time = time.time()
            self.collection_time = end_of_rollout_time - self.start_time

            loss_dict = self._training_step()

            end_of_training_time = time.time()
            self.learn_time = end_of_training_time - end_of_rollout_time

            self.current_learning_iteration = it

            log_dict = {
                "it": it,
                "loss_dict": loss_dict,
                "collection_time": self.collection_time,
                "learn_time": self.learn_time,
                "ep_infos": self.ep_infos,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "entropy_coef": self.entropy_coef,
                "num_learning_iterations": num_learning_iterations,
                "total_learning_iterations": tot_iter,
            }

            if self.is_main_process:
                if it % self.log_interval == 0:
                    self._post_epoch_logging(log_dict)
                if it % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )
            self.ep_infos.clear()

            # Synchronize processes after each iteration
            if self.use_accelerate and hasattr(self, "accelerator"):
                self.accelerator.wait_for_everyone()

                if it % 10 == 0 and self.is_main_process:
                    # Check if we're actually in distributed mode
                    if torch.distributed.is_initialized():
                        world_size = torch.distributed.get_world_size()
                        if world_size > 1:
                            logger.info(
                                f"🔗 Distributed Training Active: {world_size} processes synchronized"
                            )
                    else:
                        logger.warning(
                            "⚠️  PyTorch Distributed not initialized!"
                        )

        if self.is_main_process:
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            logger.info(f"Training completed. Model saved to {self.log_dir}")
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()

    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        with torch.no_grad():
            actions = self._actor_act_step(obs_dict)

            if hasattr(self.actor, "module"):
                action_mean = self.actor.module.action_mean.detach()
                action_sigma = self.actor.module.action_std.detach()
            else:
                action_mean = self.actor.action_mean.detach()
                action_sigma = self.actor.action_std.detach()

            if hasattr(self.actor, "module"):
                actions_log_prob = (
                    self.actor.module.get_actions_log_prob(actions)
                    .detach()
                    .unsqueeze(1)
                )
            else:
                actions_log_prob = (
                    self.actor.get_actions_log_prob(actions)
                    .detach()
                    .unsqueeze(1)
                )

        policy_state_dict["actions"] = actions
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict

    def _get_inference_policy(self, device=None):
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            actor.to(device)
        return actor.act_inference

    def _rollout_step(self, obs_dict):
        with torch.no_grad():
            for _ in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(
                    obs_dict, policy_state_dict
                )
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                obs_dict, rewards, dones, time_outs, infos = self.env.step(
                    actor_state["actions"]
                )

                rewards_stored = rewards.unsqueeze(1)  # Shape [num_envs, 1]

                if "time_outs" in infos:
                    timeout_value_bonus = (
                        self.gamma
                        * policy_state_dict["values"]
                        * infos["time_outs"].unsqueeze(1).to(self.device)
                    )
                    rewards_stored += timeout_value_bonus
                assert len(rewards_stored.shape) == 2
                self.storage.update_key("rewards", rewards_stored)
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.increment_step()

                # Episode logging following RSL-RL approach
                if self.is_main_process:
                    # Collect episode info from IsaacLab - check both "episode" and "log"
                    if "episode" in infos:
                        self.ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        self.ep_infos.append(infos["log"])

                    # Update rewards and episode lengths (RSL-RL style)
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1

                    # Handle episode completion - align with RSL-RL approach
                    done_ids = (dones > 0).nonzero(as_tuple=False)
                    # Add completed episodes to buffers (RSL-RL style - no conditional check)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[done_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[done_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    # Reset tracking for completed episodes
                    self.cur_reward_sum[done_ids] = 0
                    self.cur_episode_length[done_ids] = 0

            compute_returns_dict = dict(
                values=self.storage.query_key("values"),
                dones=self.storage.query_key("dones"),
                rewards=self.storage.query_key("rewards"),
            )

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=compute_returns_dict,
            )
            self.storage.batch_update_data("returns", returns)
            self.storage.batch_update_data("advantages", advantages)

        return obs_dict

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        last_values = self._critic_eval_step(last_obs_dict).detach()

        values = policy_state_dict["values"]
        dones = policy_state_dict["dones"]
        rewards = policy_state_dict["rewards"]

        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)

        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]

        # Scalar computation
        advantage = 0

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - values[step]
            )

            advantage = (
                delta
                + next_is_not_terminal * self.gamma * self.lam * advantage
            )
            returns[step] = advantage + values[step]

        # Compute the advantages
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        return returns, advantages

    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for policy_state_dict in generator:
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            if loss_dict[key] is not None:
                loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        # Core PPO losses - always used
        loss_dict["Value"] = 0
        loss_dict["Surrogate"] = 0

        # Conditional losses - initialize with None, update only if used
        loss_dict["Entropy"] = None
        loss_dict["Actor_Load_Balancing_Loss"] = None
        loss_dict["Critic_Load_Balancing_Loss"] = None
        loss_dict["Bound_Loss"] = None
        loss_dict["KL_Mean"] = None
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _extract_command_metrics_from_episodes(self, ep_infos):
        """Extract command metrics from episode info for consistent logging."""
        command_log_dict = {}

        if ep_infos:
            # Look for command-related metrics in episode info
            for key in ep_infos[0]:
                # Only include metrics that are specifically command metrics from IsaacLab's command manager
                # These come from the command manager's reset() method and have the format "Metrics/{command_name}/{metric_name}"
                if key.startswith("Metrics/") and any(
                    cmd_name in key for cmd_name in ["ref_motion", "command"]
                ):
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in ep_infos:
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat(
                            (infotensor, ep_info[key].to(self.device))
                        )

                    if infotensor.numel() > 0:
                        value = torch.mean(infotensor)
                        # Clean up the key format for better display
                        clean_key = key.replace("Metrics/", "").replace(
                            "ref_motion/", ""
                        )
                        formatted_key = f"Command/{clean_key}"
                        command_log_dict[formatted_key] = value

        return command_log_dict

    def _actor_act_step(self, obs_dict):
        if self.use_accelerate and hasattr(self.actor, "module"):
            return self.actor.module.act(obs_dict["actor_obs"])
        else:
            return self.actor.act(obs_dict["actor_obs"])

    def _actor_act_inference_step(self, obs_dict):
        if self.use_accelerate and hasattr(self.actor, "module"):
            return self.actor.module.act_inference(obs_dict["actor_obs"])
        else:
            return self.actor.act_inference(obs_dict["actor_obs"])

    def _critic_eval_step(self, obs_dict):
        if self.use_accelerate and hasattr(self.critic, "module"):
            values = self.critic.module.evaluate(obs_dict["critic_obs"])
        else:
            values = self.critic.evaluate(obs_dict["critic_obs"])

        # Ensure proper 2D scalar output [batch_size, 1]
        if values.dim() == 1:
            # If critic returns 1D [batch_size], convert to [batch_size, 1]
            values = values.unsqueeze(-1)
        elif values.dim() == 2 and values.size(-1) == 1:
            # Already correct shape [batch_size, 1]
            pass
        elif values.dim() == 2 and values.size(-1) > 1:
            # If critic outputs multiple values, sum them for scalar output
            values = values.sum(dim=-1, keepdim=True)
        elif values.dim() > 2:
            # Handle higher dimensional outputs by flattening
            values = values.view(values.size(0), -1)
            if values.size(-1) > 1:
                values = values.sum(dim=-1, keepdim=True)

        # Final assertion for scalar mode
        assert values.dim() == 2 and values.shape[-1] == 1, (
            f"Values must have shape [batch_size, 1], got {values.shape}"
        )

        return values

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict["actions"]
        old_actions_log_prob_batch = policy_state_dict["actions_log_prob"]
        old_mu_batch = policy_state_dict["action_mean"]
        old_sigma_batch = policy_state_dict["action_sigma"]
        target_values_batch = policy_state_dict["values"]
        advantages_batch = policy_state_dict["advantages"]
        returns_batch = policy_state_dict["returns"]

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = (
            self.actor.get_actions_log_prob(actions_batch)
            if not hasattr(self.actor, "module")
            else self.actor.module.get_actions_log_prob(actions_batch)
        )
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = (
            self.actor.action_mean
            if not hasattr(self.actor, "module")
            else self.actor.module.action_mean
        )
        sigma_batch = (
            self.actor.action_std
            if not hasattr(self.actor, "module")
            else self.actor.module.action_std
        )
        entropy_batch = (
            self.actor.entropy
            if not hasattr(self.actor, "module")
            else self.actor.module.entropy
        )

        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
                if loss_dict["KL_Mean"] is None:
                    loss_dict["KL_Mean"] = kl_mean.item()
                else:
                    loss_dict["KL_Mean"] += kl_mean.item()
                lr_scaler = 1.2
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(
                        1e-6, self.actor_learning_rate / lr_scaler
                    )
                    self.critic_learning_rate = max(
                        1e-6, self.critic_learning_rate / lr_scaler
                    )
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(
                        1e-2, self.actor_learning_rate * lr_scaler
                    )
                    self.critic_learning_rate = min(
                        1e-2, self.critic_learning_rate * lr_scaler
                    )

                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = self.critic_learning_rate

        ratio = torch.exp(
            actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        )

        # Compute surrogate loss
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (
                value_batch - target_values_batch
            ).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        if "moe" in self.critic_type.lower():
            if self.use_accelerate and hasattr(self.actor, "module"):
                critic_load_balancing_loss = (
                    self.critic.module.critic_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
            else:
                critic_load_balancing_loss = (
                    self.critic.critic_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
            value_loss = value_loss + critic_load_balancing_loss

            if loss_dict["Critic_Load_Balancing_Loss"] is None:
                loss_dict["Critic_Load_Balancing_Loss"] = (
                    critic_load_balancing_loss.item()
                )
            else:
                loss_dict["Critic_Load_Balancing_Loss"] += (
                    critic_load_balancing_loss.item()
                )

        if self.entropy_coef > 0.0:
            entropy_loss = entropy_batch.mean()
            actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
            if loss_dict["Entropy"] is None:
                loss_dict["Entropy"] = entropy_loss.item()
            else:
                loss_dict["Entropy"] += entropy_loss.item()
        else:
            actor_loss = surrogate_loss

        if (
            "moe" in self.actor_type.lower()
            or self.actor_type == "EstVAEStudent"
        ):
            if self.use_accelerate and hasattr(self.actor, "module"):
                load_balancing_loss = (
                    self.actor.module.actor_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
            else:
                load_balancing_loss = (
                    self.actor.actor_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
            actor_loss = actor_loss + load_balancing_loss
            if loss_dict["Actor_Load_Balancing_Loss"] is None:
                loss_dict["Actor_Load_Balancing_Loss"] = (
                    load_balancing_loss.item()
                )
            else:
                loss_dict["Actor_Load_Balancing_Loss"] += (
                    load_balancing_loss.item()
                )

        if self.use_accelerate and hasattr(self.actor, "module"):
            bound_loss = (
                self.actor.module.actor_module.compute_bound_loss()
                * self.config.get("bound_loss_alpha", 1.0)
            )
        else:
            bound_loss = (
                self.actor.actor_module.compute_bound_loss()
                * self.config.get("bound_loss_alpha", 1.0)
            )
        actor_loss = actor_loss + bound_loss

        if loss_dict["Bound_Loss"] is None:
            loss_dict["Bound_Loss"] = bound_loss.item()
        else:
            loss_dict["Bound_Loss"] += bound_loss.item()

        critic_loss = self.value_loss_coef * value_loss
        actor_critic_loss = actor_loss + critic_loss

        if self.use_accelerate and hasattr(self, "accelerator"):
            self.accelerator.backward(actor_critic_loss)
        else:
            actor_critic_loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict["Value"] += value_loss.item()
        loss_dict["Surrogate"] += surrogate_loss.item()

        return loss_dict

    @property
    def inference_model(self):
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        critic = (
            self.critic.module
            if hasattr(self.critic, "module")
            else self.critic
        )
        return {"actor": actor, "critic": critic}

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    @torch.no_grad()
    def get_example_obs(self):
        """Get example observations from environment reset.

        Returns:
            Dictionary of example observations moved to CPU
        """
        obs_dict = self.env.reset_all()[0]  # IsaacLab returns tuple
        for obs_key in obs_dict.keys():
            print(obs_key, obs_dict[obs_key].shape)
        # Move to CPU
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps=1000):
        """Evaluate the trained policy using deterministic actions.

        This function runs evaluation for the specified number of steps using
        mean actions (no sampling) from the policy. No learning is performed.

        Args:
            max_eval_steps: Number of evaluation steps to run

        Returns:
            Dictionary of evaluation metrics if on main process, None otherwise
        """
        logger.info("Starting policy evaluation...")

        # Check for compilation disable environment variable
        import os

        if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1":
            logger.info(
                "TORCH_COMPILE_DISABLE=1 detected - compilation should be disabled"
            )

        # Provide helpful information about common compilation issues
        logger.info("Note: If you encounter Triton compilation errors, try:")
        logger.info("  export TORCH_COMPILE_DISABLE=1")
        logger.info(
            "  or disable torch.compile in your environment configuration"
        )

        # Setup evaluation mode
        self._eval_mode()
        if hasattr(self.env, "set_is_evaluating"):
            self.env.set_is_evaluating()
        if hasattr(self.env, "is_evaluating"):
            self.env.is_evaluating = True
        if hasattr(self.env, "resample_motion"):
            self.env.resample_motion()

        # Try to disable torch compilation for evaluation to avoid Triton issues
        compilation_disabled = False
        if hasattr(torch, "_dynamo"):
            try:
                # Reset dynamo cache and disable compilation temporarily
                torch._dynamo.reset()
                # Set backend to eager mode to disable compilation
                original_backend = torch._dynamo.optimize("eager")
                compilation_disabled = True
                logger.info(
                    "Disabled torch.compile for evaluation to avoid compilation issues"
                )
            except Exception as e:
                logger.warning(f"Could not disable torch.compile: {e}")

        # Store original compilation state
        self._eval_compilation_disabled = compilation_disabled

        # Get inference policy that uses mean actions (no sampling)
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.eval()

        # Ensure motion library uses first frames for evaluation
        if hasattr(self.env, "_motion_lib"):
            logger.info(
                "Setting up motion library for evaluation (first frame sampling)"
            )
            # Resample motions for evaluation - this should use eval=True internally
            if hasattr(self.env._motion_lib, "resample_new_motions"):
                # Force resample with eval=True to start from first frames
                self.env._motion_lib.resample_new_motions(
                    num_samples=self.env.num_envs, eval=True
                )

            # Make sure any cached start frames use first frames
            if hasattr(self.env._motion_lib, "cache") and hasattr(
                self.env._motion_lib.cache, "sample_cached_global_start_frames"
            ):
                env_ids = torch.arange(self.env.num_envs, device=self.device)
                first_frames = self.env._motion_lib.cache.sample_cached_global_start_frames(
                    env_ids, eval=True
                )
                logger.info(
                    f"Sampled first frames for evaluation: {first_frames[:5]}..."
                )

        # Initialize environment with eval mode - this should start from first frames
        obs_dict = self.env.reset_all()[0]  # IsaacLab returns tuple
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        logger.info(
            "Environment initialized for evaluation starting from first frames"
        )

        # Auto-detect evaluation length from motion library if available
        if hasattr(self.env, "_motion_lib") and hasattr(
            self.env._motion_lib, "cache"
        ):
            try:
                detected_steps = getattr(
                    self.env._motion_lib.cache,
                    "max_frame_length",
                    max_eval_steps,
                )
                max_eval_steps = detected_steps
                logger.info(f"Detected max frame length: {max_eval_steps}")
            except:
                logger.info(
                    f"Using default evaluation steps: {max_eval_steps}"
                )

        logger.info(f"Running evaluation for {max_eval_steps} steps...")

        # Initialize episode tracking
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = torch.zeros(
            self.env.num_envs, device=self.device
        )
        current_episode_length = torch.zeros(
            self.env.num_envs, device=self.device
        )

        # Main evaluation loop
        for step in range(max_eval_steps):
            # Get deterministic actions (mean of policy distribution)
            actions = actor.act_inference(obs_dict["actor_obs"])

            # Step environment with error handling for compilation issues
            try:
                obs_dict, rewards, dones, time_outs, infos = self.env.step(
                    actions
                )
            except Exception as e:
                error_msg = str(e).lower()
                if any(
                    keyword in error_msg
                    for keyword in [
                        "triton",
                        "compilation",
                        "inductor",
                        "dynamo",
                    ]
                ):
                    logger.error(
                        f"PyTorch compilation error during evaluation at step {step}"
                    )
                    logger.error(f"Error details: {type(e).__name__}: {e}")
                    logger.info("=" * 80)
                    logger.info("DEBUGGING SUGGESTIONS:")
                    logger.info(
                        "1. This is likely a torch.compile compatibility issue with IsaacLab rewards"
                    )
                    logger.info(
                        "2. Try setting environment variable: TORCH_COMPILE_DISABLE=1"
                    )
                    logger.info(
                        "3. Or disable compilation in your config/environment setup"
                    )
                    logger.info(
                        "4. Check if your PyTorch/Triton versions are compatible"
                    )
                    logger.info(
                        "5. Consider running evaluation without torch.compile"
                    )
                    logger.info("=" * 80)

                    # Attempt recovery
                    if hasattr(torch, "_dynamo"):
                        logger.info(
                            "Attempting to recover by resetting compilation cache..."
                        )
                        try:
                            torch._dynamo.reset()
                            torch._dynamo.config.suppress_errors = True
                            obs_dict, rewards, dones, time_outs, infos = (
                                self.env.step(actions)
                            )
                            logger.info(
                                "✅ Successfully recovered after cache reset"
                            )
                        except Exception as e2:
                            logger.error(f"❌ Recovery failed: {e2}")
                            logger.error(
                                "Evaluation cannot continue due to compilation issues."
                            )
                            logger.error(
                                "Please disable torch.compile for evaluation."
                            )
                            raise e2
                    else:
                        logger.error(
                            "Cannot attempt recovery - torch._dynamo not available"
                        )
                        raise e
                else:
                    logger.error(
                        f"Unexpected error during environment step at step {step}: {e}"
                    )
                    raise e

            # Move observations to device
            for obs_key in obs_dict.keys():
                obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            # Update episode tracking (only on main process)
            if self.is_main_process:
                current_episode_reward += rewards
                current_episode_length += 1

                # Handle episode completions
                done_mask = dones.bool()
                if done_mask.any():
                    # Collect completed episode stats
                    completed_rewards = current_episode_reward[done_mask]
                    completed_lengths = current_episode_length[done_mask]

                    episode_rewards.extend(completed_rewards.cpu().tolist())
                    episode_lengths.extend(completed_lengths.cpu().tolist())

                    # Reset completed episodes
                    current_episode_reward[done_mask] = 0.0
                    current_episode_length[done_mask] = 0.0

            # Log progress periodically
            if step % 100 == 0 and self.is_main_process:
                logger.info(f"Evaluation step {step}/{max_eval_steps}")

        # Compute final metrics (only on main process)
        eval_metrics = None
        if self.is_main_process:
            if len(episode_rewards) > 0:
                eval_metrics = {
                    "mean_episode_reward": sum(episode_rewards)
                    / len(episode_rewards),
                    "mean_episode_length": sum(episode_lengths)
                    / len(episode_lengths),
                    "num_completed_episodes": len(episode_rewards),
                    "total_evaluation_steps": max_eval_steps,
                }

                logger.info("Evaluation Results:")
                logger.info(
                    f"  Mean Episode Reward: {eval_metrics['mean_episode_reward']:.4f}"
                )
                logger.info(
                    f"  Mean Episode Length: {eval_metrics['mean_episode_length']:.2f}"
                )
                logger.info(
                    f"  Completed Episodes: {eval_metrics['num_completed_episodes']}"
                )
                logger.info(
                    f"  Total Steps: {eval_metrics['total_evaluation_steps']}"
                )
            else:
                logger.warning("No episodes completed during evaluation")
                eval_metrics = {
                    "mean_episode_reward": 0.0,
                    "mean_episode_length": 0.0,
                    "num_completed_episodes": 0,
                    "total_evaluation_steps": max_eval_steps,
                }

        # Cleanup: return to training mode
        self._train_mode()
        if hasattr(self.env, "is_evaluating"):
            self.env.is_evaluating = False
        if hasattr(self.env, "_init_buffers"):
            self.env._init_buffers()
        if hasattr(self.env, "resample_motion"):
            self.env.resample_motion()

        # Restore torch compilation state if it was disabled
        if (
            hasattr(self, "_eval_compilation_disabled")
            and self._eval_compilation_disabled
        ):
            try:
                if hasattr(torch, "_dynamo"):
                    torch._dynamo.reset()
                    logger.info(
                        "Restored torch.compile state after evaluation"
                    )
            except Exception as e:
                logger.warning(f"Could not restore torch.compile state: {e}")

        # Reset episode tracking
        self.cur_reward_sum.zero_()
        self.cur_episode_length.zero_()

        # Re-initialize environment for training
        obs_dict = self.env.reset_all()[0]
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        if self.is_main_process:
            logger.info("Policy evaluation completed successfully!")

        return eval_metrics

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        # Skip logging if not the main process
        if not self.is_main_process:
            return

        if log_dict["ep_infos"]:
            for key in log_dict["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos (RSL-RL style)
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                if (
                    self.is_main_process
                    and self.tensorboard_writer is not None
                ):
                    # RSL-RL style logging - handle pre-formatted keys with "/"
                    if "/" in key:
                        self.tensorboard_writer.add_scalar(
                            key, value.item(), log_dict["it"]
                        )
                    else:
                        self.tensorboard_writer.add_scalar(
                            f"Episode/{key}", value.item(), log_dict["it"]
                        )

        train_log_dict = {}
        actor_model = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        mean_std = actor_model.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (log_dict["collection_time"] + log_dict["learn_time"])
        )
        train_log_dict["fps"] = fps
        train_log_dict["mean_std"] = mean_std.item()

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        # Extract command metrics from episode info for consistent logging
        command_log_dict = self._extract_command_metrics_from_episodes(
            log_dict["ep_infos"]
        )

        self._logging_to_writer(
            log_dict, train_log_dict, env_log_dict, command_log_dict
        )

        training_data = {
            "Learning Iteration": f"{log_dict['it']}/{log_dict['total_learning_iterations']}",  # noqa: E501
            "FPS": f"{train_log_dict['fps']:.0f} steps/s",
            "Collection Time": f"{log_dict['collection_time']:.3f}s",
            "Learning Time": f"{log_dict['learn_time']:.3f}s",
            "Mean Action Noise Std": f"{train_log_dict['mean_std']:.2f}",
            "Entropy Coef": f"{self.entropy_coef:.4e}",
        }

        # Always show reward and episode info
        if len(log_dict["rewbuffer"]) > 0:
            # Show completed episodes stats
            training_data["Mean Episode Reward"] = (
                f"{statistics.mean(log_dict['rewbuffer']):.2f}"
            )
            training_data["Mean Episode Length"] = (
                f"{statistics.mean(log_dict['lenbuffer']):.2f}"
            )

        # Add environment log data
        for k, v in env_log_dict.items():
            key_name = k.replace("Env/", "")  # Clean up key names
            training_data[key_name] = f"{v:.4f}"

        # Add command metrics data
        for k, v in command_log_dict.items():
            key_name = k.replace("Command/", "")  # Clean up key names
            if isinstance(v, torch.Tensor):
                training_data[key_name] = f"{v.item():.4f}"
            else:
                training_data[key_name] = f"{v:.4f}"

        training_data.update(
            {
                k: f"{v:.4f}" if isinstance(v, torch.Tensor) else f"{v:.4f}"
                for k, v in log_dict["loss_dict"].items()
                if v is not None  # Filter out None values
            }
        )

        if log_dict["ep_infos"]:
            for key in log_dict["ep_infos"][0]:
                # Skip command-related metrics since they're handled separately now
                if key.startswith("Metrics/"):
                    continue

                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos (RSL-RL style)
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                # RSL-RL style console logging - handle pre-formatted keys
                if "/" in key:
                    training_data[key] = f"{value:.4f}"
                else:
                    training_data[f"Mean Episode {key}"] = f"{value:.4f}"

        # Organize table data in logical groups
        table_data = self._organize_training_data(training_data)
        log_lines = [
            "\n" + "=" * 80,
            f"TRAINING LOG - Iteration {log_dict['it']}/{log_dict['total_learning_iterations']}",  # noqa: E501
            "=" * 80,
            tabulate(
                table_data,
                headers=["Metric", "Value"],
                tablefmt="simple_outline",
            ),
            "=" * 80,
            f"Logging Directory: {os.path.abspath(self.log_dir)}",
            "=" * 80 + "\n",
        ]
        training_log = "\n".join(log_lines)
        logger.info(training_log)

    def _logging_to_writer(
        self, log_dict, train_log_dict, env_log_dict, command_log_dict
    ):
        if not self.is_main_process or self.tensorboard_writer is None:
            return

        for loss_key, loss_value in log_dict["loss_dict"].items():
            if loss_value is not None:
                self.tensorboard_writer.add_scalar(
                    f"Loss/{loss_key}", loss_value, log_dict["it"]
                )

        self.tensorboard_writer.add_scalar(
            "Loss/actor_learning_rate",
            self.actor_learning_rate,
            log_dict["it"],
        )
        self.tensorboard_writer.add_scalar(
            "Loss/critic_learning_rate",
            self.critic_learning_rate,
            log_dict["it"],
        )
        self.tensorboard_writer.add_scalar(
            "Policy/mean_noise_std", train_log_dict["mean_std"], log_dict["it"]
        )
        self.tensorboard_writer.add_scalar(
            "Perf/total_fps", train_log_dict["fps"], log_dict["it"]
        )
        self.tensorboard_writer.add_scalar(
            "Perf/collection_time", log_dict["collection_time"], log_dict["it"]
        )
        self.tensorboard_writer.add_scalar(
            "Perf/learning_time", log_dict["learn_time"], log_dict["it"]
        )

        if len(log_dict["rewbuffer"]) > 0:
            self.tensorboard_writer.add_scalar(
                "Train/mean_reward",
                statistics.mean(log_dict["rewbuffer"]),
                log_dict["it"],
            )
            self.tensorboard_writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(log_dict["lenbuffer"]),
                log_dict["it"],
            )

        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.tensorboard_writer.add_scalar(k, v, log_dict["it"])

        # Log command metrics to TensorBoard
        if len(command_log_dict) > 0:
            for k, v in command_log_dict.items():
                value = v.item() if isinstance(v, torch.Tensor) else v
                self.tensorboard_writer.add_scalar(k, value, log_dict["it"])

        self.tensorboard_writer.add_scalar(
            "Train/entropy_coef", self.entropy_coef, log_dict["it"]
        )

    def _log_model_summary(self, model, name):
        if not model:
            logger.info(f"{name}: None")
            return

        # Get total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Format parameter counts in K, M, B for readability
        def format_params(count):
            if count >= 1e9:
                return f"{count / 1e9:.2f}B"
            elif count >= 1e6:
                return f"{count / 1e6:.2f}M"
            elif count >= 1e3:
                return f"{count / 1e3:.1f}K"
            else:
                return f"{count}"

        total_params_str = format_params(total_params)
        trainable_params_str = format_params(trainable_params)

        # Get model structure overview
        if hasattr(model, "__class__"):
            model_type = model.__class__.__name__
        else:
            model_type = str(type(model))

        # Get layer info if available
        layer_info = ""
        if hasattr(model, "children"):
            top_level_modules = list(model.children())
            if len(top_level_modules) <= 5:  # Only show if not too many
                layer_info = f", Modules: {len(top_level_modules)}"

        logger.info(
            f"{name} Summary: {model_type}, Total params: {total_params_str}, "
            f"Trainable: {trainable_params_str}{layer_info}"
        )

    def _organize_training_data(self, training_data):
        """Organize training data into logical groups for better console display."""
        # Define priority order for key display
        priority_keys = [
            # Core training info (highest priority)
            "Learning Iteration",
            "FPS",
            "Collection Time",
            "Learning Time",
            "",  # separator
            # Episode statistics
            "Mean Episode Reward",
            "Mean Episode Length",
            "",  # separator
            # Model metrics
            "Mean Action Noise Std",
            "Entropy Coef",
            "",  # separator
        ]

        # Create organized list
        organized_data = []
        used_keys = set()

        # Helper function to add section header
        def add_section_header(title):
            # Create simple section header
            organized_data.append([f"=== {title.upper()} ===", "======"])

        # Add priority keys first
        current_section = None
        for key in priority_keys:
            if key == "":  # section break
                current_section = None
            elif key in training_data:
                # Add section header for performance metrics
                if current_section != "training" and key in [
                    "Learning Iteration",
                    "FPS",
                    "Collection Time",
                    "Learning Time",
                ]:
                    add_section_header("Performance")
                    current_section = "training"
                # Add section header for episode stats
                elif current_section != "episode" and key in [
                    "Mean Episode Reward",
                    "Mean Episode Length",
                ]:
                    add_section_header("Episode Statistics")
                    current_section = "episode"
                # Add section header for model metrics
                elif current_section != "model" and key in [
                    "Mean Action Noise Std",
                    "Entropy Coef",
                ]:
                    add_section_header("Model")
                    current_section = "model"

                organized_data.append([key, training_data[key]])
                used_keys.add(key)

        # Add command metrics with section header
        command_keys = sorted(
            [k for k in training_data.keys() if k.startswith("Task/")]
        )
        if command_keys:
            add_section_header("Task Metrics")
            for key in command_keys:
                organized_data.append([key, training_data[key]])
                used_keys.add(key)

        # Add environment metrics with section header
        env_keys = sorted(
            [
                k
                for k in training_data.keys()
                if k.startswith("Env/")
                or any(
                    env_indicator in k.lower()
                    for env_indicator in ["termination", "episode"]
                )
                and k not in used_keys
            ]
        )
        if env_keys:
            add_section_header("MDP Info")
            for key in env_keys:
                organized_data.append([key, training_data[key]])
                used_keys.add(key)

        # Add loss metrics with section header - automatically detect loss keys
        loss_keys = sorted(
            [
                k
                for k in training_data.keys()
                if k.startswith("Loss/") and k not in used_keys
            ]
        )
        if loss_keys:
            add_section_header("Loss")
            for key in loss_keys:
                # Add Loss/ prefix for consistency with TensorBoard
                display_key = f"Loss/{key}"
                organized_data.append([display_key, training_data[key]])
                used_keys.add(key)

        # Add any remaining keys
        remaining_keys = sorted(
            [k for k in training_data.keys() if k not in used_keys]
        )
        if remaining_keys:
            add_section_header("Other Metrics")
            for key in remaining_keys:
                organized_data.append([key, training_data[key]])

        return organized_data


class RolloutStorage(nn.Module):
    def __init__(self, num_envs, num_transitions_per_env, device="cpu"):
        super().__init__()

        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0
        self.stored_keys = list()

    def register_key(self, key: str, shape=(), dtype=torch.float):
        assert not hasattr(self, key), key
        assert isinstance(shape, (list, tuple)), (
            "shape must be a list or tuple"
        )
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs) + shape,
            dtype=dtype,
            device=self.device,
        )
        self.register_buffer(key, buffer, persistent=False)
        self.stored_keys.append(key)

    def increment_step(self):
        self.step += 1

    def update_key(self, key: str, data: torch.Tensor):
        assert not data.requires_grad
        assert self.step < self.num_transitions_per_env, (
            "Rollout buffer overflow"
        )
        getattr(self, key)[self.step].copy_(data)

    def batch_update_data(self, key: str, data: torch.Tensor):
        assert not data.requires_grad
        getattr(self, key)[:] = data

    def clear(self):
        self.step = 0

    def query_key(self, key: str):
        assert hasattr(self, key), key
        return getattr(self, key)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        _buffer_dict = {
            key: getattr(self, key)[:].flatten(0, 1)
            for key in self.stored_keys
        }

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                _batch_buffer_dict = {
                    key: _buffer_dict[key][batch_idx]
                    for key in self.stored_keys
                }
                yield _batch_buffer_dict


class TensorAverageMeter:
    def __init__(self):
        self.tensors = []

    def add(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self):
        if len(self.tensors) == 0:
            return 0
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return 0
        else:
            return cat.mean()

    def clear(self):
        self.tensors = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self):
        mean_dict = {k: v.mean() for k, v in self.data.items()}
        return mean_dict

    def clear(self):
        self.data = {}

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean
