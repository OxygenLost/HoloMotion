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
from loguru import logger
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

from holomotion.src.modules.agent_modules import PPOActor, PPOCritic


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var + delta_mean * (mean_x - self._mean)
        )
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RolloutStorage(nn.Module):
    """Simplified rollout storage that matches rsl_rl behavior exactly."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # Core storage
        self.observations = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actor_obs_shape,
            device=self.device,
        )
        self.privileged_observations = (
            torch.zeros(
                num_transitions_per_env,
                num_envs,
                *critic_obs_shape,
                device=self.device,
            )
            if critic_obs_shape
            else None
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # PPO specific
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )
        self.sigma = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.privileged_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(
        self, last_values, gamma, lam, normalize_advantage: bool = True
    ):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + 1e-8
            )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        privileged_observations = (
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else observations
        )
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )


class PPO:
    """PPO implementation that exactly matches rsl_rl behavior."""

    def __init__(self, env, config, log_dir=None, device="cpu"):
        self.config = config
        self.env = env
        self.log_dir = log_dir
        self.device = device

        if self.log_dir:
            self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard logging enabled at: {self.log_dir}")
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

        _ = self.env.reset_all()

    def _init_config(self):
        # Environment config
        self.num_envs: int = self.env.config.num_envs
        self.num_obs = self.env.config.robot.algo_obs_dim_dict["policy"]
        self.num_privileged_obs = self.env.config.robot.algo_obs_dim_dict[
            "critic"
        ]
        self.num_actions = self.env.config.robot.actions_dim

        # Training config
        self.save_interval = self.config.save_interval
        self.log_interval = self.config.log_interval
        self.num_steps_per_env = self.config.num_steps_per_env
        self.num_learning_iterations = self.config.num_learning_iterations

        # PPO hyperparameters
        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.learning_rate = self.config.get(
            "learning_rate", self.config.actor_learning_rate
        )
        self.optimizer_type = self.config.optimizer_type
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = bool(
            self.config.get("normalize_advantage_per_mini_batch", False)
        )

        # Observation normalization
        obs_norm_cfg = self.config.get("obs_norm", {})
        self.obs_norm_enabled = obs_norm_cfg.get("enabled", False)
        self.obs_norm_epsilon = float(obs_norm_cfg.get("epsilon", 1.0e-8))

    def setup(self):
        self._setup_models_and_optimizer()
        self._setup_storage()
        self._setup_normalizers()

    def _setup_models_and_optimizer(self):
        self.obs_serializer = self.env.obs_serializer
        self.critic_obs_serializer = self.env.critic_obs_serializer
        self.actor_type = self.config.module_dict.actor.get("type", "MLP")
        self.critic_type = self.config.module_dict.critic.get("type", "MLP")

        self.actor = PPOActor(
            obs_dim_dict=self.obs_serializer,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
        ).to(self.device)

        self.critic = PPOCritic(
            obs_dim_dict=self.critic_obs_serializer,
            module_config_dict=self.config.module_dict.critic,
        ).to(self.device)

        optimizer_class = getattr(optim, self.optimizer_type)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(),
            lr=self.learning_rate,
        )
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=self.learning_rate,
        )

        logger.info("Actor:\n" + str(self.actor))
        logger.info("Critic:\n" + str(self.critic))

    def _setup_storage(self):
        self.storage = RolloutStorage(
            self.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [self.num_privileged_obs],
            [self.num_actions],
            device=self.device,
        )
        self.transition = RolloutStorage.Transition()

    def _setup_normalizers(self):
        if not self.obs_norm_enabled:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(
                self.device
            )
            return

        self.obs_normalizer = EmpiricalNormalization(
            shape=[self.num_obs], eps=self.obs_norm_epsilon, until=1.0e8
        ).to(self.device)

        self.privileged_obs_normalizer = EmpiricalNormalization(
            shape=[self.num_privileged_obs],
            eps=self.obs_norm_epsilon,
            until=1.0e8,
        ).to(self.device)

        logger.info(
            f"Observation normalizers initialized with eps: {self.obs_norm_epsilon}"
        )

    def act(self, obs, critic_obs):
        """Act function using separate actor and critic."""
        self.transition.actions = self.actor.act(obs).detach()
        self.transition.values = self.critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, time_outs, infos):
        """Process environment step that matches rsl_rl exactly."""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * time_outs.unsqueeze(1), 1
        )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute returns that matches rsl_rl exactly."""
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self):
        """Update function that matches rsl_rl exactly."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch - advantages_batch.mean()
                    ) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch
            self.actor.act(obs_batch)
            actions_log_prob_batch = self.actor.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.critic.evaluate(critic_obs_batch)
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std
            entropy_batch = self.actor.entropy

            # KL divergence
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

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            1e-5, self.learning_rate / 1.5
                        )
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            1e-2, self.learning_rate * 1.5
                        )

                    # Update both optimizers
                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    for param_group in self.critic_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch)
            )
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
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Separate actor and critic losses with auxiliary losses
            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss

            # Entropy loss
            if self.entropy_coef > 0.0:
                entropy_loss = entropy_batch.mean()
                actor_loss = actor_loss - self.entropy_coef * entropy_loss

            # Actor auxiliary losses (MoE load balancing, bound loss, etc.)
            if hasattr(self.actor.actor_module, "compute_load_balancing_loss"):
                actor_load_balancing_loss = (
                    self.actor.actor_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
                actor_loss = actor_loss + actor_load_balancing_loss

            # Critic auxiliary losses
            if hasattr(
                self.critic.critic_module, "compute_load_balancing_loss"
            ):
                critic_load_balancing_loss = (
                    self.critic.critic_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
                critic_loss = critic_loss + critic_load_balancing_loss

            # Separate gradient computations
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
            )
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )

            # Optimizer steps
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

    def learn(self):
        """Main learning loop that matches rsl_rl exactly."""
        obs_dict = self.env.reset_all()[0]
        obs = obs_dict["policy"].to(self.device)
        privileged_obs = obs_dict["critic"].to(self.device)

        self.actor.train()
        self.critic.train()

        num_learning_iterations = self.num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations

        logger.info(
            f"Starting training for {num_learning_iterations} iterations from iteration {self.current_learning_iteration}"
        )

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.act(obs, privileged_obs)

                    # Step the environment
                    obs_dict, rewards, dones, time_outs, infos = self.env.step(
                        actions
                    )
                    obs = obs_dict["policy"].to(self.device)
                    privileged_obs = obs_dict["critic"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Perform normalization
                    obs = self.obs_normalizer(obs)
                    privileged_obs = self.privileged_obs_normalizer(
                        privileged_obs
                    )

                    # Process the step
                    self.process_env_step(rewards, dones, time_outs, infos)

                    self.ep_infos.append(infos["log"])

                    # Update reward tracking
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1

                    # Handle episode completion
                    done_ids = (dones > 0).nonzero(as_tuple=False)
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
                    self.cur_reward_sum[done_ids] = 0
                    self.cur_episode_length[done_ids] = 0

                # Compute returns
                self.compute_returns(privileged_obs)

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Update policy
            loss_dict = self.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging
            if it % self.log_interval == 0:
                self._log(locals())

            # Saving
            if it % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )

            self.ep_infos.clear()

        self.save(
            os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt"
            )
        )
        logger.info(f"Training completed. Model saved to {self.log_dir}")
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def _log(self, locs: dict):
        """Enhanced logging function with beautiful tabulate formatting."""
        if not self.tensorboard_writer:
            return

        it = locs["it"]
        loss_dict = locs["loss_dict"]
        collection_time = locs["collection_time"]
        learn_time = locs["learn_time"]

        # Episode info logging to TensorBoard
        ep_info_data = {}
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
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
                    if "/" in key:
                        self.tensorboard_writer.add_scalar(
                            key, value.item(), it
                        )
                        ep_info_data[key] = value.item()
                    else:
                        self.tensorboard_writer.add_scalar(
                            f"Episode/{key}", value.item(), it
                        )
                        ep_info_data[f"Episode/{key}"] = value.item()

        # Policy std
        mean_std = self.actor.action_std.mean()
        fps = int(
            self.num_steps_per_env
            * self.num_envs
            / (collection_time + learn_time)
        )

        # Loss logging to TensorBoard
        for key, value in loss_dict.items():
            self.tensorboard_writer.add_scalar(f"Loss/{key}", value, it)

        self.tensorboard_writer.add_scalar(
            "Loss/learning_rate", self.learning_rate, it
        )
        self.tensorboard_writer.add_scalar(
            "Policy/mean_noise_std", mean_std.item(), it
        )
        self.tensorboard_writer.add_scalar("Perf/total_fps", fps, it)
        self.tensorboard_writer.add_scalar(
            "Perf/collection_time", collection_time, it
        )
        self.tensorboard_writer.add_scalar(
            "Perf/learning_time", learn_time, it
        )

        # Training metrics to TensorBoard
        if len(self.rewbuffer) > 0:
            self.tensorboard_writer.add_scalar(
                "Train/mean_reward", statistics.mean(self.rewbuffer), it
            )
            self.tensorboard_writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(self.lenbuffer),
                it,
            )

        # Beautiful console logging with tabulate
        self._post_epoch_logging(
            {
                "it": it,
                "total_learning_iterations": self.num_learning_iterations,
                "loss_dict": loss_dict,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "ep_infos": self.ep_infos,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "mean_std": mean_std.item(),
                "fps": fps,
                "learning_rate": self.learning_rate,
            }
        )

    def _post_epoch_logging(self, log_dict):
        """Beautiful console logging with tabulate formatting."""
        # Episode info processing
        ep_metrics = {}
        if log_dict["ep_infos"]:
            for key in log_dict["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict["ep_infos"]:
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
                    if "/" in key:
                        ep_metrics[key] = f"{value:.4f}"
                    else:
                        ep_metrics[f"Mean Episode {key}"] = f"{value:.4f}"

        # Build training data dictionary
        training_data = {
            "Learning Iteration": f"{log_dict['it']}/{log_dict['total_learning_iterations']}",
            "FPS": f"{log_dict['fps']:.0f} steps/s",
            "Collection Time": f"{log_dict['collection_time']:.3f}s",
            "Learning Time": f"{log_dict['learn_time']:.3f}s",
            "Mean Action Noise Std": f"{log_dict['mean_std']:.2f}",
            "Learning Rate": f"{log_dict['learning_rate']:.4e}",
        }

        # Add reward and episode info
        if len(log_dict["rewbuffer"]) > 0:
            training_data["Mean Episode Reward"] = (
                f"{statistics.mean(log_dict['rewbuffer']):.2f}"
            )
            training_data["Mean Episode Length"] = (
                f"{statistics.mean(log_dict['lenbuffer']):.2f}"
            )

        # Add loss data
        training_data.update(
            {
                k: f"{v:.4f}" if isinstance(v, (int, float)) else f"{v:.4f}"
                for k, v in log_dict["loss_dict"].items()
                if v is not None
            }
        )

        # Add episode metrics
        training_data.update(ep_metrics)

        # Organize and display
        table_data = self._organize_training_data(training_data)
        log_lines = [
            "\n" + "=" * 80,
            f"TRAINING LOG - Iteration {log_dict['it']}/{log_dict['total_learning_iterations']}",
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
            "Learning Rate",
            "",  # separator
        ]

        # Create organized list
        organized_data = []
        used_keys = set()

        # Helper function to add section header
        def add_section_header(title):
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
                    "Learning Rate",
                ]:
                    add_section_header("Model")
                    current_section = "model"

                organized_data.append([key, training_data[key]])
                used_keys.add(key)

        loss_keys = sorted(
            [
                k
                for k in training_data.keys()
                if k in ["value_function", "surrogate", "entropy"]
                and k not in used_keys
            ]
        )
        if loss_keys:
            add_section_header("Loss")
            for key in loss_keys:
                display_key = f"Loss/{key}"
                organized_data.append([display_key, training_data[key]])
                used_keys.add(key)

        remaining_keys = sorted(
            [k for k in training_data.keys() if k not in used_keys]
        )
        if remaining_keys:
            add_section_header("Other Metrics")
            for key in remaining_keys:
                organized_data.append([key, training_data[key]])

        return organized_data

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
        """Load checkpoint with your original functionality."""
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)

            # Handle both old and new checkpoint formats
            if "actor_model_state_dict" in loaded_dict:
                # Separate actor/critic format (preferred)
                actor_state = self._clean_state_dict(
                    loaded_dict["actor_model_state_dict"]
                )
                critic_state = self._clean_state_dict(
                    loaded_dict["critic_model_state_dict"]
                )

                self.actor.load_state_dict(actor_state, strict=True)
                self.critic.load_state_dict(critic_state, strict=True)

                # Load optimizers
                if "actor_optimizer_state_dict" in loaded_dict:
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["actor_optimizer_state_dict"]
                    )
                if "critic_optimizer_state_dict" in loaded_dict:
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["critic_optimizer_state_dict"]
                    )

                self.current_learning_iteration = loaded_dict.get("iter", 0)
            elif "model_state_dict" in loaded_dict:
                # rsl_rl format (single policy)
                cleaned_state_dict = self._clean_state_dict(
                    loaded_dict["model_state_dict"]
                )

                # Split into actor and critic parts
                actor_state = {}
                critic_state = {}
                for key, value in cleaned_state_dict.items():
                    if key.startswith("actor."):
                        actor_state[key[6:]] = value
                    elif key.startswith("critic."):
                        critic_state[key[7:]] = value

                if actor_state:
                    self.actor.load_state_dict(actor_state, strict=False)
                if critic_state:
                    self.critic.load_state_dict(critic_state, strict=False)

                # Load optimizer
                if "optimizer_state_dict" in loaded_dict:
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["optimizer_state_dict"]
                    )
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["optimizer_state_dict"]
                    )

                self.current_learning_iteration = loaded_dict.get("iter", 0)

            # Load normalizers if present
            if self.obs_norm_enabled:
                if "obs_norm_state_dict" in loaded_dict and hasattr(
                    self, "obs_normalizer"
                ):
                    self.obs_normalizer.load_state_dict(
                        loaded_dict["obs_norm_state_dict"]
                    )
                if "privileged_obs_norm_state_dict" in loaded_dict and hasattr(
                    self, "privileged_obs_normalizer"
                ):
                    self.privileged_obs_normalizer.load_state_dict(
                        loaded_dict["privileged_obs_norm_state_dict"]
                    )

            return loaded_dict.get("infos", None)

    def save(self, path, infos=None):
        """Save checkpoint with your original functionality."""
        logger.info(f"Saving checkpoint to {path}")

        save_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "critic_model_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        if self.obs_norm_enabled and hasattr(self, "obs_normalizer"):
            save_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        if self.obs_norm_enabled and hasattr(
            self, "privileged_obs_normalizer"
        ):
            save_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )

        torch.save(save_dict, path)

    @property
    def inference_model(self):
        """Return the separate actor and critic for inference."""
        return {
            "actor": self.actor,
            "critic": self.critic,
        }
