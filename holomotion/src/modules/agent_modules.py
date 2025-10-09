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


from __future__ import annotations
from copy import deepcopy
from typing import List, Union

import torch
import torch.nn as nn
from torch.distributions import Normal

from holomotion.src.modules.network_modules import (
    MLP,
    MoEMLP,
    MoEMLPV2,
    MoEMLPVAE,
    MoEMLPVAEV2,
    EstVAEStudent,
    TFStudent,
)

from loguru import logger


class SimpleObsSerializer:
    """Simple observation serializer for flat observations."""

    def __init__(self, obs_dim):
        self.obs_dim_dict = {"obs": obs_dim}
        self.obs_seq_len_dict = {"obs": 1}
        self.obs_flat_dim = obs_dim

    def serialize(self, obs_list):
        return obs_list[0]

    def deserialize(self, obs_tensor):
        return {"obs": obs_tensor[:, None, :]}  # Add sequence dimension


class ObsSeqSerializer:
    def __init__(self, schema_list: List[dict]):
        self.schema_list = schema_list
        self.obs_dim_dict = self._build_obs_dim_dict()
        self.obs_seq_len_dict = self._build_obs_seq_len_dict()
        self.obs_flat_dim = self._build_obs_flat_dim()

    def _build_obs_dim_dict(self):
        obs_dim_dict = {}
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            feat_dim = schema["feat_dim"]
            obs_dim_dict[obs_name] = feat_dim
        return obs_dim_dict

    def _build_obs_seq_len_dict(self):
        obs_seq_len_dict = {}
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            seq_len = schema["seq_len"]
            obs_seq_len_dict[obs_name] = seq_len
        return obs_seq_len_dict

    def _build_obs_flat_dim(self):
        obs_flat_dim = 0
        for schema in self.schema_list:
            seq_len = schema["seq_len"]
            feat_dim = schema["feat_dim"]
            obs_flat_dim += seq_len * feat_dim
        return obs_flat_dim

    def serialize(self, obs_seq_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(obs_seq_list) == len(self.schema_list)
        B = obs_seq_list[0].shape[0]
        output_tensor = []
        for schema, obs_seq in zip(self.schema_list, obs_seq_list):
            assert obs_seq.ndim == 3
            assert obs_seq.shape[0] == B
            assert obs_seq.shape[1] == schema["seq_len"]
            assert obs_seq.shape[2] == schema["feat_dim"]
            output_tensor.append(obs_seq.reshape(B, -1))
        return torch.cat(output_tensor, dim=-1)

    def deserialize(self, obs_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        assert obs_seq_tensor.ndim == 2
        output_dict = {}
        array_start_idx = 0
        B = obs_seq_tensor.shape[0]
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            seq_len = schema["seq_len"]
            feat_dim = schema["feat_dim"]
            obs_size = seq_len * feat_dim
            array_end_idx = array_start_idx + obs_size
            output_dict[obs_name] = obs_seq_tensor[
                :, array_start_idx:array_end_idx
            ].reshape(B, seq_len, feat_dim)
            array_start_idx = array_end_idx

        return output_dict


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict: Union[dict, ObsSeqSerializer],
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
    ):
        super(PPOActor, self).__init__()

        self.use_logvar = module_config_dict.get("use_logvar", False)

        module_config_dict = self._process_module_config(
            module_config_dict, num_actions
        )

        self.actor_net_type = module_config_dict.get("type", "MLP")

        if self.actor_net_type == "MLP":
            self.actor_module = MLP(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.actor_net_type == "MoEMLP":
            self.actor_module = MoEMLP(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.actor_net_type == "TFStudent":
            self.actor_module = TFStudent(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.actor_net_type == "MoEMLPV2":
            self.actor_module = MoEMLPV2(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.actor_net_type == "MoEMLPVAE":
            self.actor_module = MoEMLPVAE(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.actor_net_type == "MoEMLPVAEV2":
            self.actor_module = MoEMLPVAEV2(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.actor_net_type == "EstVAEStudent":
            self.actor_module = EstVAEStudent(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        else:
            raise NotImplementedError

        self.fix_sigma = module_config_dict.get("fix_sigma", False)
        self.max_sigma = module_config_dict.get("max_sigma", 1.0)
        self.min_sigma = module_config_dict.get("min_sigma", 0.1)

        # Noise std type aligned with rsl_rl ("scalar" or "log").
        # Fallback to legacy use_logvar if provided.
        self.noise_std_type = module_config_dict.get(
            "noise_std_type", "scalar"
        ).lower()
        if self.use_logvar:
            self.noise_std_type = "log"

        # Action noise parameters (kept outside nets so optimizer updates them)
        if self.noise_std_type == "log":
            logger.info("Using log-std parameterization for action noise")
            self.log_std = nn.Parameter(
                torch.log(torch.ones(num_actions) * init_noise_std)
            )
            if self.fix_sigma:
                self.log_std.requires_grad = False
        else:  # scalar (default)
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            if self.fix_sigma:
                self.std.requires_grad = False
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict["output_dim"]):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        # Resolve std according to parameterization
        if hasattr(self, "log_std"):
            std_val = torch.exp(self.log_std)
        else:
            std_val = self.std
        # Ensure strictly positive std for numerical stability (do not clamp to large min so it can decrease)
        std_val = torch.clamp(std_val, min=1e-6)
        self.distribution = Normal(mean, std_val)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        if not self.use_logvar:
            self.std.to("cpu")


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(PPOCritic, self).__init__()
        self.critic_net_type = module_config_dict.get("type", "MLP")
        if self.critic_net_type == "MLP":
            self.critic_module = MLP(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.critic_net_type == "MoEMLP":
            self.critic_module = MoEMLP(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.critic_net_type == "MoEMLPV2":
            self.critic_module = MoEMLPV2(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.critic_net_type == "MoEMLPVAE":
            self.critic_module = MoEMLPVAE(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        elif self.critic_net_type == "MoEMLPVAEV2":
            self.critic_module = MoEMLPVAEV2(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        else:
            raise NotImplementedError

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value

    @property
    def logits_weights(self):
        # return the last layer weights of the critic
        return self.critic_module.module[-1].weight


class RNDNet(nn.Module):
    """Random Network Distillation Network."""

    def __init__(self, obs_dim_dict, module_config_dict):
        super(RNDNet, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict

        # Extract configuration
        if isinstance(obs_dim_dict, dict):
            self.obs_dim = sum(obs_dim_dict.values())
        else:  # ObsSeqSerializer
            self.obs_dim = obs_dim_dict.obs_flat_dim

        self.hidden_dims = module_config_dict.get(
            "hidden_dims", [512, 512, 512]
        )
        self.rnd_state_dim = module_config_dict.get("rnd_state_dim", 512)

        self.rand_net = self._build_mlp()
        self.target_net = self._build_mlp()

        # freeze random net
        for param in self.rand_net.parameters():
            param.requires_grad = False

    def _build_mlp(self):
        module_list = nn.ModuleList()
        module_list.append(nn.Linear(self.obs_dim, self.hidden_dims[0]))
        for i in range(1, len(self.hidden_dims)):
            module_list.append(nn.SiLU())
            module_list.append(
                nn.Linear(
                    self.hidden_dims[i - 1],
                    self.hidden_dims[i],
                )
            )
        module_list.append(nn.SiLU())
        module_list.append(nn.Linear(self.hidden_dims[-1], self.rnd_state_dim))
        return nn.Sequential(*module_list)

    def forward(self, obs):
        """Forward pass through target network."""
        return self.target_net(obs)

    def get_rnd_reward(self, obs):
        """Compute RND intrinsic reward."""
        with torch.no_grad():
            rand_pred = self.rand_net(obs)
        target_pred = self.target_net(obs)
        return (rand_pred - target_pred).square().sum(dim=-1)

    def get_rnd_loss(self, obs):
        """Compute RND loss for training the target network."""
        with torch.no_grad():
            rand_pred = self.rand_net(obs)
        target_pred = self.target_net(obs)
        return nn.functional.mse_loss(target_pred, rand_pred)
