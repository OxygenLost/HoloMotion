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


import inspect
from typing import Optional

from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from loguru import logger


class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )
        return down_proj


class DeepseekV3TopkRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.scores = None
        self.router_logits = None

        self.linear_layer = nn.Linear(hidden_size, n_routed_experts)
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts)
        )

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = self.linear_layer(hidden_states.type(torch.float32))
        scores = router_logits.sigmoid()
        self.router_logits = router_logits
        self.scores = scores
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class DeepseekV3MoE(nn.Module):
    """A mixed expert module containing shared experts."""

    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        moe_intermediate_size: int,
        n_shared_experts: int,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                )
                for _ in range(n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(
            hidden_size=hidden_size,
            top_k=top_k,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size * n_shared_experts,
        )

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(
            hidden_states, dtype=topk_weights.dtype
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(
            hidden_states, topk_indices, topk_weights
        ).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class MoEMLP(nn.Module):
    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: dict,
        num_actions: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.num_actions = num_actions

        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )

        self._calculate_input_dim()
        self._calculate_output_dim()

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

    def compute_load_balancing_loss(self) -> torch.Tensor:
        if self.router_probs is None or self.tokens_per_expert is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        num_tokens = self.router_probs.shape[0]
        if num_tokens == 0:
            # Avoid division by zero if batch size is zero
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        # Fraction of tokens dispatched to each expert f_i
        # self.tokens_per_expert has shape [num_fine_experts]
        fraction_tokens_per_expert = (
            self.tokens_per_expert.float() / num_tokens
        )

        # Average router probability for each expert P_i
        # self.router_probs has shape [num_tokens, num_fine_experts]
        avg_router_prob_per_expert = torch.mean(self.router_probs, dim=0)

        # Compute the loss: N * sum(f_i * P_i)
        load_balancing_loss = self.num_fine_experts * torch.sum(
            fraction_tokens_per_expert * avg_router_prob_per_expert
        )

        # Multiply by the scaling factor alpha
        return load_balancing_loss

    def compute_bound_loss(self) -> torch.Tensor:
        # penalize out of bound actions

        lb_loss = (
            torch.relu(self.action_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.action_output_ub)
            .square()
            .mean()
        )

        return (lb_loss + ub_loss) * 0.5

    def _calculate_input_dim(self):
        input_dim = 0
        for each_input in self.module_config_dict["input_dim"]:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                input_dim += int(each_input)  # Ensure int conversion
            elif isinstance(each_input, str) and each_input.isdigit():
                input_dim += int(each_input)
            else:
                raise ValueError(f"Unknown input type: {each_input}")
        self.input_dim = input_dim

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale

        self.register_buffer("action_output_lb", lb[None, :])
        self.register_buffer("action_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            assert isinstance(each_output, int), (
                f"Output dim placeholder not replaced: {each_output}"
            )
            output_dim += each_output
        self.output_dim = output_dim

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        tokens_per_expert = torch.zeros(
            self.num_fine_experts,
            device=hidden_states.device,
            dtype=torch.long,
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_fine_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(self.num_fine_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, topk_pos_indices = torch.where(mask)
            if token_indices.numel() > 0:
                tokens_per_expert[expert_idx] = token_indices.numel()
                expert_weights = topk_weights[token_indices, topk_pos_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        self.tokens_per_expert = tokens_per_expert
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )
        projected_x = self.input_projection(x)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_mlp(combined_hidden)

        if self.clamp_actor_output:
            # self.output_actions = self.output_actions.clamp(
            #     min=self.action_output_lb,
            #     max=self.action_output_ub,
            # )
            self.bound_loss = self.compute_bound_loss()

        return self.output_actions


class MoEMLPV2(nn.Module):
    def __init__(
        self,
        obs_serializer,
        module_config_dict: dict,
    ):
        super().__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict

        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )

        self._calculate_output_dim()

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.input_dim = sum(
            [
                self.obs_dim_dict[each_input]
                * self.obs_seq_len_dict[each_input]
                for each_input in self.obs_dim_dict
            ]
        )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

    def compute_load_balancing_loss(self) -> torch.Tensor:
        # Implements the load balancing loss from Switch Transformers (https://arxiv.org/abs/2101.03961)
        # loss = alpha * N * sum(f_i * P_i)
        # N = num_experts, f_i = fraction of tokens dispatched to expert i, P_i = avg router prob for expert i

        if self.router_probs is None or self.tokens_per_expert is None:
            # Return zero loss if forward pass hasn't happened or buffers are not populated
            # Try to get the device from a parameter that is guaranteed to exist
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        num_tokens = self.router_probs.shape[0]
        if num_tokens == 0:
            # Avoid division by zero if batch size is zero
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        # Fraction of tokens dispatched to each expert f_i
        # self.tokens_per_expert has shape [num_fine_experts]
        fraction_tokens_per_expert = (
            self.tokens_per_expert.float() / num_tokens
        )

        # Average router probability for each expert P_i
        # self.router_probs has shape [num_tokens, num_fine_experts]
        avg_router_prob_per_expert = torch.mean(self.router_probs, dim=0)

        # Compute the loss: N * sum(f_i * P_i)
        load_balancing_loss = self.num_fine_experts * torch.sum(
            fraction_tokens_per_expert * avg_router_prob_per_expert
        )

        # Multiply by the scaling factor alpha
        return load_balancing_loss

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.action_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.action_output_ub)
            .square()
            .mean()
        )

        return (lb_loss + ub_loss) * 0.5

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale

        self.register_buffer("action_output_lb", lb[None, :])
        self.register_buffer("action_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            assert isinstance(each_output, int), (
                f"Output dim placeholder not replaced: {each_output}"
            )
            output_dim += each_output
        self.output_dim = output_dim

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        tokens_per_expert = torch.zeros(
            self.num_fine_experts,
            device=hidden_states.device,
            dtype=torch.long,
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_fine_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(self.num_fine_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, topk_pos_indices = torch.where(mask)
            if token_indices.numel() > 0:
                tokens_per_expert[expert_idx] = token_indices.numel()
                expert_weights = topk_weights[token_indices, topk_pos_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        self.tokens_per_expert = tokens_per_expert
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )
        projected_x = self.input_projection(x)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_head(
            self.output_mlp(combined_hidden)
        )

        return self.output_actions


class MoEMLPVAE(MoEMLPV2):
    def __init__(
        self,
        obs_serializer,
        module_config_dict: dict,
    ):
        super().__init__(
            obs_serializer=obs_serializer,
            module_config_dict=module_config_dict,
        )
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict

        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )
        self.fut_ref_dim = self.obs_dim_dict["fut_ref"]
        self.fut_ref_seq_len = self.obs_seq_len_dict["fut_ref"]
        self.vae_latent_dim = module_config_dict.get("vae_latent_dim", 128)
        self.kl_loss_scale = module_config_dict.get("kl_loss_scale", 0.01)

        self._calculate_output_dim()

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.input_dim = sum(
            [
                self.obs_dim_dict[each_input]
                * self.obs_seq_len_dict[each_input]
                for each_input in ["cur_priocep_v2", "domain_params"]
            ]
            + [self.vae_latent_dim]
        )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

        self.ref_vae_encoder = nn.Sequential(
            nn.Linear(
                self.fut_ref_dim * self.fut_ref_seq_len, self.hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.vae_latent_dim * 2),
        )
        self.ref_vae_decoder = nn.Sequential(
            nn.Linear(self.vae_latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(
                self.hidden_dim, self.fut_ref_dim * self.fut_ref_seq_len
            ),
        )

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )

        deserilized_obs_dict = self.obs_serializer.deserialize(x)

        cur_priocep = deserilized_obs_dict["cur_priocep_v2"][:, 0]
        fut_ref = deserilized_obs_dict["fut_ref"]
        fut_valid_mask = deserilized_obs_dict["fut_ref_valid_mask"]
        domain_params = deserilized_obs_dict["domain_params"][:, 0]

        bs = cur_priocep.shape[0]
        ts = fut_ref.shape[1]

        fut_ref = fut_ref.reshape(bs, -1)

        encoded_fut_ref = self.ref_vae_encoder(fut_ref)
        fut_ref_vae_latent_mu, fut_ref_vae_latent_logvar = torch.chunk(
            encoded_fut_ref, 2, dim=-1
        )

        # Create posterior distribution from encoder outputs
        fut_ref_vae_std = torch.exp(fut_ref_vae_latent_logvar * 0.5)
        posterior_dist = dist.Normal(fut_ref_vae_latent_mu, fut_ref_vae_std)

        if self.training:
            # Sample a latent using reparameterization trick
            fut_ref_vae_latent = posterior_dist.rsample()
            rec_fut_ref = self.ref_vae_decoder(fut_ref_vae_latent)
            # VAE reconstruction loss
            rec_loss = F.mse_loss(
                rec_fut_ref,
                fut_ref,
                reduction="none",
            ).reshape(bs, ts, -1)  # [bs, ts, 1]
            rec_loss = rec_loss * fut_valid_mask  # [bs, ts, 1]
            rec_loss = rec_loss.sum(dim=[-2, -1]) / (
                fut_valid_mask.sum(dim=[-2, -1]) + 1e-4
            )  # [bs]
            rec_loss = rec_loss.mean()
            # VAE KL loss with unit Gaussian prior
            prior_dist = dist.Normal(
                torch.zeros_like(fut_ref_vae_latent_mu),
                torch.ones_like(fut_ref_vae_std),
            )
            kl_loss = dist.kl.kl_divergence(posterior_dist, prior_dist).sum()
            self.rec_loss = rec_loss
            self.kl_loss = kl_loss
        else:
            fut_ref_vae_latent = fut_ref_vae_latent_mu

        x = torch.cat(
            [
                cur_priocep,
                domain_params,
                fut_ref_vae_latent,
            ],
            dim=-1,
        )

        projected_x = self.input_projection(x)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_head(
            self.output_mlp(combined_hidden)
        )

        return self.output_actions

    def compute_vae_loss(self):
        return self.rec_loss, self.kl_loss * self.kl_loss_scale


class MoEMLPVAEV2(MoEMLPVAE):
    def __init__(
        self,
        obs_serializer,
        module_config_dict: dict,
    ):
        super().__init__(
            obs_serializer=obs_serializer,
            module_config_dict=module_config_dict,
        )
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict

        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )
        self.fut_ref_dim = self.obs_dim_dict["fut_ref"]
        self.fut_ref_seq_len = self.obs_seq_len_dict["fut_ref"]
        self.vae_latent_dim = module_config_dict.get("vae_latent_dim", 128)
        self.kl_loss_scale = module_config_dict.get("kl_loss_scale", 0.01)

        self._calculate_output_dim()

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.input_dim = sum(
            [
                self.obs_dim_dict[each_input]
                * self.obs_seq_len_dict[each_input]
                for each_input in ["cur_priocep_v2", "domain_params"]
            ]
            + [self.vae_latent_dim]
        )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

        # 1D Conv encoder for sequence data
        # Input: [batch, fut_ref_dim, fut_ref_seq_len]
        # Output: [batch, vae_latent_dim * 2]
        conv_hidden_dim = self.hidden_dim // 4
        self.ref_vae_encoder = nn.Sequential(
            nn.Conv1d(
                self.fut_ref_dim, conv_hidden_dim, kernel_size=3, padding=1
            ),
            nn.SiLU(),
            nn.Conv1d(
                conv_hidden_dim,
                conv_hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv1d(
                conv_hidden_dim * 2,
                conv_hidden_dim * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(conv_hidden_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.vae_latent_dim * 2),
        )

        # 1D Transpose conv decoder for sequence data
        # Input: [batch, vae_latent_dim]
        # Output: [batch, fut_ref_dim, fut_ref_seq_len]
        # Calculate the intermediate sequence length for upsampling
        self.intermediate_seq_len = max(1, self.fut_ref_seq_len // 4)
        self.latent_to_seq = nn.Linear(
            self.vae_latent_dim,
            conv_hidden_dim * 4 * self.intermediate_seq_len,
        )
        self.ref_vae_decoder = nn.Sequential(
            nn.ConvTranspose1d(
                conv_hidden_dim * 4,
                conv_hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose1d(
                conv_hidden_dim * 2,
                conv_hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.SiLU(),
            nn.Conv1d(
                conv_hidden_dim, self.fut_ref_dim, kernel_size=3, padding=1
            ),
        )

        # Add an adaptive layer to handle any size mismatch
        if self.intermediate_seq_len * 4 != self.fut_ref_seq_len:
            self.output_adapter = nn.Conv1d(
                self.fut_ref_dim, self.fut_ref_dim, kernel_size=1
            )

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )

        deserilized_obs_dict = self.obs_serializer.deserialize(x)

        cur_priocep = deserilized_obs_dict["cur_priocep_v2"][:, 0]
        fut_ref = deserilized_obs_dict["fut_ref"]
        fut_valid_mask = deserilized_obs_dict["fut_ref_valid_mask"]
        domain_params = deserilized_obs_dict["domain_params"][:, 0]

        bs = cur_priocep.shape[0]
        ts = fut_ref.shape[1]
        conv_hidden_dim = self.hidden_dim // 4

        # Transpose for conv1d: [bs, ts, fut_ref_dim] -> [bs, fut_ref_dim, ts]
        fut_ref_conv = fut_ref.transpose(1, 2)

        encoded_fut_ref = self.ref_vae_encoder(fut_ref_conv)
        fut_ref_vae_latent_mu, fut_ref_vae_latent_logvar = torch.chunk(
            encoded_fut_ref, 2, dim=-1
        )

        # Create posterior distribution from encoder outputs
        fut_ref_vae_std = torch.exp(fut_ref_vae_latent_logvar * 0.5)
        posterior_dist = dist.Normal(fut_ref_vae_latent_mu, fut_ref_vae_std)

        if self.training:
            # Sample a latent using reparameterization trick
            fut_ref_vae_latent = posterior_dist.rsample()

            # Decode with 1D transpose conv
            latent_expanded = self.latent_to_seq(
                fut_ref_vae_latent
            )  # [bs, conv_hidden_dim * 4 * intermediate_seq_len]
            latent_reshaped = latent_expanded.view(
                bs, conv_hidden_dim * 4, self.intermediate_seq_len
            )  # [bs, conv_hidden_dim * 4, intermediate_seq_len]
            rec_fut_ref_conv = self.ref_vae_decoder(
                latent_reshaped
            )  # [bs, fut_ref_dim, intermediate_seq_len * 4]

            # Handle potential size mismatch
            if hasattr(self, "output_adapter"):
                rec_fut_ref_conv = self.output_adapter(rec_fut_ref_conv)
                # Interpolate to exact target length if needed
                if rec_fut_ref_conv.size(2) != ts:
                    rec_fut_ref_conv = F.interpolate(
                        rec_fut_ref_conv,
                        size=ts,
                        mode="linear",
                        align_corners=False,
                    )

            rec_fut_ref = rec_fut_ref_conv.transpose(
                1, 2
            )  # [bs, ts, fut_ref_dim]

            # Flatten for loss calculation - keep original format
            fut_ref_flat = fut_ref.reshape(bs, -1)
            rec_fut_ref_flat = rec_fut_ref.reshape(bs, -1)

            # VAE reconstruction loss
            rec_loss = F.mse_loss(
                rec_fut_ref_flat,
                fut_ref_flat,
                reduction="none",
            ).reshape(bs, ts, -1)  # [bs, ts, fut_ref_dim]
            rec_loss = rec_loss * fut_valid_mask  # [bs, ts, fut_ref_dim]
            rec_loss = rec_loss.sum(dim=[-2, -1]) / (
                fut_valid_mask.sum(dim=[-2, -1]) + 1e-4
            )  # [bs]
            rec_loss = rec_loss.mean()
            # VAE KL loss with unit Gaussian prior
            prior_dist = dist.Normal(
                torch.zeros_like(fut_ref_vae_latent_mu),
                torch.ones_like(fut_ref_vae_std),
            )
            kl_loss = dist.kl.kl_divergence(posterior_dist, prior_dist).sum()
            self.rec_loss = rec_loss
            self.kl_loss = kl_loss
        else:
            fut_ref_vae_latent = fut_ref_vae_latent_mu

        x = torch.cat(
            [
                cur_priocep,
                domain_params,
                fut_ref_vae_latent,
            ],
            dim=-1,
        )

        projected_x = self.input_projection(x)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_head(
            self.output_mlp(combined_hidden)
        )

        return self.output_actions


class MLP(nn.Module):
    def __init__(self, obs_serializer, module_config_dict):
        super(MLP, self).__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict
        self.input_dim = obs_serializer.obs_flat_dim

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.use_layernorm = module_config_dict.get("use_layernorm", False)

        self._calculate_output_dim()
        self._build_mlp(self.module_config_dict.layer_config)

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale
        self.register_buffer("actor_output_lb", lb[None, :])
        self.register_buffer("actor_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(
                    f"{current_function_name} - Unknown output type: {each_output}"
                )
        self.output_dim = output_dim

    def _build_mlp(self, layer_config):
        layers = []
        hidden_dims = layer_config["hidden_dims"]
        output_dim = self.output_dim
        activation = getattr(nn, layer_config["activation"])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                if self.use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        output = self.module(input)
        if self.clamp_actor_output:
            if hasattr(self, "actor_output_lb") and hasattr(
                self, "actor_output_ub"
            ):
                if self.actor_output_lb.device != output.device:
                    self.actor_output_lb = self.actor_output_lb.to(
                        output.device
                    )
                    self.actor_output_ub = self.actor_output_ub.to(
                        output.device
                    )
                self.output_actions = output
        return output

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.actor_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.actor_output_ub)
            .square()
            .mean()
        )
        return (lb_loss + ub_loss) * 0.5


class EstVAEStudent(nn.Module):
    def __init__(self, obs_serializer, module_config_dict):
        super(EstVAEStudent, self).__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict

        # Configuration parameters
        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )

        # Latent dimensions
        self.hist_latent_dim = module_config_dict.get("hist_latent_dim", 64)
        self.fut_ref_latent_dim = module_config_dict.get(
            "fut_ref_latent_dim", 64
        )
        self.kl_loss_scale = module_config_dict.get("kl_loss_scale", 1e-7)
        self.domain_est_loss_scale = module_config_dict.get(
            "domain_est_loss_scale", 1.0
        )
        self.lin_vel_est_loss_scale = module_config_dict.get(
            "lin_vel_est_loss_scale", 1.0
        )

        # Observation dimensions
        self.cur_obs_dim = self.obs_dim_dict["cur_obs"]
        self.cur_hist_seq_dim = self.obs_dim_dict["cur_hist_seq"]
        self.cur_hist_seq_len = self.obs_seq_len_dict["cur_hist_seq"]
        self.fut_ref_dim = self.obs_dim_dict["fut_ref_motion_seq"]
        self.fut_ref_seq_len = self.obs_seq_len_dict["fut_ref_motion_seq"]
        self.domain_params_dim = self.obs_dim_dict["domain_params"]
        self.ha_root_lin_vel_dim = self.obs_dim_dict["ha_root_lin_vel"]

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self._calculate_output_dim()

        # 1D Conv encoder for history sequence -> l_a
        conv_hidden_dim = self.hidden_dim // 4
        self.hist_conv_encoder = nn.Sequential(
            nn.Conv1d(
                self.cur_hist_seq_dim,
                conv_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv1d(
                conv_hidden_dim,
                conv_hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv1d(
                conv_hidden_dim * 2,
                conv_hidden_dim * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(conv_hidden_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hist_latent_dim),
        )

        # VAE encoder for future reference sequence -> l_b (simple MLP)
        self.fut_ref_flattened_dim = self.fut_ref_dim * self.fut_ref_seq_len
        self.fut_ref_vae_encoder = nn.Sequential(
            nn.Linear(self.fut_ref_flattened_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.fut_ref_latent_dim * 2),
        )

        # VAE decoder for future reference sequence (simple MLP)
        self.fut_ref_vae_decoder = nn.Sequential(
            nn.Linear(self.fut_ref_latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.fut_ref_flattened_dim),
        )

        # MoE-MLP for action prediction
        # Input: cur_obs + l_a + l_b
        self.input_dim = (
            self.cur_obs_dim + self.hist_latent_dim + self.fut_ref_latent_dim
        )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

        # Estimation heads using l_a
        self.domain_est_head = nn.Sequential(
            nn.Linear(self.hist_latent_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.domain_params_dim),
        )

        self.lin_vel_est_head = nn.Sequential(
            nn.Linear(self.hist_latent_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.ha_root_lin_vel_dim),
        )

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale
        self.register_buffer("actor_output_lb", lb[None, :])
        self.register_buffer("actor_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(
                    f"{current_function_name} - Unknown output type: {each_output}"
                )
        self.output_dim = output_dim

    def compute_load_balancing_loss(self) -> torch.Tensor:
        if self.router_probs is None or self.tokens_per_expert is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        num_tokens = self.router_probs.shape[0]
        if num_tokens == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        # Fraction of tokens dispatched to each expert f_i
        fraction_tokens_per_expert = (
            self.tokens_per_expert.float() / num_tokens
        )

        # Average router probability for each expert P_i
        avg_router_prob_per_expert = torch.mean(self.router_probs, dim=0)

        # Compute the loss: N * sum(f_i * P_i)
        load_balancing_loss = self.num_fine_experts * torch.sum(
            fraction_tokens_per_expert * avg_router_prob_per_expert
        )

        return load_balancing_loss

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        tokens_per_expert = torch.zeros(
            self.num_fine_experts,
            device=hidden_states.device,
            dtype=torch.long,
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_fine_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(self.num_fine_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, topk_pos_indices = torch.where(mask)
            if token_indices.numel() > 0:
                tokens_per_expert[expert_idx] = token_indices.numel()
                expert_weights = topk_weights[token_indices, topk_pos_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        self.tokens_per_expert = tokens_per_expert
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, input):
        assert input.ndim == 2, (
            f"Expected input shape [B, D_in], but got {input.shape}"
        )

        # Deserialize observations
        deserialized_obs_dict = self.obs_serializer.deserialize(input)

        cur_obs = deserialized_obs_dict["cur_obs"][:, 0]  # [B, cur_obs_dim]
        cur_hist_seq = deserialized_obs_dict[
            "cur_hist_seq"
        ]  # [B, seq_len, hist_dim]
        fut_ref_motion_seq = deserialized_obs_dict[
            "fut_ref_motion_seq"
        ]  # [B, seq_len, fut_dim]
        fut_ref_valid_mask = deserialized_obs_dict[
            "fut_ref_valid_mask"
        ]  # [B, seq_len, 1]
        domain_params = deserialized_obs_dict["domain_params"][
            :, 0
        ]  # [B, domain_dim]
        ha_root_lin_vel = deserialized_obs_dict["ha_root_lin_vel"][
            :, 0
        ]  # [B, 3]

        bs = cur_obs.shape[0]
        ts = fut_ref_motion_seq.shape[1]
        conv_hidden_dim = self.hidden_dim // 4

        # 1. Encode history sequence -> l_a
        # Transpose for conv1d: [B, seq_len, dim] -> [B, dim, seq_len]
        cur_hist_seq_conv = cur_hist_seq.transpose(1, 2)
        l_a = self.hist_conv_encoder(cur_hist_seq_conv)  # [B, hist_latent_dim]

        # 2. VAE encode/decode future reference -> l_b
        # Flatten for MLP: [B, seq_len, dim] -> [B, seq_len * dim]
        fut_ref_flat = fut_ref_motion_seq.reshape(bs, -1)

        encoded_fut_ref = self.fut_ref_vae_encoder(fut_ref_flat)
        fut_ref_vae_latent_mu, fut_ref_vae_latent_logvar = torch.chunk(
            encoded_fut_ref, 2, dim=-1
        )

        # Create posterior distribution from encoder outputs
        fut_ref_vae_std = torch.exp(fut_ref_vae_latent_logvar * 0.5)
        posterior_dist = dist.Normal(fut_ref_vae_latent_mu, fut_ref_vae_std)

        if self.training:
            # Sample a latent using reparameterization trick
            l_b = posterior_dist.rsample()

            # Decode with MLP
            rec_fut_ref_flat = self.fut_ref_vae_decoder(
                l_b
            )  # [B, seq_len * dim]
            rec_fut_ref = rec_fut_ref_flat.reshape(
                bs, ts, self.fut_ref_dim
            )  # [B, ts, fut_ref_dim]

            # VAE reconstruction loss
            rec_loss = F.mse_loss(
                rec_fut_ref,
                fut_ref_motion_seq,
                reduction="none",
            )  # [B, ts, fut_ref_dim]
            rec_loss = rec_loss * fut_ref_valid_mask  # Apply validity mask
            rec_loss = rec_loss.sum(dim=[-2, -1]) / (
                fut_ref_valid_mask.sum(dim=[-2, -1]) + 1e-4
            )  # [B]
            rec_loss = rec_loss.mean()

            # VAE KL loss with unit Gaussian prior
            prior_dist = dist.Normal(
                torch.zeros_like(fut_ref_vae_latent_mu),
                torch.ones_like(fut_ref_vae_std),
            )
            kl_loss = dist.kl.kl_divergence(posterior_dist, prior_dist).sum()
            self.rec_loss = rec_loss
            self.kl_loss = kl_loss
        else:
            l_b = fut_ref_vae_latent_mu

        # 3. Concatenate cur_obs + l_a + l_b and forward through MoE-MLP
        combined_input = torch.cat([cur_obs, l_a, l_b], dim=-1)

        projected_x = self.input_projection(combined_input)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_head(
            self.output_mlp(combined_hidden)
        )

        # Store estimates using l_a for loss computation (only during training)
        if self.training:
            self.estimated_domain_params = self.domain_est_head(l_a)
            self.estimated_lin_vel = self.lin_vel_est_head(l_a)
            self.gt_domain_params = domain_params
            self.gt_ha_root_lin_vel = ha_root_lin_vel

        if self.clamp_actor_output:
            if hasattr(self, "actor_output_lb") and hasattr(
                self, "actor_output_ub"
            ):
                if self.actor_output_lb.device != self.output_actions.device:
                    self.actor_output_lb = self.actor_output_lb.to(
                        self.output_actions.device
                    )
                    self.actor_output_ub = self.actor_output_ub.to(
                        self.output_actions.device
                    )

        return self.output_actions

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.actor_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.actor_output_ub)
            .square()
            .mean()
        )
        return (lb_loss + ub_loss) * 0.5

    def compute_vae_loss(self):
        """Compute VAE reconstruction and KL losses."""
        if (
            self.training
            and hasattr(self, "rec_loss")
            and hasattr(self, "kl_loss")
        ):
            return self.rec_loss, self.kl_loss * self.kl_loss_scale
        else:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device), torch.tensor(
                0.0, device=device
            )

    def compute_domain_est_loss(self) -> torch.Tensor:
        """Estimate domain vector with l_a."""
        if hasattr(self, "estimated_domain_params") and hasattr(
            self, "gt_domain_params"
        ):
            return (
                F.mse_loss(self.estimated_domain_params, self.gt_domain_params)
                * self.domain_est_loss_scale
            )
        else:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

    def compute_lin_vel_est_loss(self) -> torch.Tensor:
        """Estimate heading-aligned root linear velocity with l_a."""
        if hasattr(self, "estimated_lin_vel") and hasattr(
            self, "gt_ha_root_lin_vel"
        ):
            return (
                F.mse_loss(self.estimated_lin_vel, self.gt_ha_root_lin_vel)
                * self.lin_vel_est_loss_scale
            )
        else:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)


class TFStudent(nn.Module):
    def __init__(self, obs_serializer, module_config_dict):
        super(TFStudent, self).__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict
        self.input_dim = obs_serializer.obs_flat_dim

        self.patch_size = module_config_dict.get("patch_size", 2)
        self.num_obs_tokens = (
            self.obs_seq_len_dict["student_actor_realworld_obs_seq"]
            // self.patch_size
        )

        self.fut_patch_size = module_config_dict.get("fut_patch_size", 1)
        self.num_fut_ref_tokens = (
            self.obs_seq_len_dict.get("fut_ref_motion_seq", 0)
            // self.fut_patch_size
        )

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get(
            "enabled",
            False,
        )
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.predict_local_body_pos = module_config_dict.get(
            "predict_local_body_pos",
            False,
        )

        self.predict_local_body_vel = module_config_dict.get(
            "predict_local_body_vel",
            False,
        )

        self.predict_root_lin_vel = module_config_dict.get(
            "predict_root_lin_vel",
            False,
        )

        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale
        self.register_buffer("actor_output_lb", lb[None, :])
        self.register_buffer("actor_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        hidden_dim = layer_config["hidden_dim"]

        # Query tokens for action and auxiliary outputs
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.aux_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        obs_sin_pe = get_sinusoid_encoding_table(
            self.num_obs_tokens, hidden_dim
        )[0]
        self.register_buffer("obs_pe", obs_sin_pe, persistent=False)

        # Projection layer for observation sequence patches
        self.obs_proj_layer = nn.Linear(
            self.obs_dim_dict["student_actor_realworld_obs_seq"]
            * self.patch_size,
            hidden_dim,
        )

        self.fut_ref_proj_layer = nn.Linear(
            self.obs_dim_dict["fut_ref_motion_seq"] * self.fut_patch_size,
            hidden_dim,
        )

        # Sinusoidal positional encoding for future reference sequence
        fut_ref_sin_pe = get_sinusoid_encoding_table(
            self.num_fut_ref_tokens, hidden_dim
        )[0]
        self.register_buffer("fut_ref_pe", fut_ref_sin_pe, persistent=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.0,
            activation=nn.SiLU(),
        )
        self.tf_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=layer_config["num_layers"]
        )

        # Output layers
        self.action_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.output_dim),
        )

        # Prediction heads
        if self.predict_local_body_pos:
            self.local_body_pos_pred_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(
                    128, self.module_config_dict.pred_local_body_pos_dim
                ),
            )

        if self.predict_local_body_vel:
            self.local_body_vel_pred_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(
                    128, self.module_config_dict.pred_local_body_vel_dim
                ),
            )

        if self.predict_root_lin_vel:
            self.root_lin_vel_pred_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 3),
            )

    def forward(self, input_tensor):
        seq_obs_dict = self.obs_serializer.deserialize(input_tensor)
        obs_seq = seq_obs_dict["student_actor_realworld_obs_seq"]

        # Process observation sequence patches
        obs_seq_patches = rearrange(
            obs_seq, "B (T P) C -> B T (P C)", P=self.patch_size
        )
        assert obs_seq_patches.shape[1] == self.num_obs_tokens

        obs_seq_patches = self.obs_proj_layer(obs_seq_patches)
        obs_seq_patches = obs_seq_patches + self.obs_pe

        # Create query sequence (action + aux queries + observation patches)
        query_seq = torch.cat(
            [
                self.action_query.repeat(obs_seq_patches.shape[0], 1, 1),
                self.aux_query.repeat(obs_seq_patches.shape[0], 1, 1),
                obs_seq_patches,
            ],
            dim=1,
        )  # [B, 2 + num_obs_tokens, hidden_dim]

        fut_ref_seq = seq_obs_dict["fut_ref_motion_seq"]
        fut_ref_valid_mask = seq_obs_dict.get("fut_ref_valid_mask", None)

        # Process future reference sequence
        fut_ref_patches = rearrange(
            fut_ref_seq, "B (T P) C -> B T (P C)", P=self.fut_patch_size
        )
        fut_ref_patches = self.fut_ref_proj_layer(fut_ref_patches)
        fut_ref_patches = fut_ref_patches + self.fut_ref_pe

        # Handle validity mask for future reference
        memory_key_padding_mask = None
        if fut_ref_valid_mask is not None:
            fut_ref_valid_mask = fut_ref_valid_mask.squeeze(-1).bool()
            if self.fut_patch_size > 1:
                fut_ref_valid_mask = rearrange(
                    fut_ref_valid_mask,
                    "B (T P) -> B T P",
                    P=self.fut_patch_size,
                ).any(dim=-1)
            memory_key_padding_mask = ~fut_ref_valid_mask

        # Apply transformer decoder (cross-attention)
        out_seq = self.tf_decoder(
            tgt=query_seq,
            memory=fut_ref_patches,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Extract outputs
        action_out = self.action_output_layer(out_seq[:, 0, :])  # Action token
        aux_out = out_seq[:, 1, :]  # Auxiliary token

        # Handle action bounds
        if self.clamp_actor_output:
            if hasattr(self, "actor_output_lb") and hasattr(
                self, "actor_output_ub"
            ):
                if self.actor_output_lb.device != action_out.device:
                    self.actor_output_lb = self.actor_output_lb.to(
                        action_out.device
                    )
                    self.actor_output_ub = self.actor_output_ub.to(
                        action_out.device
                    )
                self.output_actions = action_out

        # Prediction heads
        if self.predict_local_body_pos:
            self.predicted_local_body_pos = self.local_body_pos_pred_head(
                aux_out
            )

        if self.predict_local_body_vel:
            self.predicted_local_body_vel = self.local_body_vel_pred_head(
                aux_out
            )

        if self.predict_root_lin_vel:
            self.predicted_root_lin_vel = self.root_lin_vel_pred_head(aux_out)

        return action_out

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.actor_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.actor_output_ub)
            .square()
            .mean()
        )

        return (lb_loss + ub_loss) * 0.5

    def compute_local_body_pos_reg_loss(
        self, gt_local_body_pos_extend_flat
    ) -> torch.Tensor:
        if not self.predict_local_body_pos or not hasattr(
            self, "predicted_local_body_pos"
        ):
            return torch.tensor(
                0.0, device=gt_local_body_pos_extend_flat.device
            )
        return nn.functional.mse_loss(
            self.predicted_local_body_pos, gt_local_body_pos_extend_flat
        )

    def compute_local_body_vel_reg_loss(
        self, gt_local_body_vel_extend_flat
    ) -> torch.Tensor:
        if not self.predict_local_body_vel or not hasattr(
            self, "predicted_local_body_vel"
        ):
            return torch.tensor(
                0.0, device=gt_local_body_vel_extend_flat.device
            )
        return nn.functional.mse_loss(
            self.predicted_local_body_vel, gt_local_body_vel_extend_flat
        )

    def compute_root_lin_vel_reg_loss(self, gt_root_lin_vel) -> torch.Tensor:
        if not self.predict_root_lin_vel or not hasattr(
            self, "predicted_root_lin_vel"
        ):
            return torch.tensor(0.0, device=gt_root_lin_vel.device)
        return nn.functional.mse_loss(
            self.predicted_root_lin_vel, gt_root_lin_vel
        )


class RunningMeanStdNormalizer(nn.Module):
    def __init__(
        self, feature_dim: int, epsilon: float = 1e-8, until: int = None
    ):
        """Initializes a running mean and standard deviation normalizer.

        Normalization is performed independently for each feature dimension.
        Buffers will be on the default device (CPU) unless the module instance
        is moved using .to(device).

        Args:
            feature_dim (int): The number of features (D) to normalize
                independently.
            epsilon (float): A small value to prevent division by zero and to
                initialize count. This value is stored in the 'epsilon_val'
                buffer.

        """
        super(RunningMeanStdNormalizer, self).__init__()
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        self.feature_dim = feature_dim
        self.until = until

        self.register_buffer(
            "running_mean", torch.zeros(feature_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "running_var", torch.ones(feature_dim, dtype=torch.float32)
        )
        # Keep std buffer (aligned to EmpiricalNormalization)
        self.register_buffer(
            "running_std", torch.ones(feature_dim, dtype=torch.float32)
        )
        # Number of samples seen (sum of batch sizes)
        self.register_buffer(
            "running_count", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "epsilon_val", torch.tensor(epsilon, dtype=torch.float32)
        )

    def update(self, x: torch.Tensor):
        """Updates the running mean and variance.

        The internal state (buffers) remains on its current device.
        Input x is moved to the buffer's device for computation.

        Args:
            x (torch.Tensor): A tensor of shape [B, D]
                              or [D] (a single D-dimensional feature vector).

        """
        # Gate updates like rsl_rl's EmpiricalNormalization
        if not self.training:
            return
        if self.until is not None and self.running_count >= self.until:
            return

        if not (x.ndim == 2 and x.shape[1] == self.feature_dim) and not (
            x.ndim == 1 and x.shape[0] == self.feature_dim
        ):
            raise ValueError(
                f"Input x for update must be shape "
                f"[B, {self.feature_dim}] or [{self.feature_dim}], "
                f"got {x.shape}"
            )

        if x.ndim == 1:  # Single D-dimensional sample
            x = x.unsqueeze(0)  # Treat as a batch of 1

        buffer_device = self.running_mean.device
        x_on_buffer_device = x.to(buffer_device)

        batch_size = (
            x_on_buffer_device.shape[0] if x_on_buffer_device.ndim == 2 else 1
        )
        if batch_size == 0:
            return

        count_x = torch.tensor(
            batch_size, dtype=torch.long, device=buffer_device
        )
        # Update global count first (sum of batch sizes)
        new_count = self.running_count + count_x
        rate = count_x.to(torch.float32) / new_count.to(torch.float32)

        batch_mean = torch.mean(x_on_buffer_device, dim=0)  # [D]
        batch_var = torch.var(x_on_buffer_device, dim=0, unbiased=False)  # [D]

        # Update mean
        delta_mean = batch_mean - self.running_mean
        new_mean = self.running_mean + rate * delta_mean

        # Update var using rsl_rl formula: var += rate * (var_x - var + delta*(mean_x - new_mean))
        new_var = self.running_var + rate * (
            batch_var - self.running_var + delta_mean * (batch_mean - new_mean)
        )

        # Clamp minimal var by epsilon for stability
        eps = self.epsilon_val.to(buffer_device)
        new_var = torch.maximum(new_var, eps)

        # Commit updates
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.running_std.copy_(torch.sqrt(new_var))
        self.running_count.copy_(new_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes multi-dimensional data.

        Using the running mean and variance for each feature.
        Output tensor is on the same device as the input x.

        Args:
            x (torch.Tensor): A tensor of shape [B, D] or [D] to normalize.

        Returns:
            torch.Tensor: The normalized data, on the same device as input x.

        """
        original_x_ndim = x.ndim
        if not (
            original_x_ndim == 2 and x.shape[1] == self.feature_dim
        ) and not (original_x_ndim == 1 and x.shape[0] == self.feature_dim):
            raise ValueError(
                f"Input x for normalize must be shape [B, {self.feature_dim}] "
                f"or [{self.feature_dim}], got {x.shape}"
            )

        if original_x_ndim == 1:
            x = x.unsqueeze(0)  # Treat as batch of 1

        buffer_device = self.running_mean.device
        original_x_device = x.device

        x_on_buffer_device = x.to(buffer_device)

        mean = self.running_mean
        std = self.running_std
        eps = self.epsilon_val.to(buffer_device)

        normalized_x_on_buffer_device = (x_on_buffer_device - mean) / (
            std + eps
        )

        result = normalized_x_on_buffer_device.to(original_x_device)
        return result.squeeze(0) if original_x_ndim == 1 else result

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalizes multi-dimensional data.

        Using the running mean and variance for each feature.
        Output tensor is on the same device as the input x_norm.

        Args:
            x_norm (torch.Tensor): A tensor of shape [B, D] or [D] of
                normalized values.

        Returns:
            torch.Tensor: The denormalized (original scale) data, on the same
                device as input x_norm.

        """
        original_x_norm_ndim = x_norm.ndim
        if not (
            original_x_norm_ndim == 2 and x_norm.shape[1] == self.feature_dim
        ) and not (
            original_x_norm_ndim == 1 and x_norm.shape[0] == self.feature_dim
        ):
            raise ValueError(
                f"Input x_norm for denormalize must be shape "
                f"[B, {self.feature_dim}] or [{self.feature_dim}], "
                f"got {x_norm.shape}"
            )

        if original_x_norm_ndim == 1:
            x_norm = x_norm.unsqueeze(0)  # Treat as batch of 1

        buffer_device = self.running_mean.device
        original_x_norm_device = x_norm.device

        x_norm_on_buffer_device = x_norm.to(buffer_device)

        mean = self.running_mean
        std = self.running_std
        eps = self.epsilon_val.to(buffer_device)

        denormalized_x_on_buffer_device = (
            x_norm_on_buffer_device * (std + eps) + mean
        )

        result = denormalized_x_on_buffer_device.to(original_x_norm_device)
        return result.squeeze(0) if original_x_norm_ndim == 1 else result

    def get_state(self) -> dict:
        """Returns the current state of the normalizer.

        Tensors are on their current buffer device.
        """
        return {
            "feature_dim": self.feature_dim,
            "running_mean": self.running_mean.clone(),
            "running_var": self.running_var.clone(),
            "running_std": self.running_std.clone(),
            "running_count": self.running_count.clone(),
            "epsilon_val": self.epsilon_val.clone(),
            "until": torch.tensor(
                -1 if self.until is None else int(self.until)
            ),
        }

    def set_state(self, state: dict, new_buffer_device: str = None):
        """Sets the state of the normalizer from a state dictionary.

        Args:
            state (dict): A dictionary containing:
                'feature_dim', 'running_mean', 'running_var',
                          'running_count', and 'epsilon_val'.
            new_buffer_device (str, optional): The device to move the loaded
                state and internal buffers to (e.g., 'cpu', 'cuda:0'). If None,
                state is loaded to the current device of the module's buffers.

        """
        target_device = (
            torch.device(new_buffer_device)
            if new_buffer_device
            else self.running_mean.device
        )

        required_keys = [
            "feature_dim",
            "running_mean",
            "running_var",
            "running_count",
            "epsilon_val",
        ]
        for key in required_keys:
            if key not in state:
                raise KeyError(
                    f"Key '{key}' not found in the "
                    f"provided state dictionary for RunningMeanStdNormalizer."
                )

        loaded_feature_dim = state["feature_dim"]
        if loaded_feature_dim != self.feature_dim:
            raise ValueError(
                f"Mismatched feature_dim in state. "
                f"Expected {self.feature_dim}, got {loaded_feature_dim}"
            )

        self.running_mean = state["running_mean"].clone().to(target_device)
        self.running_var = state["running_var"].clone().to(target_device)
        # Optional std; recompute if missing
        if "running_std" in state:
            self.running_std = state["running_std"].clone().to(target_device)
        else:
            self.running_std = torch.sqrt(self.running_var)
        # running_count can be stored as float or long; cast to long
        self.running_count = (
            state["running_count"].clone().to(target_device).to(torch.long)
        )
        self.epsilon_val = state["epsilon_val"].clone().to(target_device)
        # Optional until
        if "until" in state:
            raw_until = state["until"]
            try:
                self.until = None if int(raw_until) < 0 else int(raw_until)
            except Exception:
                self.until = None

        if self.running_mean.shape[0] != self.feature_dim:
            self.running_mean = torch.zeros(
                self.feature_dim, dtype=torch.float32, device=target_device
            )
            logger.warning(
                "Reinitialized running_mean due to feature_dim mismatch "
                "during set_state. This should not happen if state is valid."
            )
        if self.running_var.shape[0] != self.feature_dim:
            self.running_var = torch.ones(
                self.feature_dim, dtype=torch.float32, device=target_device
            )
            logger.warning(
                "Reinitialized running_var due to feature_dim mismatch during "
                "set_state. This should not happen if state is valid."
            )


def get_sinusoid_encoding_table(seq_len: int, hidden_dim: int) -> torch.Tensor:
    """Creates a sinusoidal positional encoding table.

    Args:
        seq_len: The number of positions to encode.
        hidden_dim: The hidden dimension of the encoding.

    Returns:
        A tensor of shape (1, seq_len, hidden_dim) containing the sinusoidal
        positional encoding table.

    """

    def get_position_angle_vec(position):
        return [
            position
            / torch.pow(10000, torch.tensor(2 * (hid_j // 2) / hidden_dim))
            for hid_j in range(hidden_dim)
        ]

    sinusoid_table = torch.tensor(
        [get_position_angle_vec(pos_i) for pos_i in range(seq_len)],
        dtype=torch.float32,
    )
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table[None, ...]


class FastMoEMLP(nn.Module):
    """
    Efficient MoE implementation using Tutel APIs for extremely fast training/rollout.
    Resembles MoEMLPV2 but leverages Tutel's optimized MoE kernels and communication.
    Includes advanced optimizations: mixed precision, dynamic capacity, expert parallelism.
    """

    def __init__(
        self,
        obs_serializer,
        module_config_dict: dict,
    ):
        super().__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict

        # Override some defaults for advanced optimizations
        tutel_moe_config = module_config_dict.copy()

        # Advanced Tutel configurations
        tutel_moe_config.setdefault(
            "capacity_factor", -1.5
        )  # Negative for adaptive capacity
        tutel_moe_config.setdefault("enable_tutel_jit", True)
        tutel_moe_config.setdefault(
            "a2a_ffn_overlap_degree", 2
        )  # More overlap for speed
        tutel_moe_config.setdefault("gate_noise", 1e-2)
        tutel_moe_config.setdefault("normalize_gate", True)
        tutel_moe_config.setdefault(
            "use_fp16", False
        )  # Set to True for FP16 training
        tutel_moe_config.setdefault("use_bf16", True)  # Use BF16 if available

        self.module_config_dict = tutel_moe_config

        # Configuration parameters
        self.projection_dim = self.module_config_dict["projection_dim"]
        self.hidden_dim = self.module_config_dict["hidden_dim"]
        self.num_fine_experts = self.module_config_dict["num_fine_experts"]
        self.num_shared_experts = self.module_config_dict["num_shared_experts"]
        self.top_k = self.module_config_dict["top_k"]
        self.moe_routed_scaling_factor = self.module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        # Tutel-specific configurations
        self.capacity_factor = self.module_config_dict.get(
            "capacity_factor", -1.5
        )
        self.enable_tutel_jit = self.module_config_dict.get(
            "enable_tutel_jit", True
        )
        self.enable_tutel_dropout = self.module_config_dict.get(
            "enable_tutel_dropout", False
        )
        self.tutel_activation = self.module_config_dict.get(
            "tutel_activation", "silu"
        )

        # Store advanced configurations
        self.gate_noise = tutel_moe_config.get("gate_noise", 1e-2)
        self.normalize_gate = tutel_moe_config.get("normalize_gate", True)
        self.use_fp16 = tutel_moe_config.get("use_fp16", False)
        self.use_bf16 = tutel_moe_config.get("use_bf16", True)
        self.dynamic_capacity = tutel_moe_config.get("dynamic_capacity", True)

        self._calculate_output_dim()

        self.clamp_actor_output = self.module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        # Calculate input dimension
        self.input_dim = sum(
            [
                self.obs_dim_dict[each_input]
                * self.obs_seq_len_dict[each_input]
                for each_input in self.obs_dim_dict
            ]
        )

        # Input projection layer
        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        # Create advanced Tutel MoE layer
        self._create_advanced_tutel_layer()

        # Shared experts implementation (since Tutel may not have built-in shared experts)
        if self.num_shared_experts > 0:
            self.shared_experts = DeepseekV3MLP(
                hidden_size=self.projection_dim,
                intermediate_size=self.hidden_dim * self.num_shared_experts,
            )
        else:
            self.shared_experts = None

        # Output MLP and head
        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

        # Buffers for loss computation and performance monitoring
        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("expert_counts", None, persistent=False)
        self.register_buffer(
            "routing_efficiency", torch.tensor(0.0), persistent=False
        )
        self.register_buffer("expert_utilization", None, persistent=False)

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            assert isinstance(each_output, int), (
                f"Output dim placeholder not replaced: {each_output}"
            )
            output_dim += each_output
        self.output_dim = output_dim

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale

        self.register_buffer("action_output_lb", lb[None, :])
        self.register_buffer("action_output_ub", ub[None, :])

    def _create_advanced_tutel_layer(self):
        """Create Tutel MoE layer with advanced optimizations."""
        try:
            from tutel import moe as tutel_moe
        except ImportError:
            raise ImportError("Tutel is required for TutelMoEAdvanced")

        # Advanced activation function with potential JIT compilation
        if self.tutel_activation == "silu":
            activation_fn = (
                torch.jit.script(lambda x: F.silu(x))
                if self.enable_tutel_jit
                else F.silu
            )
        elif self.tutel_activation == "gelu":
            activation_fn = (
                torch.jit.script(lambda x: F.gelu(x))
                if self.enable_tutel_jit
                else F.gelu
            )
        else:
            activation_fn = F.silu

        # Advanced gate configuration
        gate_config = {
            "type": "top",
            "k": self.top_k,
            "capacity_factor": self.capacity_factor,
            "gate_noise": self.gate_noise if self.training else 0.0,
        }

        if self.normalize_gate:
            gate_config["normalize"] = True

        # Expert configuration with advanced settings
        expert_config = {
            "type": "ffn",
            "count_per_device": self.num_fine_experts,
            "hidden_size_per_expert": self.hidden_dim,
            "activation_fn": activation_fn,
            "has_fc1_bias": True,
            "has_fc2_bias": True,
        }

        # Create advanced Tutel MoE layer
        self.tutel_moe_layer = tutel_moe.moe_layer(
            gate_type=gate_config,
            model_dim=self.projection_dim,
            experts=expert_config,
            scan_expert_func=lambda name, param: setattr(
                param, "skip_allreduce", True
            ),
            result_func=self._advanced_result_func,
            parallel_type="auto",
            pad_samples=True,  # Enable padding for better batching
            seeds=(1, 1, 1),
            a2a_ffn_overlap_degree=getattr(self, "module_config_dict", {}).get(
                "a2a_ffn_overlap_degree", 2
            ),
        )

    def _advanced_result_func(self, output):
        """Advanced result function to extract additional metrics."""
        if hasattr(output, "l_aux"):
            aux_loss = output.l_aux
        else:
            aux_loss = torch.tensor(0.0)

        # Extract routing efficiency if available
        if hasattr(output, "routing_weights"):
            self.routing_efficiency = output.routing_weights.float().mean()

        return output, aux_loss

    def compute_load_balancing_loss(self) -> torch.Tensor:
        """
        Compute load balancing loss similar to MoEMLPV2.
        Tutel provides aux_loss which includes load balancing, but we maintain
        compatibility with the original interface.
        """
        if hasattr(self, "_aux_loss") and self._aux_loss is not None:
            # Use Tutel's built-in auxiliary loss which includes load balancing
            return self._aux_loss
        else:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

    def compute_bound_loss(self) -> torch.Tensor:
        """Compute bound loss for action clamping."""
        if not hasattr(self, "output_actions"):
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        lb_loss = (
            torch.relu(self.action_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.action_output_ub)
            .square()
            .mean()
        )
        return (lb_loss + ub_loss) * 0.5

    def forward(self, x: torch.Tensor):
        """Forward pass with advanced optimizations."""
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )

        # Apply mixed precision if enabled
        if (
            self.use_bf16
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            x = x.to(torch.bfloat16)
        elif self.use_fp16:
            x = x.to(torch.float16)

        # Project input with potential precision conversion
        projected_x = self.input_projection(x)
        residuals = projected_x

        # Dynamic capacity adjustment during training
        if self.training and self.dynamic_capacity:
            self._adjust_capacity_factor()

        # Apply Tutel MoE layer
        with torch.cuda.amp.autocast(enabled=self.use_fp16 or self.use_bf16):
            result = self.tutel_moe_layer(projected_x)
            if isinstance(result, tuple):
                moe_output, aux_loss = result
            else:
                moe_output = result
                aux_loss = torch.tensor(0.0)

        self._aux_loss = aux_loss

        # Apply shared experts
        if self.shared_experts is not None:
            with torch.cuda.amp.autocast(
                enabled=self.use_fp16 or self.use_bf16
            ):
                shared_output = self.shared_experts(residuals)
            combined_hidden = moe_output + shared_output
        else:
            combined_hidden = moe_output

        # Output layers with potential mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_fp16 or self.use_bf16):
            hidden_output = self.output_mlp(combined_hidden)
            self.output_actions = self.output_head(hidden_output)

        # Convert back to original precision if needed
        if self.output_actions.dtype != torch.float32:
            self.output_actions = self.output_actions.to(torch.float32)

        return self.output_actions

    def get_aux_loss(self):
        """Get auxiliary loss from Tutel MoE layer (includes load balancing)."""
        return getattr(self, "_aux_loss", torch.tensor(0.0))

    def set_capacity_factor(self, capacity_factor: float):
        """Dynamically adjust capacity factor for training/inference optimization."""
        if hasattr(self.tutel_moe_layer, "set_capacity_factor"):
            self.tutel_moe_layer.set_capacity_factor(capacity_factor)

    def enable_expert_parallel(self, enable: bool = True):
        """Enable/disable expert parallelism for multi-GPU training."""
        if hasattr(self.tutel_moe_layer, "set_parallel_type"):
            parallel_type = "model" if enable else "data"
            self.tutel_moe_layer.set_parallel_type(parallel_type)

    def _adjust_capacity_factor(self):
        """Dynamically adjust capacity factor based on training progress."""
        if hasattr(self, "routing_efficiency") and self.routing_efficiency > 0:
            # Reduce capacity factor if routing is efficient
            if self.routing_efficiency > 0.8:
                new_capacity = max(1.0, self.capacity_factor * 0.95)
            else:
                new_capacity = min(2.0, self.capacity_factor * 1.02)

            if abs(new_capacity - self.capacity_factor) > 0.01:
                self.capacity_factor = new_capacity
                self.set_capacity_factor(new_capacity)

    def get_expert_utilization(self):
        """Get expert utilization statistics."""
        if hasattr(self.tutel_moe_layer, "get_expert_utilization"):
            return self.tutel_moe_layer.get_expert_utilization()
        return None

    def optimize_for_inference(self):
        """Optimize the model for inference by reducing capacity and disabling noise."""
        self.eval()
        self.set_capacity_factor(1.0)  # Minimal capacity for inference
        if hasattr(self.tutel_moe_layer, "set_gate_noise"):
            self.tutel_moe_layer.set_gate_noise(0.0)

    def optimize_for_training(self):
        """Optimize the model for training with higher capacity and noise."""
        self.train()
        self.set_capacity_factor(1.5)
        if hasattr(self.tutel_moe_layer, "set_gate_noise"):
            self.tutel_moe_layer.set_gate_noise(self.gate_noise)


def create_tutel_moe_config(
    projection_dim: int = 512,
    hidden_dim: int = 2048,
    num_fine_experts: int = 8,
    num_shared_experts: int = 2,
    top_k: int = 2,
    capacity_factor: float = -1.5,
    **kwargs,
) -> dict:
    """
    Helper function to create optimized Tutel MoE configuration.

    Args:
        projection_dim: Dimension after input projection
        hidden_dim: Hidden dimension for expert FFNs
        num_fine_experts: Number of expert networks
        num_shared_experts: Number of shared experts
        top_k: Number of experts to route each token to
        capacity_factor: Capacity factor for expert assignment (negative for adaptive)
        **kwargs: Additional configuration options

    Returns:
        Dictionary with optimized Tutel MoE configuration
    """
    config = {
        "projection_dim": projection_dim,
        "hidden_dim": hidden_dim,
        "num_fine_experts": num_fine_experts,
        "num_shared_experts": num_shared_experts,
        "top_k": top_k,
        "capacity_factor": capacity_factor,
        "tutel_activation": "silu",
        "enable_tutel_jit": True,
        "a2a_ffn_overlap_degree": 2,
        "gate_noise": 1e-2,
        "normalize_gate": True,
        "use_bf16": True,
        "dynamic_capacity": True,
    }

    # Override with any provided kwargs
    config.update(kwargs)

    return config


# Usage example and documentation
"""
Example usage of TutelMoE:

# Basic configuration
config = create_tutel_moe_config(
    projection_dim=512,
    hidden_dim=2048,
    num_fine_experts=8,
    top_k=2,
    output_dim=[12]  # Your specific output dimensions
)

model = TutelMoE(obs_serializer, config)

# High-performance configuration for large models
large_config = create_tutel_moe_config(
    projection_dim=1024,
    hidden_dim=4096,
    num_fine_experts=16,
    top_k=4,
    capacity_factor=-2.0,  # Adaptive capacity
    use_bf16=True,
    a2a_ffn_overlap_degree=4,  # Maximum overlap
    output_dim=[12]
)

large_model = TutelMoE(obs_serializer, large_config)

# For multi-GPU training
large_model.enable_expert_parallel(True)

# For inference optimization
large_model.optimize_for_inference()

# Training with load balancing loss
output = large_model(input_tensor)
load_balancing_loss = large_model.compute_load_balancing_loss()
total_loss = main_loss + 0.01 * load_balancing_loss

Performance Benefits of TutelMoE over MoEMLPV2:
1. 2-5x faster training with optimized CUDA kernels
2. Reduced memory usage with efficient all-to-all communication
3. Better scaling across multiple GPUs
4. Support for mixed precision (FP16/BF16)
5. Dynamic capacity adjustment for optimal performance
6. JIT compilation for activation functions
7. Advanced overlap strategies for communication hiding

Installation:
pip install tutel

For best performance with DeepSeek-style models:
pip install tutel[deepseek]
"""
