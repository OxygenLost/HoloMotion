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

import copy
import json
import math
import os
from omegaconf import OmegaConf
import weakref
from collections import defaultdict
from dataclasses import MISSING
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from loguru import logger

from holomotion.src.env.isaaclab_components import (
    ActionsCfg,
    ObservationsCfg,
    RewardsCfg,
    MotionTrackingSceneCfg,
    TerminationsCfg,
)
from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    MotionCommandCfg,
)
from holomotion.src.modules.agent_modules import ObsSeqSerializer
from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib


class MotionTrackingEnv:
    """IsaacLab-based Motion Tracking Environment.

    This environment integrates motion tracking capabilities with IsaacLab's
    manager-based architecture, supporting curriculum learning, domain randomization,
    and various termination conditions.

    This is a wrapper class that handles Isaac Sim initialization and delegates
    to an internal ManagerBasedRLEnv instance.
    """

    def __init__(
        self,
        config,
        device: torch.device = None,
        log_dir: str = None,
        render_mode: str | None = None,
        headless: bool = True,
    ):
        """Initialize the Motion Tracking Environment.

        Args:
            config: Configuration for the environment
            device: Device for tensor operations
            log_dir: Logging directory
            render_mode: Render mode for the environment
            headless: Whether to run in headless mode
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self._seed = self.config.get("seed", 666)
        self._device = device

        self.log_dir = log_dir
        self.headless = headless
        self.init_done = False
        self.is_evaluating = False
        self.render_mode = render_mode

        self._init_motion_tracking_components()
        self._init_isaaclab_env(holomotion_env_config=config)

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def _init_isaaclab_env(self, holomotion_env_config):
        _device = self._device

        @configclass
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            policy_freq: int = 50
            sim_freq: int = 200
            decimation: int = int(sim_freq / policy_freq)
            episode_length_s: float = 1000.0
            action_scale: float = 0.25
            device = _device
            dt = 1.0 / sim_freq

            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=PhysxCfg(
                    bounce_threshold_velocity=0.2,
                    gpu_max_rigid_patch_count=int(
                        10 * 2**15
                    ),  # two times the default value
                ),
                device="cuda" if _device is None else str(_device),
            )
            scene: MotionTrackingSceneCfg = MotionTrackingSceneCfg()
            viewer: ViewerCfg = ViewerCfg(origin_type="world")
            observations = ObservationsCfg()
            actions = ActionsCfg()
            rewards = RewardsCfg()
            terminations = TerminationsCfg()

            @configclass
            class CommandsCfg:
                ref_motion = MotionCommandCfg(
                    command_obs_name="bydmmc_ref_motion",
                    motion_lib_cfg=OmegaConf.to_container(
                        self.config.robot.motion,
                        resolve=True,
                    ),
                    process_id=self.config.process_id,
                    num_processes=self.config.num_processes,
                    resample_time_interval_s=self.config.resample_time_interval_s,
                    is_evaluating=self.is_evaluating,
                    n_fut_frames=self.n_fut_frames,
                    target_fps=self.target_fps,
                    asset_name="robot",
                    debug_vis=True,
                    root_pose_perturb_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (-0.01, 0.01),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-0.2, 0.2),
                    },
                    root_vel_perturb_range={
                        "x": (-0.3, 0.3),
                        "y": (-0.3, 0.3),
                        "z": (-0.1, 0.1),
                        "roll": (-0.3, 0.3),
                        "pitch": (-0.3, 0.3),
                        "yaw": (-0.4, 0.4),
                    },
                    dof_pos_perturb_range=(-0.1, 0.1),
                    dof_vel_perturb_range=(-1.0, 1.0),
                )

            commands = CommandsCfg()
            # events = EventsCfg()
            # curriculum = CurriculumCfg()

        isaac_lab_cfg = MotionTrackingEnvCfg()
        self._env = ManagerBasedRLEnv(isaac_lab_cfg, self.render_mode)
        return isaac_lab_cfg

    def _init_motion_tracking_components(self):
        self.num_extend_bodies = len(
            getattr(self.config, "robot", {})
            .get("motion", {})
            .get("extend_config", [])
        )
        self.n_fut_frames = getattr(self.config, "obs", {}).get(
            "n_fut_frames", 1
        )
        self.target_fps = getattr(self.config, "target_fps", 50)
        self._init_curriculum_settings()
        self._init_serializers()

    def step(self, actor_state: dict):
        return self._env.step(actor_state)

    def reset_idx(self, env_ids: torch.Tensor):
        return self._env.reset(seed=self._seed, env_ids=env_ids)

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._env.reset(seed=self._seed, env_ids=env_ids)

    def set_is_evaluating(self):
        logger.info("Setting environment to evaluation mode")
        self.is_evaluating = True

    def _init_serializers(self):
        if hasattr(self.config, "obs"):
            obs_config = self.config.obs

            if obs_config.get("serialization_schema", None):
                self.obs_serializer = ObsSeqSerializer(
                    obs_config.serialization_schema
                )

            if obs_config.get("critic_serialization_schema", None):
                self.critic_obs_serializer = ObsSeqSerializer(
                    obs_config.critic_serialization_schema
                )

            if obs_config.get("teacher_serialization_schema", None):
                self.teacher_obs_serializer = ObsSeqSerializer(
                    obs_config.teacher_serialization_schema
                )

            if obs_config.get("command_serilization_schema", None):
                self.command_obs_serializer = ObsSeqSerializer(
                    obs_config.command_serilization_schema
                )

    def _init_curriculum_settings(self):
        self.entropy_coef = getattr(self.config, "init_entropy_coef", 0.01)

        if hasattr(self.config, "termination"):
            term_config = self.config.termination
            term_curriculum = getattr(
                self.config, "termination_curriculum", {}
            )
            term_scales = getattr(self.config, "termination_scales", {})

            if term_config.get(
                "terminate_when_motion_far", False
            ) and term_curriculum.get(
                "terminate_when_motion_far_curriculum", False
            ):
                self.terminate_when_motion_far_threshold = term_curriculum.get(
                    "terminate_when_motion_far_initial_threshold", 0.5
                )
            else:
                self.terminate_when_motion_far_threshold = term_scales.get(
                    "termination_motion_far_threshold", 0.5
                )

            if term_config.get(
                "terminate_when_joint_far", False
            ) and term_curriculum.get(
                "terminate_when_joint_far_curriculum", False
            ):
                self.terminate_when_joint_far_threshold = term_curriculum.get(
                    "terminate_when_joint_far_initial_threshold", 1.0
                )
            else:
                self.terminate_when_joint_far_threshold = term_scales.get(
                    "terminate_when_joint_far_threshold", 1.0
                )

            self.terminate_when_joint_far_patience_steps = term_config.get(
                "terminate_when_joint_far_patience_steps", 1
            )
            self.terminate_when_motion_far_patience_steps = term_config.get(
                "terminate_when_motion_far_patience_steps", 1
            )
        else:
            self.terminate_when_motion_far_threshold = 0.5
            self.terminate_when_joint_far_threshold = 1.0
            self.terminate_when_joint_far_patience_steps = 1
            self.terminate_when_motion_far_patience_steps = 1
