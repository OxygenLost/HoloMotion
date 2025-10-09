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

import torch
import time
import yaml
from functools import wraps
from easydict import EasyDict
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_yaml
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.env.isaaclab_components import (
    ActionsCfg,
    CommandsCfg,
    EventsCfg,
    MotionTrackingSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    build_actions_config,
    build_commands_config,
    build_domain_rand_config,
    build_observations_config,
    build_rewards_config,
    build_scene_config,
    build_terminations_config,
)
from holomotion.src.modules.agent_modules import ObsSeqSerializer
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg, EventTermCfg
from isaaclab.utils import configclass


from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers import EventTermCfg as EventTerm


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING, Literal
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(
        asset.data.default_joint_pos[0]
    )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device
        )

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("dof_pos")._offset[
            env_ids, joint_ids
        ] = pos


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
        self._seed = self.config.get("seed", 42)
        self._device = device

        self.log_dir = log_dir
        self.headless = headless
        self.init_done = False
        self.is_evaluating = False
        self.render_mode = render_mode

        # self._init_motion_tracking_components()
        self._init_isaaclab_env()
        self._init_serializers()

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def _init_isaaclab_env(self):
        _device = self._device

        # curriculum = CurriculumCfg()

        _robot_config_dict = EasyDict(
            OmegaConf.to_container(self.config.robot, resolve=True)
        )
        _terrain_config_dict = EasyDict(
            OmegaConf.to_container(self.config.terrain, resolve=True)
        )
        _obs_config_dict = EasyDict(
            OmegaConf.to_container(self.config.obs, resolve=True)
        )
        _rewards_config_dict = EasyDict(
            OmegaConf.to_container(self.config.rewards, resolve=True)
        )
        _domain_rand_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.domain_rand,
                resolve=True,
            )
        )
        _terminations_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.terminations,
                resolve=True,
            )
        )
        _scene_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.scene,
                resolve=True,
            )
        )
        _commands_config_dict = OmegaConf.to_container(
            self.config.commands,
            resolve=True,
        )

        _simulation_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.simulation,
                resolve=True,
            )
        )
        _actions_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.actions,
                resolve=True,
            )
        )

        @configclass
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            scene_config_dict = {
                "num_envs": self.config.num_envs,
                "env_spacing": self.config.env_spacing,
                "replicate_physics": self.config.replicate_physics,
                "robot": _robot_config_dict,
                "terrain": _terrain_config_dict,
                "lighting": _scene_config_dict.lighting,
                "contact_sensor": _scene_config_dict.contact_sensor,
            }

            _commands_config_dict["ref_motion"]["params"].update(
                {
                    "process_id": self.config.process_id,
                    "num_processes": self.config.num_processes,
                    "is_evaluating": self.is_evaluating,
                }
            )

            decimation: int = _simulation_config_dict.control_decimation
            episode_length_s: int = _simulation_config_dict.episode_length_s
            sim_freq = _simulation_config_dict.sim_freq
            dt = 1.0 / sim_freq
            physx = PhysxCfg(
                bounce_threshold_velocity=_simulation_config_dict.physx.bounce_threshold_velocity,
                gpu_max_rigid_patch_count=_simulation_config_dict.physx.gpu_max_rigid_patch_count,
                enable_stabilization=True,
            )
            seed = self.config.get("seed", 666)

            scene: MotionTrackingSceneCfg = build_scene_config(
                scene_config_dict
            )

            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device=_device,
                enable_scene_query_support=True,
            )
            sim.physics_material = scene.terrain.physics_material

            viewer: ViewerCfg = ViewerCfg(origin_type="world")
            commands: CommandsCfg = build_commands_config(
                _commands_config_dict
            )
            observations: ObservationsCfg = build_observations_config(
                _obs_config_dict.obs_groups
            )
            rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)
            terminations: TerminationsCfg = build_terminations_config(
                _terminations_config_dict
            )
            events: EventsCfg = build_domain_rand_config(
                _domain_rand_config_dict
            )
            # events: EventCfg = EventCfg()
            actions: ActionsCfg = build_actions_config(_actions_config_dict)
            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device="cuda",
                enable_scene_query_support=True,
            )
            sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
            sim.physx.enable_stabilization = True
            sim.physics_material = scene.terrain.physics_material

        isaaclab_env_cfg = MotionTrackingEnvCfg()

        # dump_yaml("./isaaclab_env_cfg.yaml", isaaclab_env_cfg)

        self._env = ManagerBasedRLEnv(isaaclab_env_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env

    def _init_motion_tracking_components(self):
        self.num_extend_bodies = len(
            getattr(self.config, "robot", {})
            .get("motion", {})
            .get("extend_config", [])
        )
        self.n_fut_frames = self.config.commands.ref_motion.params.n_fut_frames
        self.target_fps = self.config.commands.ref_motion.params.target_fps
        self._init_serializers()

    def step(self, actor_state: dict):
        obs_dict, rewards, terminated, time_outs, infos = self._env.step(
            actor_state
        )
        # IsaacLab separates terminated vs time_outs, combine them for consistency
        dones = terminated | time_outs
        return obs_dict, rewards, dones, time_outs, infos

    def reset_idx(self, env_ids: torch.Tensor):
        return self._env.reset(seed=self._seed, env_ids=env_ids)

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        out = self._env.reset(seed=self._seed, env_ids=env_ids)
        return out

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
