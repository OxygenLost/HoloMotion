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
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from loguru import logger
from omegaconf import OmegaConf
from easydict import EasyDict

from holomotion.src.env.isaaclab_components import (
    ObservationsCfg,
    build_actions_config,
    build_commands_config,
    build_domain_rand_config,
    build_observations_config,
    build_rewards_config,
    build_scene_config,
    build_terminations_config,
    CommandsCfg,
    RewardsCfg,
    TerminationsCfg,
    EventsCfg,
    ActionsCfg,
    MotionTrackingSceneCfg,
)
from holomotion.src.modules.agent_modules import ObsSeqSerializer


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
        self._init_isaaclab_env()

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def _init_isaaclab_env(self):
        _device = self._device

        # curriculum = CurriculumCfg()

        _sim_config_dict = EasyDict(
            OmegaConf.to_container(self.config.simulator, resolve=True)
        )
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

        @configclass
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            scene_config_dict = {
                "num_envs": self.config.num_envs,
                "env_spacing": self.config.env_spacing,
                "replicate_physics": self.config.replicate_physics,
                "robot": _robot_config_dict,
                "terrain": {
                    "terrain_type": "plane",
                    "static_friction": 1.0,
                    "dynamic_friction": 1.0,
                },
                "lighting": {
                    "distant_light_intensity": 3000.0,
                    "dome_light_intensity": 1000.0,
                },
                "contact_sensor": {
                    "history_length": 3,
                    "force_threshold": 10.0,
                    "debug_vis": False,
                },
            }

            command_config_dict = {
                "ref_motion": {
                    "type": "MotionCommandCfg",
                    "params": {
                        "command_obs_name": "bydmmc_ref_motion",
                        "motion_lib_cfg": OmegaConf.to_container(
                            self.config.robot.motion,
                            resolve=True,
                        ),
                        "process_id": self.config.process_id,
                        "num_processes": self.config.num_processes,
                        "resample_time_interval_s": self.config.resample_time_interval_s,
                        "is_evaluating": self.is_evaluating,
                        "n_fut_frames": self.n_fut_frames,
                        "target_fps": self.target_fps,
                        "anchor_bodylink_name": self.config.robot.motion.get(
                            "anchor_bodylink_name", "torso_link"
                        ),
                        "asset_name": "robot",
                        "debug_vis": True,
                        "root_pose_perturb_range": {
                            "x": (-0.05, 0.05),
                            "y": (-0.05, 0.05),
                            "z": (-0.01, 0.01),
                            "roll": (-0.1, 0.1),
                            "pitch": (-0.1, 0.1),
                            "yaw": (-0.2, 0.2),
                        },
                        "root_vel_perturb_range": {
                            "x": (-0.3, 0.3),
                            "y": (-0.3, 0.3),
                            "z": (-0.1, 0.1),
                            "roll": (-0.3, 0.3),
                            "pitch": (-0.3, 0.3),
                            "yaw": (-0.4, 0.4),
                        },
                        "dof_pos_perturb_range": (-0.1, 0.1),
                        "dof_vel_perturb_range": (-1.0, 1.0),
                    },
                }
            }

            terminations_config_dict = {
                "motion_end": {"time_out": True},
                "anchor_ref_z_far": {
                    "params": {
                        "threshold": 0.25,
                    }
                },
                "ref_gravity_projection_far": {
                    "params": {
                        "threshold": 0.8,
                    }
                },
                "keybody_ref_z_far": {
                    "params": {
                        "threshold": 0.25,
                        "keybody_idxs": [0, 1],
                    },
                },
            }

            domain_rand_config_dict = {
                "default_dof_pos_bias": {
                    "mode": "startup",
                    "params": {
                        "joint_names": [".*"],
                        "pos_distribution_params": (-0.01, 0.01),
                        "operation": "add",
                        "distribution": "uniform",
                    },
                },
                "rigid_body_com": {
                    "mode": "startup",
                    "params": {
                        "body_names": "torso_link",
                        "com_range": {
                            "x": (-0.025, 0.025),
                            "y": (-0.05, 0.05),
                            "z": (-0.05, 0.05),
                        },
                    },
                },
                "rigid_body_material": {
                    "mode": "startup",
                    "params": {
                        "body_names": ".*",
                        "static_friction_range": (0.3, 1.6),
                        "dynamic_friction_range": (0.3, 1.2),
                        "restitution_range": (0.0, 0.5),
                        "num_buckets": 64,
                    },
                },
                "push_by_setting_velocity": {
                    "mode": "interval",
                    "interval_range_s": (1.0, 3.0),
                    "params": {
                        "velocity_range": {
                            "x": (-0.5, 0.5),
                            "y": (-0.5, 0.5),
                            "z": (-0.2, 0.2),
                            "roll": (-0.52, 0.52),
                            "pitch": (-0.52, 0.52),
                            "yaw": (-0.78, 0.78),
                        },
                    },
                },
            }

            actions_config_dict = {
                "dof_pos": {
                    "type": "joint_position",
                    "params": {
                        "asset_name": "robot",
                        "joint_names": [".*"],
                        "use_default_offset": True,
                    },
                }
            }

            episode_length_s: int = 1000
            sim_freq = _sim_config_dict.get("sim_freq", 200)
            dt = 1.0 / sim_freq
            decimation = _sim_config_dict.get("control_decimation", 4)
            physx_config = _sim_config_dict.get("physx", {})
            physx = PhysxCfg(
                bounce_threshold_velocity=physx_config.get(
                    "bounce_threshold_velocity", 0.2
                ),
                gpu_max_rigid_patch_count=physx_config.get(
                    "gpu_max_rigid_patch_count", int(10 * 2**15)
                ),
            )

            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device=_device,
            )

            scene: MotionTrackingSceneCfg = build_scene_config(
                scene_config_dict
            )
            viewer: ViewerCfg = ViewerCfg(origin_type="world")
            commands: CommandsCfg = build_commands_config(command_config_dict)
            observations: ObservationsCfg = build_observations_config(
                _obs_config_dict.obs_groups
            )
            rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)
            terminations: TerminationsCfg = build_terminations_config(
                terminations_config_dict
            )
            events: EventsCfg = build_domain_rand_config(
                domain_rand_config_dict
            )
            actions: ActionsCfg = build_actions_config(actions_config_dict)

        self._env = ManagerBasedRLEnv(MotionTrackingEnvCfg(), self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env

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
