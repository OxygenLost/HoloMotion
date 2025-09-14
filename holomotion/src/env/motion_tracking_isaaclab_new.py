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
import os
import weakref
from collections import defaultdict
from typing import Dict, List, Optional, Literal, Sequence, Any
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
import json

from dataclasses import MISSING
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
        self.debug_viz = True
        self.is_evaluating = False
        self.render_mode = render_mode

        self._setup_isaac_sim()
        self._init_isaaclab_env(holomotion_env_config=config)
        self._init_motion_tracking_components()

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def _setup_isaac_sim(self):
        """Setup Isaac Sim and import IsaacLab packages."""
        from isaaclab.app import AppLauncher

        # Set environment variables for headless mode
        if self.headless:
            os.environ["ISAAC_SIM_HEADLESS"] = "1"
            try:
                from pyvirtualdisplay import Display

                display = Display(visible=0, size=(1024, 768))
                display.start()
            except ImportError:
                logger.warning(
                    "pyvirtualdisplay not available, skipping virtual display setup"
                )

        # Launch Isaac Sim app
        app_launcher_flags = {
            "headless": self.headless,
        }
        self._sim_app_launcher = AppLauncher(app_launcher_flags)
        self._sim_app = self._sim_app_launcher.app

        logger.info("Isaac Sim initialized successfully")

    def _init_isaaclab_env(self, holomotion_env_config):
        from isaaclab.assets import Articulation
        from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
        from isaaclab.sim import SimulationCfg, PhysxCfg
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils import configclass
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.envs import ManagerBasedRLEnv
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import (
            ObservationTermCfg,
            ObservationGroupCfg,
            ActionTermCfg,
            RewardTermCfg,
            TerminationTermCfg,
            SceneEntityCfg,
            CommandTerm,
            CommandTermCfg,
        )
        from isaaclab.envs.mdp.actions import JointEffortActionCfg
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
        from isaaclab.markers import (
            VisualizationMarkers,
            VisualizationMarkersCfg,
        )
        from isaaclab.markers.config import FRAME_MARKER_CFG
        from isaaclab.utils.math import (
            quat_apply,
            quat_error_magnitude,
            quat_from_euler_xyz,
            quat_inv,
            quat_mul,
            sample_uniform,
            yaw_quat,
        )

        urdf_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/assets/robots/unitree/G1/23dof/official_g1_23dof_rev_1_0.urdf"
        usd_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/assets/robots/unitree/G1/23dof"

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        logger.info(f"Using URDF path: {urdf_path}")
        logger.info(f"Using USD directory: {usd_dir}")

        @configclass
        class MotionCommandCfg(CommandTermCfg):
            """Configuration for the motion command."""

            class_type: type = MISSING
            asset_name: str = MISSING
            motion_file: str = MISSING
            anchor_body_name: str = MISSING
            body_names: list[str] = MISSING

            pose_range: dict[str, tuple[float, float]] = {}
            velocity_range: dict[str, tuple[float, float]] = {}
            joint_position_range: tuple[float, float] = (-0.52, 0.52)

            anchor_visualizer_cfg: VisualizationMarkersCfg = (
                FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
            )
            anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

            body_visualizer_cfg: VisualizationMarkersCfg = (
                FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
            )
            body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

        class MotionCommand(CommandTerm):
            cfg: MotionCommandCfg

            def __init__(
                self,
                cfg: MotionCommandCfg,
                env: ManagerBasedRLEnv,
                motion_lib: LmdbMotionLib,
            ):
                super().__init__(cfg, env)
                self._motion_lib = motion_lib
                self._env = env

                self.time_steps = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                self.body_pos_relative_w = torch.zeros(
                    self.num_envs, len(cfg.body_names), 3, device=self.device
                )
                self.body_quat_relative_w = torch.zeros(
                    self.num_envs, len(cfg.body_names), 4, device=self.device
                )
                self.body_quat_relative_w[:, :, 0] = 1.0

                self.metrics["error_anchor_pos"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_anchor_rot"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_anchor_lin_vel"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_anchor_ang_vel"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_body_pos"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_body_rot"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_joint_pos"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["error_joint_vel"] = torch.zeros(
                    self.num_envs, device=self.device
                )

            def _init_configs(self):
                self.robot: Articulation = self._env.scene[self.cfg.asset_name]
                self.robot_anchor_body_index = self.robot.body_names.index(
                    self.cfg.anchor_body_name
                )
                self.motion_anchor_body_index = self.cfg.body_names.index(
                    self.cfg.anchor_body_name
                )
                self.body_indexes = torch.tensor(
                    self.robot.find_bodies(
                        self.cfg.body_names, preserve_order=True
                    )[0],
                    dtype=torch.long,
                    device=self.device,
                )

            @property
            def command(
                self,
            ) -> (
                torch.Tensor
            ):  # TODO Consider again if this is the best observation
                return torch.cat([self.joint_pos, self.joint_vel], dim=1)

            @property
            def joint_pos(self) -> torch.Tensor:
                return self.motion.joint_pos[self.time_steps]

            @property
            def joint_vel(self) -> torch.Tensor:
                return self.motion.joint_vel[self.time_steps]

            @property
            def body_pos_w(self) -> torch.Tensor:
                return (
                    self.motion.body_pos_w[self.time_steps]
                    + self._env.scene.env_origins[:, None, :]
                )

            @property
            def body_quat_w(self) -> torch.Tensor:
                return self.motion.body_quat_w[self.time_steps]

            @property
            def body_lin_vel_w(self) -> torch.Tensor:
                return self.motion.body_lin_vel_w[self.time_steps]

            @property
            def body_ang_vel_w(self) -> torch.Tensor:
                return self.motion.body_ang_vel_w[self.time_steps]

            @property
            def anchor_pos_w(self) -> torch.Tensor:
                return (
                    self.motion.body_pos_w[
                        self.time_steps, self.motion_anchor_body_index
                    ]
                    + self._env.scene.env_origins
                )

            @property
            def anchor_quat_w(self) -> torch.Tensor:
                return self.motion.body_quat_w[
                    self.time_steps, self.motion_anchor_body_index
                ]

            @property
            def anchor_lin_vel_w(self) -> torch.Tensor:
                return self.motion.body_lin_vel_w[
                    self.time_steps, self.motion_anchor_body_index
                ]

            @property
            def anchor_ang_vel_w(self) -> torch.Tensor:
                return self.motion.body_ang_vel_w[
                    self.time_steps, self.motion_anchor_body_index
                ]

            @property
            def robot_joint_pos(self) -> torch.Tensor:
                return self.robot.data.joint_pos

            @property
            def robot_joint_vel(self) -> torch.Tensor:
                return self.robot.data.joint_vel

            @property
            def robot_body_pos_w(self) -> torch.Tensor:
                return self.robot.data.body_pos_w[:, self.body_indexes]

            @property
            def robot_body_quat_w(self) -> torch.Tensor:
                return self.robot.data.body_quat_w[:, self.body_indexes]

            @property
            def robot_body_lin_vel_w(self) -> torch.Tensor:
                return self.robot.data.body_lin_vel_w[:, self.body_indexes]

            @property
            def robot_body_ang_vel_w(self) -> torch.Tensor:
                return self.robot.data.body_ang_vel_w[:, self.body_indexes]

            @property
            def robot_anchor_pos_w(self) -> torch.Tensor:
                return self.robot.data.body_pos_w[
                    :, self.robot_anchor_body_index
                ]

            @property
            def robot_anchor_quat_w(self) -> torch.Tensor:
                return self.robot.data.body_quat_w[
                    :, self.robot_anchor_body_index
                ]

            @property
            def robot_anchor_lin_vel_w(self) -> torch.Tensor:
                return self.robot.data.body_lin_vel_w[
                    :, self.robot_anchor_body_index
                ]

            @property
            def robot_anchor_ang_vel_w(self) -> torch.Tensor:
                return self.robot.data.body_ang_vel_w[
                    :, self.robot_anchor_body_index
                ]

            def _update_metrics(self):
                self.metrics["error_anchor_pos"] = torch.norm(
                    self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
                )
                self.metrics["error_anchor_rot"] = quat_error_magnitude(
                    self.anchor_quat_w, self.robot_anchor_quat_w
                )
                self.metrics["error_anchor_lin_vel"] = torch.norm(
                    self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
                )
                self.metrics["error_anchor_ang_vel"] = torch.norm(
                    self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
                )

                self.metrics["error_body_pos"] = torch.norm(
                    self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
                ).mean(dim=-1)
                self.metrics["error_body_rot"] = quat_error_magnitude(
                    self.body_quat_relative_w, self.robot_body_quat_w
                ).mean(dim=-1)

                self.metrics["error_body_lin_vel"] = torch.norm(
                    self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
                ).mean(dim=-1)
                self.metrics["error_body_ang_vel"] = torch.norm(
                    self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
                ).mean(dim=-1)

                self.metrics["error_joint_pos"] = torch.norm(
                    self.joint_pos - self.robot_joint_pos, dim=-1
                )
                self.metrics["error_joint_vel"] = torch.norm(
                    self.joint_vel - self.robot_joint_vel, dim=-1
                )

            def _resample_command(self, env_ids: Sequence[int]):
                if len(env_ids) == 0:
                    return

                root_pos = self.body_pos_w[:, 0].clone()
                root_ori = self.body_quat_w[:, 0].clone()
                root_lin_vel = self.body_lin_vel_w[:, 0].clone()
                root_ang_vel = self.body_ang_vel_w[:, 0].clone()

                range_list = [
                    self.cfg.pose_range.get(key, (0.0, 0.0))
                    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
                ]
                ranges = torch.tensor(range_list, device=self.device)
                rand_samples = sample_uniform(
                    ranges[:, 0],
                    ranges[:, 1],
                    (len(env_ids), 6),
                    device=self.device,
                )
                root_pos[env_ids] += rand_samples[:, 0:3]
                orientations_delta = quat_from_euler_xyz(
                    rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
                )
                root_ori[env_ids] = quat_mul(
                    orientations_delta, root_ori[env_ids]
                )
                range_list = [
                    self.cfg.velocity_range.get(key, (0.0, 0.0))
                    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
                ]
                ranges = torch.tensor(range_list, device=self.device)
                rand_samples = sample_uniform(
                    ranges[:, 0],
                    ranges[:, 1],
                    (len(env_ids), 6),
                    device=self.device,
                )
                root_lin_vel[env_ids] += rand_samples[:, :3]
                root_ang_vel[env_ids] += rand_samples[:, 3:]

                joint_pos = self.joint_pos.clone()
                joint_vel = self.joint_vel.clone()

                joint_pos += sample_uniform(
                    *self.cfg.joint_position_range,
                    joint_pos.shape,
                    joint_pos.device,
                )
                soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[
                    env_ids
                ]
                joint_pos[env_ids] = torch.clip(
                    joint_pos[env_ids],
                    soft_joint_pos_limits[:, :, 0],
                    soft_joint_pos_limits[:, :, 1],
                )
                self.robot.write_joint_state_to_sim(
                    joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
                )
                self.robot.write_root_state_to_sim(
                    torch.cat(
                        [
                            root_pos[env_ids],
                            root_ori[env_ids],
                            root_lin_vel[env_ids],
                            root_ang_vel[env_ids],
                        ],
                        dim=-1,
                    ),
                    env_ids=env_ids,
                )

            def _update_command(self):
                self.time_steps += 1
                env_ids = torch.where(
                    self.time_steps >= self.motion.time_step_total
                )[0]
                self._resample_command(env_ids)

                anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
                    1, len(self.cfg.body_names), 1
                )
                anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
                    1, len(self.cfg.body_names), 1
                )
                robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[
                    :, None, :
                ].repeat(1, len(self.cfg.body_names), 1)
                robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[
                    :, None, :
                ].repeat(1, len(self.cfg.body_names), 1)

                delta_pos_w = robot_anchor_pos_w_repeat
                delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
                delta_ori_w = yaw_quat(
                    quat_mul(
                        robot_anchor_quat_w_repeat,
                        quat_inv(anchor_quat_w_repeat),
                    )
                )

                self.body_quat_relative_w = quat_mul(
                    delta_ori_w, self.body_quat_w
                )
                self.body_pos_relative_w = delta_pos_w + quat_apply(
                    delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
                )

                self.bin_failed_count = (
                    self.cfg.adaptive_alpha * self._current_bin_failed
                    + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
                )
                self._current_bin_failed.zero_()

            def _set_debug_vis_impl(self, debug_vis: bool):
                if debug_vis:
                    if not hasattr(self, "current_anchor_visualizer"):
                        self.current_anchor_visualizer = VisualizationMarkers(
                            self.cfg.anchor_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/anchor"
                            )
                        )
                        self.goal_anchor_visualizer = VisualizationMarkers(
                            self.cfg.anchor_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/anchor"
                            )
                        )

                        self.current_body_visualizers = []
                        self.goal_body_visualizers = []
                        for name in self.cfg.body_names:
                            self.current_body_visualizers.append(
                                VisualizationMarkers(
                                    self.cfg.body_visualizer_cfg.replace(
                                        prim_path="/Visuals/Command/current/"
                                        + name
                                    )
                                )
                            )
                            self.goal_body_visualizers.append(
                                VisualizationMarkers(
                                    self.cfg.body_visualizer_cfg.replace(
                                        prim_path="/Visuals/Command/goal/"
                                        + name
                                    )
                                )
                            )

                    self.current_anchor_visualizer.set_visibility(True)
                    self.goal_anchor_visualizer.set_visibility(True)
                    for i in range(len(self.cfg.body_names)):
                        self.current_body_visualizers[i].set_visibility(True)
                        self.goal_body_visualizers[i].set_visibility(True)

                else:
                    if hasattr(self, "current_anchor_visualizer"):
                        self.current_anchor_visualizer.set_visibility(False)
                        self.goal_anchor_visualizer.set_visibility(False)
                        for i in range(len(self.cfg.body_names)):
                            self.current_body_visualizers[i].set_visibility(
                                False
                            )
                            self.goal_body_visualizers[i].set_visibility(False)

            def _debug_vis_callback(self, event):
                if not self.robot.is_initialized:
                    return

                self.current_anchor_visualizer.visualize(
                    self.robot_anchor_pos_w, self.robot_anchor_quat_w
                )
                self.goal_anchor_visualizer.visualize(
                    self.anchor_pos_w, self.anchor_quat_w
                )

                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].visualize(
                        self.robot_body_pos_w[:, i],
                        self.robot_body_quat_w[:, i],
                    )
                    self.goal_body_visualizers[i].visualize(
                        self.body_pos_relative_w[:, i],
                        self.body_quat_relative_w[:, i],
                    )

        @configclass
        class CommandsCfg:
            motion = MotionCommandCfg(
                class_type=MotionCommand,
                asset_name="robot",
                resampling_time_range=(1.0e9, 1.0e9),
                debug_vis=True,
                pose_range={
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.01, 0.01),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
                },
                joint_position_range=(-0.1, 0.1),
            )

        ARMATURE_5020 = 0.003609725
        ARMATURE_7520_14 = 0.010177520
        ARMATURE_7520_22 = 0.025101925
        ARMATURE_4010 = 0.00425

        NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
        DAMPING_RATIO = 2.0

        STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
        STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
        STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
        STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

        DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
        DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
        DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
        DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

        @configclass
        class MotionTrackingSceneCfg(InteractiveSceneCfg):
            """Configuration for the Motion Tracking scene."""

            num_envs = 4
            env_spacing = 4.0
            replicate_physics = True

            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
            )

            robot: ArticulationCfg = ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UrdfFileCfg(
                    fix_base=False,
                    replace_cylinders_with_capsules=True,
                    asset_path=urdf_path,
                    usd_dir=usd_dir,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        retain_accelerations=False,
                        linear_damping=0.0,
                        angular_damping=0.0,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        max_depenetration_velocity=1.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=4,
                    ),
                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                            stiffness=0, damping=0
                        )
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.76),
                    joint_pos={
                        ".*_hip_pitch_joint": -0.312,
                        ".*_knee_joint": 0.669,
                        ".*_ankle_pitch_joint": -0.363,
                        ".*_elbow_joint": 0.6,
                        "left_shoulder_roll_joint": 0.2,
                        "left_shoulder_pitch_joint": 0.2,
                        "right_shoulder_roll_joint": -0.2,
                        "right_shoulder_pitch_joint": 0.2,
                    },
                    joint_vel={".*": 0.0},
                ),
                soft_joint_pos_limit_factor=0.9,
                actuators={
                    "legs": ImplicitActuatorCfg(
                        joint_names_expr=[
                            ".*_hip_yaw_joint",
                            ".*_hip_roll_joint",
                            ".*_hip_pitch_joint",
                            ".*_knee_joint",
                        ],
                        effort_limit_sim={
                            ".*_hip_yaw_joint": 88.0,
                            ".*_hip_roll_joint": 139.0,
                            ".*_hip_pitch_joint": 88.0,
                            ".*_knee_joint": 139.0,
                        },
                        velocity_limit_sim={
                            ".*_hip_yaw_joint": 32.0,
                            ".*_hip_roll_joint": 20.0,
                            ".*_hip_pitch_joint": 32.0,
                            ".*_knee_joint": 20.0,
                        },
                        stiffness={
                            ".*_hip_pitch_joint": STIFFNESS_7520_14,
                            ".*_hip_roll_joint": STIFFNESS_7520_22,
                            ".*_hip_yaw_joint": STIFFNESS_7520_14,
                            ".*_knee_joint": STIFFNESS_7520_22,
                        },
                        damping={
                            ".*_hip_pitch_joint": DAMPING_7520_14,
                            ".*_hip_roll_joint": DAMPING_7520_22,
                            ".*_hip_yaw_joint": DAMPING_7520_14,
                            ".*_knee_joint": DAMPING_7520_22,
                        },
                        armature={
                            ".*_hip_pitch_joint": ARMATURE_7520_14,
                            ".*_hip_roll_joint": ARMATURE_7520_22,
                            ".*_hip_yaw_joint": ARMATURE_7520_14,
                            ".*_knee_joint": ARMATURE_7520_22,
                        },
                    ),
                    "feet": ImplicitActuatorCfg(
                        effort_limit_sim=50.0,
                        velocity_limit_sim=37.0,
                        joint_names_expr=[
                            ".*_ankle_pitch_joint",
                            ".*_ankle_roll_joint",
                        ],
                        stiffness=2.0 * STIFFNESS_5020,
                        damping=2.0 * DAMPING_5020,
                        armature=2.0 * ARMATURE_5020,
                    ),
                    "waist": ImplicitActuatorCfg(
                        effort_limit_sim=50,
                        velocity_limit_sim=37.0,
                        joint_names_expr=[
                            "waist_roll_joint",
                            "waist_pitch_joint",
                        ],
                        stiffness=2.0 * STIFFNESS_5020,
                        damping=2.0 * DAMPING_5020,
                        armature=2.0 * ARMATURE_5020,
                    ),
                    "waist_yaw": ImplicitActuatorCfg(
                        effort_limit_sim=88,
                        velocity_limit_sim=32.0,
                        joint_names_expr=["waist_yaw_joint"],
                        stiffness=STIFFNESS_7520_14,
                        damping=DAMPING_7520_14,
                        armature=ARMATURE_7520_14,
                    ),
                    "arms": ImplicitActuatorCfg(
                        joint_names_expr=[
                            ".*_shoulder_pitch_joint",
                            ".*_shoulder_roll_joint",
                            ".*_shoulder_yaw_joint",
                            ".*_elbow_joint",
                        ],
                        effort_limit_sim={
                            ".*_shoulder_pitch_joint": 25.0,
                            ".*_shoulder_roll_joint": 25.0,
                            ".*_shoulder_yaw_joint": 25.0,
                            ".*_elbow_joint": 25.0,
                        },
                        velocity_limit_sim={
                            ".*_shoulder_pitch_joint": 37.0,
                            ".*_shoulder_roll_joint": 37.0,
                            ".*_shoulder_yaw_joint": 37.0,
                            ".*_elbow_joint": 37.0,
                        },
                        stiffness={
                            ".*_shoulder_pitch_joint": STIFFNESS_5020,
                            ".*_shoulder_roll_joint": STIFFNESS_5020,
                            ".*_shoulder_yaw_joint": STIFFNESS_5020,
                            ".*_elbow_joint": STIFFNESS_5020,
                        },
                        damping={
                            ".*_shoulder_pitch_joint": DAMPING_5020,
                            ".*_shoulder_roll_joint": DAMPING_5020,
                            ".*_shoulder_yaw_joint": DAMPING_5020,
                            ".*_elbow_joint": DAMPING_5020,
                        },
                        armature={
                            ".*_shoulder_pitch_joint": ARMATURE_5020,
                            ".*_shoulder_roll_joint": ARMATURE_5020,
                            ".*_shoulder_yaw_joint": ARMATURE_5020,
                            ".*_elbow_joint": ARMATURE_5020,
                        },
                    ),
                },
            )

            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(
                    color=(0.75, 0.75, 0.75), intensity=3000.0
                ),
            )
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    color=(0.13, 0.13, 0.13), intensity=1000.0
                ),
            )

            contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=3,
                track_air_time=True,
                force_threshold=10.0,
                debug_vis=False,
            )

        @configclass
        class ObservationsCfg:
            """Observation specifications for the MDP."""

            @configclass
            class ActorCfg(ObsGroup):
                """Observations for policy group."""

                base_lin_vel = ObsTerm(
                    func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
                )
                base_ang_vel = ObsTerm(
                    func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
                )
                joint_pos = ObsTerm(
                    func=mdp.joint_pos_rel,
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                )
                joint_vel = ObsTerm(
                    func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
                )
                actions = ObsTerm(func=mdp.last_action)

                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True

            @configclass
            class PrivilegedCfg(ObsGroup):
                base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
                base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                joint_vel = ObsTerm(func=mdp.joint_vel_rel)
                actions = ObsTerm(func=mdp.last_action)

            # observation groups
            actor_obs: ActorCfg = ActorCfg()
            critic_obs: PrivilegedCfg = PrivilegedCfg()

        @configclass
        class ActionsCfg:
            """Actions configuration."""

            joint_efforts: JointEffortActionCfg = JointEffortActionCfg(
                asset_name="robot",
                joint_names=[".*"],
            )

        def constant_reward(env):
            """Simple constant reward for testing."""
            return torch.ones(env.num_envs, device=env.device)

        def time_limit_termination(env):
            """Simple time limit termination."""
            return env.episode_length_buf >= env.max_episode_length

        @configclass
        class RewardsCfg:
            """Rewards configuration."""

            alive: RewardTermCfg = RewardTermCfg(
                func=constant_reward, weight=1.0
            )

        @configclass
        class TerminationsCfg:
            time_out: TerminationTermCfg = TerminationTermCfg(
                func=time_limit_termination
            )

        @configclass
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            """Configuration for the Motion Tracking environment."""

            # Environment settings
            episode_length_s: float = 3600.0
            decimation: int = 4
            action_scale: float = 0.25

            # Manager configurations (will be populated after creation)
            observations = ObservationsCfg()
            actions = ActionsCfg()
            rewards = RewardsCfg()
            terminations = TerminationsCfg()

            # Simulation settings
            sim: SimulationCfg = SimulationCfg(
                dt=1.0 / 50.0,  # 50 Hz
                render_interval=4,  # decimation
                physx=PhysxCfg(bounce_threshold_velocity=0.2),
                device="cuda" if self._device is None else str(self._device),
            )

            # Scene configuration
            scene: MotionTrackingSceneCfg = MotionTrackingSceneCfg()

            # Viewer configuration for proper camera positioning
            viewer: ViewerCfg = ViewerCfg(origin_type="world")

        isaac_lab_cfg = MotionTrackingEnvCfg()

        if self._device is not None:
            isaac_lab_cfg.device = str(self._device)
            logger.info(
                f"Setting IsaacLab environment device to: {self._device}"
            )
        else:
            isaac_lab_cfg.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(
                f"Device not specified, defaulting to: {isaac_lab_cfg.device}"
            )

        # Assign the populated configurations
        # isaac_lab_cfg.observations = ObservationsCfg()
        # isaac_lab_cfg.actions = ActionsCfg()
        # isaac_lab_cfg.rewards = RewardsCfg()
        # isaac_lab_cfg.terminations = TerminationsCfg()

        self._env = ManagerBasedRLEnv(isaac_lab_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized successfully !")

        return isaac_lab_cfg

    def _init_motion_tracking_components(self):
        """Initialize motion tracking specific components."""
        self.num_extend_bodies = len(
            getattr(self.config, "robot", {})
            .get("motion", {})
            .get("extend_config", [])
        )
        self.n_fut_frames = getattr(self.config, "obs", {}).get(
            "n_fut_frames", 1
        )

        self._init_motion_lib()
        self._init_tracking_config()
        self._init_curriculum_settings()
        self._init_serializers()

        self.log_dict_holomotion = {}
        self.log_dict_nonreduced_holomotion = {}

        if (
            hasattr(self.config, "disable_ref_viz")
            and self.config.disable_ref_viz
        ):
            self.debug_viz = False

    def step(self, actor_state: dict):
        """Execute one environment step."""
        return self._env.step(actor_state)

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments by ID."""
        return self._env.reset(seed=self._seed, env_ids=env_ids)

    def reset_all(self):
        """Reset all environments."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._env.reset(seed=self._seed, env_ids=env_ids)

    def set_is_evaluating(self):
        """Set the environment to evaluation mode."""
        logger.info("Setting environment to evaluation mode")
        self.is_evaluating = True

    def test_simulation_loop(
        self,
        num_steps: int = 1000,
    ):
        obs, extras = self._env.reset()
        logger.info(
            f"Environment reset complete. Observation keys: {list(obs.keys())}"
        )
        action_dim = self._env.action_manager.total_action_dim
        logger.info(f"Action dimension: {action_dim}")
        for step in range(num_steps):
            actions = (
                torch.rand(self._env.num_envs, action_dim, device=self.device)
                * 2
                - 1
            )  # [-1, 1]
            obs, rewards, terminated, truncated, extras = self._env.step(
                actions
            )

            if self.render_mode == "human":
                import time

                time.sleep(0.01)  # 10ms delay for smoother visualization

    def _init_serializers(self):
        """Initialize observation serializers."""
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

    def _init_curriculum_settings(self):
        """Initialize curriculum learning and termination settings."""
        # Entropy coefficient
        self.entropy_coef = getattr(self.config, "init_entropy_coef", 0.01)

        # Termination thresholds
        if hasattr(self.config, "termination"):
            term_config = self.config.termination
            term_curriculum = getattr(
                self.config, "termination_curriculum", {}
            )
            term_scales = getattr(self.config, "termination_scales", {})

            # Motion far termination
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

            # Joint far termination
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

            # Patience steps for termination
            self.terminate_when_joint_far_patience_steps = term_config.get(
                "terminate_when_joint_far_patience_steps", 1
            )
            self.terminate_when_motion_far_patience_steps = term_config.get(
                "terminate_when_motion_far_patience_steps", 1
            )
        else:
            # Default values
            self.terminate_when_motion_far_threshold = 0.5
            self.terminate_when_joint_far_threshold = 1.0
            self.terminate_when_joint_far_patience_steps = 1
            self.terminate_when_motion_far_patience_steps = 1

        # Waist DOF curriculum
        if hasattr(self.config, "rewards"):
            self.use_waist_dof_curriculum = self.config.rewards.get(
                "use_waist_dof_curriculum", False
            )
            if self.use_waist_dof_curriculum:
                self.waist_dof_penalty_scale = torch.tensor(
                    1.0, device=self.device
                )
                logger.info(
                    f"Waist DOF curriculum enabled with scale: {self.waist_dof_penalty_scale}"
                )
        else:
            self.use_waist_dof_curriculum = False

    def _init_motion_lib(self):
        """Initialize motion library."""
        if hasattr(self.config, "robot") and hasattr(
            self.config.robot, "motion"
        ):
            self._motion_lib = LmdbMotionLib(
                self.config.robot.motion,
                self.device,
                process_id=getattr(self.config, "process_id", 0),
                num_processes=getattr(self.config, "num_processes", 1),
            )
            self.motion_start_idx = 0
            self.num_motions = self._motion_lib._num_unique_motions
            logger.info(
                f"Motion library initialized with {self.num_motions} motions"
            )
        else:
            logger.warning(
                "No motion configuration found, motion library not initialized"
            )

    def _init_tracking_config(self):
        """Initialize tracking configuration for body and joint indices."""
        logger.info("Motion tracking configuration initialized (placeholder)")
