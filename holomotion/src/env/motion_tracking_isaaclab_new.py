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

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

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
        self.device = device
        self.log_dir = log_dir
        self.headless = headless
        self.init_done = False
        self.debug_viz = True
        self.is_evaluating = False
        self.render_mode = render_mode

        # Initialize Isaac Sim and import packages
        self._setup_isaac_sim()

        # Create IsaacLab environment configuration
        isaac_lab_cfg = self._init_isaaclab_env(holomotion_env_config=config)

        # Initialize motion tracking specific components
        # self._init_motion_tracking_components()

        # self.init_done = True
        # logger.info("Motion Tracking Environment initialized successfully")

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
        app_launcher_flags = {"headless": self.headless}
        self._sim_app_launcher = AppLauncher(app_launcher_flags)
        self._sim_app = self._sim_app_launcher.app

        logger.info("Isaac Sim initialized successfully")

    def _init_isaaclab_env(self, holomotion_env_config):
        """Create IsaacLab configuration from HoloMotion config."""
        # Import necessary classes directly
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
        )
        from isaaclab.envs.mdp.actions import JointEffortActionCfg
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

        urdf_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/assets/robots/unitree/G1/23dof/official_g1_23dof_rev_1_0.urdf"
        usd_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/assets/robots/unitree/G1/23dof"

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        logger.info(f"Using URDF path: {urdf_path}")
        logger.info(f"Using USD directory: {usd_dir}")

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
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            """Configuration for the Motion Tracking environment."""

            # Environment settings
            episode_length_s: float = 3600.0
            decimation: int = 4
            action_scale: float = 0.25

            # Manager configurations (will be populated after creation)
            observations = None
            actions = None
            rewards = None
            terminations = None

            # Simulation settings
            sim: SimulationCfg = SimulationCfg(
                dt=1.0 / 50.0,  # 50 Hz
                render_interval=4,  # decimation
                physx=PhysxCfg(bounce_threshold_velocity=0.2),
            )

            # Scene configuration
            scene: MotionTrackingSceneCfg = MotionTrackingSceneCfg()

            # Viewer configuration for proper camera positioning
            viewer: ViewerCfg = ViewerCfg(origin_type="world")

        # Create the IsaacLab config
        isaac_lab_cfg = MotionTrackingEnvCfg()

        def constant_reward(env):
            """Simple constant reward for testing."""
            return torch.ones(env.num_envs, device=env.device)

        def time_limit_termination(env):
            """Simple time limit termination."""
            return env.episode_length_buf >= env.max_episode_length

        @configclass
        class ObservationsCfg:
            """Observation specifications for the MDP."""

            @configclass
            class PolicyCfg(ObsGroup):
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
            policy: PolicyCfg = PolicyCfg()
            critic: PrivilegedCfg = PrivilegedCfg()

        @configclass
        class ActionsCfg:
            """Actions configuration."""

            joint_efforts: JointEffortActionCfg = JointEffortActionCfg(
                asset_name="robot",
                joint_names=[".*"],
            )

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

        # Assign the populated configurations
        isaac_lab_cfg.observations = ObservationsCfg()
        isaac_lab_cfg.actions = ActionsCfg()
        isaac_lab_cfg.rewards = RewardsCfg()
        isaac_lab_cfg.terminations = TerminationsCfg()

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
        return self._env.reset_idx(env_ids)

    def reset_all(self):
        """Reset all environments."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.reset_idx(env_ids)

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
