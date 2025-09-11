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
from typing import Dict, List, Optional, Literal, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.transform import Rotation as sRot

# IsaacLab imports
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.app import AppLauncher
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventManager,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    RayCaster,
    RayCasterCfg,
    patterns,
)
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.timer import Timer

# HoloMotion imports
from holomotion.src.modules.agent_modules import ObsSeqSerializer
from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib
from holomotion.src.utils.isaac_utils.rotations import (
    calc_heading_quat_inv,
    get_euler_xyz,
    my_quat_rotate,
    quat_inverse,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_matrix,
    wrap_to_pi,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from holomotion.src.utils.torch_utils import (
    quat_error_magnitude,
    subtract_frame_transforms,
    quat_to_tan_norm,
)


@configclass
class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Motion Tracking environment."""

    # Environment settings
    episode_length_s: float = 3600.0
    decimation: int = 4
    action_scale: float = 0.25

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 50.0,  # 50 Hz
        render_interval=decimation,
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # Robot configuration (will be set from config)
    robot: ArticulationCfg = None

    # Terrain configuration
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Height scanner
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


class MotionTrackingEnv(ManagerBasedRLEnv):
    """IsaacLab-based Motion Tracking Environment.

    This environment integrates motion tracking capabilities with IsaacLab's
    manager-based architecture, supporting curriculum learning, domain randomization,
    and various termination conditions.
    """

    def __init__(
        self,
        cfg: MotionTrackingEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        """Initialize the Motion Tracking Environment.

        Args:
            cfg: Configuration for the environment
            render_mode: Render mode for the environment
            **kwargs: Additional keyword arguments
        """
        # Store original config for compatibility
        self.config = cfg
        self.init_done = False
        self.debug_viz = True
        self.is_evaluating = False

        # Initialize motion tracking specific attributes
        self.num_extend_bodies = len(
            getattr(cfg, "robot", {})
            .get("motion", {})
            .get("extend_config", [])
        )
        self.n_fut_frames = getattr(cfg, "obs", {}).get("n_fut_frames", 1)

        # Initialize IsaacLab environment
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize motion tracking components
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()
        self._init_curriculum_settings()
        self._init_serializers()

        self.init_done = True

        # Logging dictionaries
        self.log_dict_holomotion = {}
        self.log_dict_nonreduced_holomotion = {}

        if hasattr(cfg, "disable_ref_viz") and cfg.disable_ref_viz:
            self.debug_viz = False

        logger.info("Motion Tracking Environment initialized successfully")

    def _setup_scene(self):
        """Set up the scene with robot, terrain, and sensors."""
        # Setup robot articulation from config
        self._setup_robot_articulation()

        # Setup terrain based on configuration
        self._setup_terrain()

        # Setup sensors
        self._setup_sensors()

        # Add robot to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if hasattr(self, "_height_scanner"):
            self.scene.sensors["height_scanner"] = self._height_scanner

        # Clone environments and setup lighting
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(
            global_prim_paths=[self.cfg.terrain.prim_path]
        )

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_cfg.func("/World/DomeLight", light_cfg, translation=(1, 0, 10))

    def _setup_robot_articulation(self):
        """Setup robot articulation from configuration."""
        # This will be configured based on the robot config passed in
        if self.cfg.robot is None:
            # Create default robot configuration
            self.cfg.robot = self._create_default_robot_config()

        self._robot = Articulation(self.cfg.robot)

    def _setup_terrain(self):
        """Setup terrain based on configuration."""
        # Configure terrain based on mesh type
        if hasattr(self.cfg, "terrain_config"):
            terrain_config = self.cfg.terrain_config
            if terrain_config.mesh_type in ["heightfield", "trimesh"]:
                self._setup_complex_terrain(terrain_config)
            else:
                self._setup_plane_terrain()
        else:
            self._setup_plane_terrain()

    def _setup_sensors(self):
        """Setup contact sensor and height scanner."""
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        if hasattr(self.cfg, "height_scanner"):
            self._height_scanner = RayCaster(self.cfg.height_scanner)

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

    def _init_motion_lib(self):
        """Initialize motion library."""
        cache_device = self.device

        if hasattr(self.config, "robot") and hasattr(
            self.config.robot, "motion"
        ):
            self._motion_lib = LmdbMotionLib(
                self.config.robot.motion,
                cache_device,
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
        if not hasattr(self.config, "robot") or not hasattr(
            self.config.robot, "motion"
        ):
            logger.warning("No robot motion configuration found")
            return

        # Get body and joint names from the articulation
        self._body_list = self._robot.body_names
        self._dof_names = self._robot.joint_names

        motion_config = self.config.robot.motion

        # Motion tracking links
        if "motion_tracking_link" in motion_config:
            self.motion_tracking_id = [
                self._body_list.index(link)
                for link in motion_config.motion_tracking_link
            ]
        else:
            self.motion_tracking_id = []

        # Hand links
        if "hand_link" in motion_config:
            self.hand_indices = [
                self._body_list.index(link) for link in motion_config.hand_link
            ]
        else:
            self.hand_indices = []

        # Lower and upper body links
        if "lower_body_link" in motion_config:
            self.lower_body_id = [
                self._body_list.index(link)
                for link in motion_config.lower_body_link
            ]
        else:
            self.lower_body_id = []

        if "upper_body_link" in motion_config:
            self.upper_body_id = [
                self._body_list.index(link)
                for link in motion_config.upper_body_link
            ]
        else:
            self.upper_body_id = []

        # Joint IDs for lower and upper body
        if "lower_body_joint_ids" in motion_config:
            self.lower_body_joint_ids = [
                self._dof_names.index(link)
                for link in motion_config.lower_body_joint_ids
            ]
        else:
            self.lower_body_joint_ids = []

        if "upper_body_joint_ids" in motion_config:
            self.upper_body_joint_ids = [
                self._dof_names.index(link)
                for link in motion_config.upper_body_joint_ids
            ]
        else:
            self.upper_body_joint_ids = []

        # Key bodies configuration
        if self.config.robot.get("has_key_bodies", False):
            self.key_body_names = self.config.robot.key_bodies
            self.key_body_indices = [
                self._body_list.index(link) for link in self.key_body_names
            ]

            # Initialize filtered indices
            if (
                hasattr(self, "_need_to_precalculate_indices")
                and self._need_to_precalculate_indices
            ):
                self._precalculate_filtered_indices()
            else:
                self.filtered_upper_joint_ids = self.upper_body_joint_ids
                self.filtered_lower_joint_ids = self.lower_body_joint_ids
                self.filtered_key_body_indices_pos = self.key_body_indices
                self.filtered_key_body_indices_rot = self.key_body_indices
                self.filtered_key_body_indices_combined = self.key_body_indices
        else:
            self.key_body_indices = []
            self.filtered_key_body_indices_pos = []
            self.filtered_key_body_indices_rot = []
            self.filtered_key_body_indices_combined = []

            if not hasattr(self, "filtered_upper_joint_ids"):
                self.filtered_upper_joint_ids = self.upper_body_joint_ids
                self.filtered_lower_joint_ids = self.lower_body_joint_ids

        # End effector body links
        if "ee_bodylink_ids" in motion_config:
            self.ee_bodylink_ids = [
                self._body_list.index(link)
                for link in motion_config.ee_bodylink_ids
            ]
        else:
            self.ee_bodylink_ids = []

        # Ankle tracking configuration
        self.no_ankle_tracking = self.config.robot.get(
            "no_ankle_tracking", False
        )
        if self.no_ankle_tracking:
            self._setup_ankle_tracking_config()
        else:
            self._need_to_precalculate_indices = False

        # Anchor body
        anchor_body_name = self.config.robot.get("anchor_body", "pelvis")
        try:
            self.anchor_body_id = self._body_list.index(anchor_body_name)
        except ValueError:
            logger.warning(
                f"Anchor body {anchor_body_name} not found, using index 0"
            )
            self.anchor_body_id = 0

        # Undesired contact bodies
        self.undesired_contact_bodies = self.config.robot.get(
            "undesired_contact_bodies", []
        )
        if self.undesired_contact_bodies:
            self.undesired_contact_body_ids = [
                self.find_rigid_body_indice(link)
                for link in self.undesired_contact_bodies
            ]
        self.undesired_contact_threshold = self.config.robot.get(
            "undesired_contact_threshold", 1.0
        )

        # Motion resampling configuration
        if (
            hasattr(self.config, "resample_motion_when_training")
            and self.config.resample_motion_when_training
        ):
            self.resample_time_interval = np.ceil(
                self.config.resample_time_interval_s / self.dt
            )

        # Evaluation threshold
        if hasattr(self.config, "termination_scales"):
            self.eval_motion_far_threshold = (
                self.config.termination_scales.get(
                    "eval_motion_far_threshold", 0.25
                )
            )
        else:
            self.eval_motion_far_threshold = 0.25

    def _setup_ankle_tracking_config(self):
        """Setup ankle tracking ignore configuration."""
        robot_config = self.config.robot

        logger.warning(
            f"No ankle tracking enabled! Position/velocity tracking disabled for: "
            f"{robot_config.get('ignore_pos_ankle_bodies', [])}, "
            f"Rotation/angular velocity tracking disabled for: "
            f"{robot_config.get('ignore_rot_ankle_bodies', [])}, "
            f"Joint tracking disabled for: {robot_config.get('ignore_ankle_joints', [])}"
        )

        self.ignore_pos_ankle_bodies = [
            self._body_list.index(link)
            for link in robot_config.get("ignore_pos_ankle_bodies", [])
            if link in self._body_list
        ]
        self.ignore_rot_ankle_bodies = [
            self._body_list.index(link)
            for link in robot_config.get("ignore_rot_ankle_bodies", [])
            if link in self._body_list
        ]
        self.ignore_ankle_joints = [
            self._dof_names.index(link)
            for link in robot_config.get("ignore_ankle_joints", [])
            if link in self._dof_names
        ]

        self._need_to_precalculate_indices = True
        self._precalculate_joint_filtering()

    def find_rigid_body_indice(self, body_name: str) -> int:
        """Find the index of a rigid body by name."""
        try:
            return self._body_list.index(body_name)
        except ValueError:
            logger.warning(f"Body {body_name} not found in body list")
            return -1

    def _precalculate_joint_filtering(self):
        """Precalculate filtered joint indices to avoid torch.compile issues."""
        ignore_ankle_joints_set = set(self.ignore_ankle_joints)
        self.filtered_upper_joint_ids = [
            idx
            for idx in self.upper_body_joint_ids
            if idx not in ignore_ankle_joints_set
        ]
        self.filtered_lower_joint_ids = [
            idx
            for idx in self.lower_body_joint_ids
            if idx not in ignore_ankle_joints_set
        ]

    def _precalculate_filtered_indices(self):
        """Precalculate filtered key body indices to avoid torch.compile issues in reward functions."""
        # Filter key body indices (this assumes key_body_indices is already set)
        ignore_pos_ankle_bodies_set = set(self.ignore_pos_ankle_bodies)
        ignore_rot_ankle_bodies_set = set(self.ignore_rot_ankle_bodies)
        ignore_combined_ankle_bodies_set = ignore_pos_ankle_bodies_set.union(
            ignore_rot_ankle_bodies_set
        )

        self.filtered_key_body_indices_pos = [
            idx
            for idx in self.key_body_indices
            if idx not in ignore_pos_ankle_bodies_set
        ]
        self.filtered_key_body_indices_rot = [
            idx
            for idx in self.key_body_indices
            if idx not in ignore_rot_ankle_bodies_set
        ]
        self.filtered_key_body_indices_combined = [
            idx
            for idx in self.key_body_indices
            if idx not in ignore_combined_ankle_bodies_set
        ]

    def _create_default_robot_config(self) -> ArticulationCfg:
        """Create a default robot configuration."""
        # This should be implemented based on your robot specifications
        # For now, return a basic configuration
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="path/to/robot.usd",  # Update with actual path
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
        )

    def _init_motion_extend(self):
        """Initialize motion extension configuration."""
        if (
            hasattr(self.config, "robot")
            and hasattr(self.config.robot, "motion")
            and "extend_config" in self.config.robot.motion
        ):
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                try:
                    parent_idx = self._body_list.index(
                        extend_config["parent_name"]
                    )
                    extend_parent_ids.append(parent_idx)
                    extend_pos.append(extend_config["pos"])
                    extend_rot.append(extend_config["rot"])
                    self._body_list.append(extend_config["joint_name"])
                except ValueError:
                    logger.warning(
                        f"Parent body {extend_config['parent_name']} not found"
                    )
                    continue

            if extend_parent_ids:  # Only create if we have valid extensions
                self.extend_body_parent_ids = torch.tensor(
                    extend_parent_ids, device=self.device, dtype=torch.long
                )
                self.extend_body_pos_in_parent = (
                    torch.tensor(extend_pos)
                    .repeat(self.num_envs, 1, 1)
                    .to(self.device)
                )
                self.extend_body_rot_in_parent_wxyz = (
                    torch.tensor(extend_rot)
                    .repeat(self.num_envs, 1, 1)
                    .to(self.device)
                )
                self.extend_body_rot_in_parent_xyzw = (
                    self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
                )

                # Initialize extended body tracking tensors
                total_bodies = self.num_bodies + self.num_extend_bodies
                self.marker_coords = torch.zeros(
                    self.num_envs,
                    total_bodies,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )

                self.ref_body_pos_extend = torch.zeros(
                    self.num_envs,
                    total_bodies,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                self.dif_global_body_pos = torch.zeros(
                    self.num_envs,
                    total_bodies,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )

                logger.info(
                    f"Motion extension initialized with {len(extend_parent_ids)} extended bodies"
                )
        else:
            # Initialize empty extension configuration
            self.extend_body_parent_ids = torch.tensor(
                [], device=self.device, dtype=torch.long
            )
            self.num_extend_bodies = 0

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self.scene.num_envs

    @property
    def num_bodies(self) -> int:
        """Number of rigid bodies in the robot."""
        return self._robot.num_bodies

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._robot.num_joints

    @property
    def device(self) -> torch.device:
        """Device for tensor operations."""
        return self.sim.device

    @property
    def dt(self) -> float:
        """Environment step time."""
        return self.cfg.decimation * self.sim.cfg.dt

    # IsaacLab-specific properties for compatibility
    @property
    def dof_pos(self) -> torch.Tensor:
        """Joint positions."""
        return self._robot.data.joint_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        """Joint velocities."""
        return self._robot.data.joint_vel

    @property
    def base_quat(self) -> torch.Tensor:
        """Base quaternion in xyzw format (HoloMotion standard)."""
        # IsaacSim/IsaacLab uses wxyz, convert to xyzw for HoloMotion compatibility
        wxyz_quat = self._robot.data.root_quat_w
        return wxyz_quat[:, [1, 2, 3, 0]]  # Convert wxyz to xyzw

    @property
    def contact_forces(self) -> torch.Tensor:
        """Contact forces."""
        return self._contact_sensor.data.net_forces_w

    @property
    def robot_root_states(self) -> torch.Tensor:
        """Robot root states in [pos, quat_xyzw, lin_vel, ang_vel] format (HoloMotion standard)."""
        root_pos = self._robot.data.root_pos_w
        root_quat_wxyz = self._robot.data.root_quat_w
        root_quat_xyzw = root_quat_wxyz[
            :, [1, 2, 3, 0]
        ]  # Convert wxyz to xyzw for HoloMotion
        root_lin_vel = self._robot.data.root_lin_vel_w
        root_ang_vel = self._robot.data.root_ang_vel_w

        return torch.cat(
            [root_pos, root_quat_xyzw, root_lin_vel, root_ang_vel], dim=-1
        )

    @property
    def _rigid_body_pos(self) -> torch.Tensor:
        """Rigid body positions."""
        return self._robot.data.body_pos_w

    @property
    def _rigid_body_rot(self) -> torch.Tensor:
        """Rigid body rotations in xyzw format (HoloMotion standard)."""
        wxyz_rot = self._robot.data.body_quat_w
        return wxyz_rot[
            :, :, [1, 2, 3, 0]
        ]  # Convert wxyz to xyzw for HoloMotion

    @property
    def _rigid_body_vel(self) -> torch.Tensor:
        """Rigid body linear velocities."""
        return self._robot.data.body_lin_vel_w

    @property
    def _rigid_body_ang_vel(self) -> torch.Tensor:
        """Rigid body angular velocities."""
        return self._robot.data.body_ang_vel_w

    def _setup_plane_terrain(self):
        """Setup simple plane terrain."""
        self.cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )

    def _setup_complex_terrain(self, terrain_config):
        """Setup complex terrain (heightfield or trimesh)."""
        sub_terrains = {}
        terrain_types = terrain_config.get("terrain_types", ["flat"])
        terrain_proportions = terrain_config.get("terrain_proportions", [1.0])

        for terrain_type, proportion in zip(
            terrain_types, terrain_proportions
        ):
            if proportion > 0:
                if terrain_type == "flat":
                    sub_terrains[terrain_type] = (
                        terrain_gen.MeshPlaneTerrainCfg(proportion=proportion)
                    )
                elif terrain_type == "rough":
                    sub_terrains[terrain_type] = (
                        terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion,
                            noise_range=(0.02, 0.10),
                            noise_step=0.02,
                            border_width=0.25,
                        )
                    )
                elif terrain_type == "low_obst":
                    sub_terrains[terrain_type] = (
                        terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion,
                            grid_width=0.45,
                            grid_height_range=(0.05, 0.2),
                            platform_width=2.0,
                        )
                    )

        terrain_generator_cfg = TerrainGeneratorCfg(
            curriculum=terrain_config.get("curriculum", False),
            size=(
                terrain_config.get("terrain_length", 8.0),
                terrain_config.get("terrain_width", 8.0),
            ),
            border_width=terrain_config.get("border_size", 0.25),
            num_rows=terrain_config.get("num_rows", 10),
            num_cols=terrain_config.get("num_cols", 10),
            horizontal_scale=terrain_config.get("horizontal_scale", 0.1),
            vertical_scale=terrain_config.get("vertical_scale", 0.005),
            slope_threshold=terrain_config.get("slope_threshold", 0.75),
            use_cache=False,
            sub_terrains=sub_terrains,
        )

        self.cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator_cfg,
            max_init_terrain_level=9,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=terrain_config.get("static_friction", 1.0),
                dynamic_friction=terrain_config.get("dynamic_friction", 1.0),
            ),
            debug_vis=False,
        )

    def _post_init_callback(self):
        """Post-initialization callback for additional setup."""
        # Initialize motion tracking buffers after scene setup
        self._init_motion_tracking_buffers()

        # Setup domain randomization events if configured
        self._setup_domain_randomization()

        # Log environment configuration
        if hasattr(self.config, "main_process") and self.config.main_process:
            logger.info(
                f"Motion tracking environment initialized with {self.num_envs} environments"
            )
            if hasattr(self.config, "termination"):
                logger.info(f"Termination strategy: {self.config.termination}")

    def _init_motion_tracking_buffers(self):
        """Initialize motion tracking specific buffers."""
        # Motion sampling buffers
        if hasattr(self, "_motion_lib"):
            self.motion_ids = self._motion_lib.resample_new_motions(
                self.num_envs, eval=self.is_evaluating
            ).to(self.device)

            self.motion_global_start_frame_ids = (
                self._motion_lib.cache.sample_cached_global_start_frames(
                    torch.arange(self.num_envs),
                    n_fut_frames=self.n_fut_frames,
                    eval=self.is_evaluating,
                ).to(self.device)
            )

            self.motion_global_end_frame_ids = (
                self._motion_lib.cache.cached_motion_global_end_frames.to(
                    self.device
                )
            )

        # Heading and reset buffers
        self.cur_heading_inv_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.reset_buf_motion_far = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.robot_initial_heading_inv_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.ref_initial_heading_inv_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Patience counters for termination
        self.joint_far_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.motion_far_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        # History buffers (initialize as None, will be created when needed)
        self.pos_history_buffer = None
        self.rot_history_buffer = None
        self.ref_pos_history_buffer = None
        self.current_accel = None
        self.ref_body_accel = None
        self.current_ang_accel = None

        # Logging dictionaries
        self.log_dict = {}
        self.log_dict_nonreduced = {}

        logger.info("Motion tracking buffers initialized")

    def _setup_domain_randomization(self):
        """Setup domain randomization events."""
        if not hasattr(self.config, "domain_rand"):
            return

        domain_rand_config = self.config.domain_rand
        events_cfg = {}

        # Base mass randomization
        if domain_rand_config.get("randomize_base_mass", False):
            events_cfg["randomize_base_mass"] = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": tuple(
                        domain_rand_config.get("added_mass_range", [0.0, 5.0])
                    ),
                    "operation": "add",
                },
            )

        # Link mass randomization
        if domain_rand_config.get("randomize_link_mass", False):
            events_cfg["randomize_link_mass"] = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(
                        domain_rand_config.get("link_mass_range", [0.8, 1.2])
                    ),
                    "operation": "scale",
                },
            )

        # Friction randomization
        if domain_rand_config.get("randomize_friction", False):
            events_cfg["randomize_friction"] = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": tuple(
                        domain_rand_config.get("friction_range", [0.5, 1.5])
                    ),
                    "operation": "scale",
                },
            )

        if events_cfg:
            # Create event manager with the configured events
            from isaaclab.utils import configclass

            @configclass
            class EventsCfg:
                pass

            events_cfg_obj = EventsCfg()
            for name, event in events_cfg.items():
                setattr(events_cfg_obj, name, event)

            self.event_manager = EventManager(events_cfg_obj, self)
            logger.info(
                f"Domain randomization setup with events: {list(events_cfg.keys())}"
            )

    def _update_terminate_when_motion_far_curriculum(self):
        """Update motion far termination threshold based on curriculum."""
        if not (
            hasattr(self.config, "termination")
            and hasattr(self.config, "termination_curriculum")
        ):
            return

        term_config = self.config.termination
        term_curriculum = self.config.termination_curriculum

        if not (
            term_config.get("terminate_when_motion_far", False)
            and term_curriculum.get(
                "terminate_when_motion_far_curriculum", False
            )
        ):
            return

        if not hasattr(self, "average_episode_length"):
            return

        if self.average_episode_length < term_curriculum.get(
            "terminate_when_motion_far_curriculum_level_down_threshold", 50
        ):
            self.terminate_when_motion_far_threshold *= (
                1
                + term_curriculum.get(
                    "terminate_when_motion_far_curriculum_degree", 0.01
                )
            )
        elif self.average_episode_length > term_curriculum.get(
            "terminate_when_motion_far_curriculum_level_up_threshold", 200
        ):
            self.terminate_when_motion_far_threshold *= (
                1
                - term_curriculum.get(
                    "terminate_when_motion_far_curriculum_degree", 0.01
                )
            )

        self.terminate_when_motion_far_threshold = np.clip(
            self.terminate_when_motion_far_threshold,
            term_curriculum.get(
                "terminate_when_motion_far_threshold_min", 0.1
            ),
            term_curriculum.get(
                "terminate_when_motion_far_threshold_max", 2.0
            ),
        )

    def _update_terminate_when_joint_far_curriculum(self):
        """Update joint far termination threshold based on curriculum."""
        if not (
            hasattr(self.config, "termination")
            and hasattr(self.config, "termination_curriculum")
        ):
            return

        term_config = self.config.termination
        term_curriculum = self.config.termination_curriculum

        if not (
            term_config.get("terminate_when_joint_far", False)
            and term_curriculum.get(
                "terminate_when_joint_far_curriculum", False
            )
        ):
            return

        if not hasattr(self, "average_episode_length"):
            return

        if self.average_episode_length < term_curriculum.get(
            "terminate_when_joint_far_curriculum_level_down_threshold", 50
        ):
            self.terminate_when_joint_far_threshold *= 1 + term_curriculum.get(
                "terminate_when_joint_far_curriculum_degree", 0.01
            )
        elif self.average_episode_length > term_curriculum.get(
            "terminate_when_joint_far_curriculum_level_up_threshold", 200
        ):
            self.terminate_when_joint_far_threshold *= 1 - term_curriculum.get(
                "terminate_when_joint_far_curriculum_degree", 0.01
            )

        self.terminate_when_joint_far_threshold = np.clip(
            self.terminate_when_joint_far_threshold,
            term_curriculum.get("terminate_when_joint_far_threshold_min", 0.1),
            term_curriculum.get("terminate_when_joint_far_threshold_max", 2.0),
        )

    def _update_entropy_curriculum(self):
        """Update entropy coefficient based on curriculum."""
        if not hasattr(self.config, "entropy_curriculum"):
            return

        entropy_config = self.config.entropy_curriculum
        if not entropy_config.get("enable_entropy_curriculum", False):
            return

        if not hasattr(self, "average_episode_length"):
            return

        if self.average_episode_length > entropy_config.get(
            "entropy_curriculum_threshold", 100
        ):
            self.entropy_coef *= 1 - entropy_config.get(
                "entropy_curriculum_degree", 0.01
            )

        self.entropy_coef = np.clip(
            self.entropy_coef,
            entropy_config.get("entropy_curriculum_threshold_min", 0.001),
            entropy_config.get("entropy_curriculum_threshold_max", 0.1),
        )

    def _update_tasks_callback(self):
        """Update task-specific state each step."""
        # Motion resampling during training
        if (
            hasattr(self.config, "resample_motion_when_training")
            and self.config.resample_motion_when_training
            and not self.is_evaluating
        ):
            if hasattr(self, "common_step_counter") and hasattr(
                self, "resample_time_interval"
            ):
                if self.common_step_counter % self.resample_time_interval == 0:
                    logger.info(
                        f"Resampling motion at step {self.common_step_counter}"
                    )
                    self.resample_motion()

    def _check_termination(self):
        """Check termination conditions specific to motion tracking."""
        # This will be implemented when termination logic is added
        # For now, just initialize basic termination checking
        terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Motion far termination (placeholder)
        if hasattr(self.config, "termination") and self.config.termination.get(
            "terminate_when_motion_far", False
        ):
            # Will be implemented with motion tracking logic
            pass

        # Joint far termination (placeholder)
        if hasattr(self.config, "termination") and self.config.termination.get(
            "terminate_when_joint_far", False
        ):
            # Will be implemented with joint tracking logic
            pass

        return terminated

    def _update_timeout_buf(self):
        """Update timeout buffer for episode termination."""
        if not hasattr(self, "episode_length_buf"):
            return

        max_episode_length = int(self.cfg.episode_length_s / self.dt)
        timeout_mask = self.episode_length_buf >= max_episode_length

        # Check motion end termination if configured
        if (
            hasattr(self.config, "termination")
            and self.config.termination.get("terminate_when_motion_end", False)
            and hasattr(self, "motion_global_end_frame_ids")
        ):
            current_global_frame_ids = (
                self.episode_length_buf + self.motion_global_start_frame_ids
            )
            motion_end_mask = current_global_frame_ids >= (
                self.motion_global_end_frame_ids - 1 - self.n_fut_frames
            )
            timeout_mask |= motion_end_mask

        return timeout_mask

    # PPO Compatibility Methods
    def step(self, actor_state: dict) -> tuple:
        """Execute one environment step with PPO compatibility.

        Args:
            actor_state: Dictionary containing 'actions' key

        Returns:
            Tuple of (obs_dict, rewards, dones, extras)
        """
        actions = actor_state["actions"]

        self._process_actions(actions)

        self._physics_step()

        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _physics_step(self):
        """Execute physics simulation steps."""
        # Render if needed
        if self.sim.has_gui():
            self.sim.render()

        # Execute control decimation steps
        decimation = getattr(self.cfg, "decimation", 4)
        for _ in range(decimation):
            if (
                hasattr(self.config, "direct_state_control")
                and self.config.direct_state_control
            ):
                self._apply_direct_state_control()
            else:
                self._apply_torque_control()
            self.sim.step(render=False)

    def _post_physics_step(self):
        """Post-physics step processing."""
        # Update scene data
        self.scene.update(self.dt)

        # Update episode length
        if not hasattr(self, "episode_length_buf"):
            self.episode_length_buf = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        self.episode_length_buf += 1

        # Update counters if they exist
        if hasattr(self, "common_step_counter"):
            self.common_step_counter += 1

        # Pre-compute observations callback
        self._pre_compute_observations_callback()

        # Update tasks
        self._update_tasks_callback()

        # Check termination
        self._check_termination()

        # Compute rewards
        self._compute_reward()

        # Reset environments if needed
        env_ids = (
            self.reset_buf.nonzero(as_tuple=False).flatten()
            if hasattr(self, "reset_buf")
            else torch.tensor([], device=self.device)
        )
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Compute observations
        self._compute_observations()

        # Post-compute observations callback
        self._post_compute_observations_callback()

        # Clip observations if configured
        if hasattr(self.config, "normalization") and hasattr(
            self, "obs_buf_dict"
        ):
            clip_obs = self.config.normalization.get("clip_observations", 10.0)
            for obs_key, obs_val in self.obs_buf_dict.items():
                self.obs_buf_dict[obs_key] = torch.clip(
                    obs_val, -clip_obs, clip_obs
                )

        # Setup extras for logging
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["to_log"] = getattr(self, "log_dict", {})
        self.extras["log_nonreduced"] = getattr(
            self, "log_dict_nonreduced", {}
        )

        # Debug visualization
        if self.debug_viz and hasattr(self, "viewer") and self.viewer:
            self._draw_debug_vis()

    def _pre_compute_observations_callback(self):
        """Prepare shared variables for observations calculation."""
        # This will be implemented when observation logic is added
        # For now, just ensure basic state is available
        pass

    def _compute_observations(self):
        """Compute observations for the environment."""
        # Initialize empty observation dict for now
        if not hasattr(self, "obs_buf_dict"):
            self.obs_buf_dict = {}

        # This will be implemented separately with observation functions

    def _post_compute_observations_callback(self):
        """Post-compute observations callback."""
        # Update action history if needed
        if hasattr(self, "last_actions") and hasattr(self, "actions"):
            self.last_actions = self.actions.clone()

    # Initialize essential buffers for PPO compatibility
    def _init_essential_buffers(self):
        """Initialize essential buffers for PPO compatibility."""
        # Reward buffer
        self.rew_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # Reset buffer
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Episode length buffer
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Actions buffer
        if hasattr(self.config, "robot") and hasattr(
            self.config.robot, "actions_dim"
        ):
            action_dim = self.config.robot.actions_dim
        else:
            action_dim = self.num_dof

        self.actions = torch.zeros(
            self.num_envs, action_dim, dtype=torch.float, device=self.device
        )
        self.last_actions = torch.zeros(
            self.num_envs, action_dim, dtype=torch.float, device=self.device
        )

        # Observation buffer dict
        self.obs_buf_dict = {}

        # Extras dict for logging
        self.extras = {}

        # Common step counter
        self.common_step_counter = 0

        logger.info("Essential buffers initialized for PPO compatibility")

    def _process_actions(self, actions: torch.Tensor):
        """Process and store actions."""
        # Clip actions if configured
        if hasattr(self.config, "robot") and hasattr(
            self.config.robot, "control"
        ):
            clip_limit = self.config.robot.control.get(
                "action_clip_value", 1.0
            )
            actions = torch.clamp(actions, -clip_limit, clip_limit)

        self.actions = actions

    def _apply_actions(self):
        """Apply actions to the robot."""
        if (
            hasattr(self.config, "direct_state_control")
            and self.config.direct_state_control
        ):
            self._apply_direct_state_control()
        else:
            self._apply_torque_control()

    def _apply_torque_control(self):
        """Apply torque control to robot joints."""
        # Convert actions to torques based on control type
        control_type = getattr(self.config.robot.control, "control_type", "P")

        if control_type == "P":
            # PD control
            kp = torch.tensor(
                [100.0] * self.num_dof, device=self.device
            )  # Default gains
            kd = torch.tensor([5.0] * self.num_dof, device=self.device)

            target_pos = self.actions + self._get_default_joint_pos()
            pos_error = target_pos - self.dof_pos
            vel_error = -self.dof_vel

            torques = kp * pos_error + kd * vel_error
        elif control_type == "T":
            # Direct torque control
            torques = self.actions
        else:
            logger.warning(
                f"Unknown control type: {control_type}, using direct torque"
            )
            torques = self.actions

        # Apply torques
        self._robot.set_joint_effort_target(torques)

    def _apply_direct_state_control(self):
        """Apply direct state control (kinematic control)."""
        if not hasattr(self, "_motion_lib"):
            return

        # Get current motion state
        motion_frame_ids = (
            self.episode_length_buf + self.motion_global_start_frame_ids
        )

        motion_res = self._motion_lib.cache.get_motion_state(
            motion_frame_ids, global_offset=self.scene.env_origins
        )

        # Set DOF states
        dof_pos = motion_res["dof_pos"].to(self.device)
        dof_vel = (
            motion_res["dof_vel"].to(self.device) * 0
        )  # Zero velocity for kinematic

        self._robot.write_joint_state_to_sim(dof_pos, dof_vel)

        # Set root states
        root_pos = motion_res["root_pos"].to(self.device)
        root_rot_xyzw = motion_res["root_rot"].to(
            self.device
        )  # Motion lib uses xyzw
        root_rot_wxyz = xyzw_to_wxyz(
            root_rot_xyzw
        )  # Convert to wxyz for IsaacSim
        root_vel = motion_res["root_vel"].to(self.device)
        root_ang_vel = motion_res["root_ang_vel"].to(self.device)

        self._robot.write_root_pose_to_sim(
            torch.cat([root_pos, root_rot_wxyz], dim=-1)
        )
        self._robot.write_root_velocity_to_sim(
            torch.cat([root_vel, root_ang_vel], dim=-1)
        )

    def _get_default_joint_pos(self) -> torch.Tensor:
        """Get default joint positions."""
        if hasattr(self.config, "robot") and hasattr(
            self.config.robot, "init_state"
        ):
            default_angles = self.config.robot.init_state.get(
                "default_joint_angles", {}
            )
            default_pos = torch.zeros(self.num_dof, device=self.device)

            for i, joint_name in enumerate(self._dof_names):
                if joint_name in default_angles:
                    default_pos[i] = default_angles[joint_name]

            return default_pos.repeat(self.num_envs, 1)
        else:
            return torch.zeros(self.num_envs, self.num_dof, device=self.device)

    def _compute_reward(self):
        """Compute reward for the current step."""
        # This will be implemented when reward functions are added
        # For now, just initialize the reward buffer
        self.rew_buf[:] = 0.0

        # Initialize reward components buffer for vectorized value function
        if not hasattr(self, "reward_components_buf"):
            max_num_values = getattr(self.config, "max_num_values", 30)
            self.reward_components_buf = torch.zeros(
                self.num_envs,
                max_num_values,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self.reward_components_buf[:] = 0.0

        # Reward computation will be implemented separately
        # For now, just return to maintain compatibility
        return

    def get_reward_components(self):
        """Get individual reward components for vectorized value function.

        Returns:
            torch.Tensor: Shape [num_envs, max_num_values] containing individual reward values
        """
        if hasattr(self, "reward_components_buf"):
            return self.reward_components_buf
        else:
            # Fallback if not initialized
            max_num_values = getattr(self.config, "max_num_values", 20)
            return torch.zeros(
                self.num_envs,
                max_num_values,
                dtype=torch.float,
                device=self.device,
            )

    def set_is_evaluating(self):
        """Set the environment to evaluation mode."""
        logger.info("Setting environment to evaluation mode")
        self.is_evaluating = True

    def reset_all(self):
        """Reset all environments."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.reset_idx(env_ids)

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments by ID."""
        if len(env_ids) == 0:
            return

        # Reset motion tracking state
        self._reset_motion_tracking_state(env_ids)

        # Reset robot states
        self._reset_robot_states(env_ids)

        # Update scene
        self.scene.write_data_to_sim()

        # Return observations
        return self._get_observations()

    def _reset_motion_tracking_state(self, env_ids: torch.Tensor):
        """Reset motion tracking specific state."""
        if hasattr(self, "_motion_lib"):
            # Resample motion frames
            self._resample_motion_frame_ids(env_ids)

        # Reset patience counters
        if hasattr(self, "joint_far_counter"):
            self.joint_far_counter[env_ids] = 0
        if hasattr(self, "motion_far_counter"):
            self.motion_far_counter[env_ids] = 0

        # Reset history buffers if they exist
        if (
            hasattr(self, "pos_history_buffer")
            and self.pos_history_buffer is not None
        ):
            # Initialize with current positions
            current_pos = self._robot.data.body_pos_w[env_ids]
            self.pos_history_buffer[env_ids] = current_pos.unsqueeze(1).repeat(
                1, 3, 1, 1
            )

    def _reset_robot_states(self, env_ids: torch.Tensor):
        """Reset robot root and joint states."""
        if hasattr(self, "_motion_lib"):
            self._reset_root_states_from_motion(env_ids)
            self._reset_dofs_from_motion(env_ids)
        else:
            # Default reset to initial configuration
            self._reset_to_default_state(env_ids)

    def _reset_root_states_from_motion(self, env_ids: torch.Tensor):
        """Reset root states from motion library."""
        if not hasattr(self, "_motion_lib"):
            return

        # Get motion state for reset
        motion_frame_ids = (
            self.episode_length_buf[env_ids]
            + self.motion_global_start_frame_ids[env_ids]
        )

        motion_res = self._motion_lib.cache.get_motion_state(
            motion_frame_ids, global_offset=self.scene.env_origins[env_ids]
        )

        # Apply noise if configured
        noise_scales = getattr(self.config, "init_noise_scale", {})
        root_pos_noise = noise_scales.get("root_pos", 0.0) * getattr(
            self.config, "noise_to_initial_level", 1.0
        )
        root_rot_noise = noise_scales.get("root_rot", 0.0) * getattr(
            self.config, "noise_to_initial_level", 1.0
        )
        root_vel_noise = noise_scales.get("root_vel", 0.0) * getattr(
            self.config, "noise_to_initial_level", 1.0
        )
        root_ang_vel_noise = noise_scales.get("root_ang_vel", 0.0) * getattr(
            self.config, "noise_to_initial_level", 1.0
        )

        # Extract motion data
        root_pos = motion_res["root_pos"][:, 0].to(self.device)
        root_rot = motion_res["root_rot"][:, 0].to(self.device)
        root_vel = motion_res["root_vel"][:, 0].to(self.device)
        root_ang_vel = motion_res["root_ang_vel"][:, 0].to(self.device)

        # Add noise
        root_pos += torch.randn_like(root_pos) * root_pos_noise
        root_vel += torch.randn_like(root_vel) * root_vel_noise
        root_ang_vel += torch.randn_like(root_ang_vel) * root_ang_vel_noise

        # Apply small random rotation if configured
        if root_rot_noise > 0:
            random_quat = self.small_random_quaternions(
                len(env_ids), root_rot_noise
            )
            root_rot = quat_mul(random_quat, root_rot, w_last=True)

        # CRITICAL: Motion library returns xyzw, IsaacSim expects wxyz
        # Convert from HoloMotion xyzw to IsaacSim wxyz format
        root_rot_wxyz = xyzw_to_wxyz(root_rot)

        # Set robot states
        self._robot.write_root_pose_to_sim(
            torch.cat([root_pos, root_rot_wxyz], dim=-1), env_ids
        )
        self._robot.write_root_velocity_to_sim(
            torch.cat([root_vel, root_ang_vel], dim=-1), env_ids
        )

    def _reset_dofs_from_motion(self, env_ids: torch.Tensor):
        """Reset DOF states from motion library."""
        if not hasattr(self, "_motion_lib"):
            return

        # Get motion state for reset
        motion_frame_ids = (
            self.episode_length_buf[env_ids]
            + self.motion_global_start_frame_ids[env_ids]
        )

        motion_res = self._motion_lib.cache.get_motion_state(
            motion_frame_ids, global_offset=self.scene.env_origins[env_ids]
        )

        # Apply noise if configured
        noise_scales = getattr(self.config, "init_noise_scale", {})
        dof_pos_noise = noise_scales.get("dof_pos", 0.0) * getattr(
            self.config, "noise_to_initial_level", 1.0
        )
        dof_vel_noise = noise_scales.get("dof_vel", 0.0) * getattr(
            self.config, "noise_to_initial_level", 1.0
        )

        # Extract DOF data
        dof_pos = motion_res["dof_pos"][:, 0].to(self.device)
        dof_vel = motion_res["dof_vel"][:, 0].to(self.device)

        # Add noise
        dof_pos += torch.randn_like(dof_pos) * dof_pos_noise
        dof_vel += torch.randn_like(dof_vel) * dof_vel_noise

        # Set DOF states
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    def _reset_to_default_state(self, env_ids: torch.Tensor):
        """Reset to default robot configuration."""
        # Reset to initial state defined in robot config
        init_state = self.cfg.robot.init_state

        # Set root pose
        root_pos = torch.tensor(init_state.pos, device=self.device).repeat(
            len(env_ids), 1
        )
        root_rot = torch.tensor([0, 0, 0, 1], device=self.device).repeat(
            len(env_ids), 1
        )  # wxyz

        self._robot.write_root_pose_to_sim(
            torch.cat([root_pos, root_rot], dim=-1), env_ids
        )

        # Set joint positions to default
        default_joint_pos = torch.zeros(
            len(env_ids), self.num_dof, device=self.device
        )
        default_joint_vel = torch.zeros(
            len(env_ids), self.num_dof, device=self.device
        )

        self._robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, env_ids=env_ids
        )

    def small_random_quaternions(
        self, n: int, max_angle: float
    ) -> torch.Tensor:
        """Generate small random quaternions for noise in xyzw format (HoloMotion standard)."""
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        angles = max_angle * torch.rand((n, 1), device=self.device)

        sin_half_angle = torch.sin(angles / 2)
        cos_half_angle = torch.cos(angles / 2)

        # Return quaternion in xyzw format for HoloMotion compatibility
        q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)
        return q

    def _resample_motion_frame_ids(self, env_ids: torch.Tensor):
        """Resample motion frame IDs for given environments."""
        if len(env_ids) == 0 or not hasattr(self, "_motion_lib"):
            return

        if self.is_evaluating and not getattr(
            self.config, "enforce_randomize_motion_start_eval", False
        ):
            # Keep current frame IDs during evaluation
            return
        else:
            self.motion_global_start_frame_ids[env_ids] = (
                self._motion_lib.cache.sample_cached_global_start_frames(
                    env_ids,
                    n_fut_frames=self.n_fut_frames,
                    eval=self.is_evaluating,
                ).to(self.device)
            )

    def resample_motion(self):
        """Resample motion for all environments."""
        if not hasattr(self, "_motion_lib"):
            logger.warning(
                "Motion library not initialized, cannot resample motion"
            )
            return

        self.motion_ids = (
            self._motion_lib.resample_new_motions(
                self.num_envs, eval=self.is_evaluating
            )
            .to(self.device)
            .clone()
        )

        self.motion_global_start_frame_ids = (
            self._motion_lib.cache.sample_cached_global_start_frames(
                torch.arange(self.num_envs),
                n_fut_frames=self.n_fut_frames,
                eval=self.is_evaluating,
            ).to(self.device)
        )

        self.motion_global_end_frame_ids = (
            self._motion_lib.cache.cached_motion_global_end_frames.to(
                self.device
            )
        )

        self.reset_all()

    def resample_motion_eval(self):
        """Resample motion for evaluation."""
        if not hasattr(self, "_motion_lib"):
            return True

        is_last_eval_batch = self._motion_lib.load_next_eval_batch(
            self.num_envs
        )

        self.motion_global_start_frame_ids = (
            self._motion_lib.cache.sample_cached_global_start_frames(
                torch.arange(self.num_envs),
                n_fut_frames=self.n_fut_frames,
                eval=self.is_evaluating,
            ).to(self.device)
        )

        self.motion_global_end_frame_ids = (
            self._motion_lib.cache.cached_motion_global_end_frames.to(
                self.device
            )
        )

        self.reset_all()
        return is_last_eval_batch

    # Motion tracking logging placeholder methods
    def _log_motion_tracking_info(self):
        """Log motion tracking information."""
        # Initialize logging dictionaries
        if not hasattr(self, "log_dict"):
            self.log_dict = {}
        if not hasattr(self, "log_dict_nonreduced"):
            self.log_dict_nonreduced = {}

        # This method will be fully implemented when motion tracking
        # reward and observation logic is added separately

    # Essential methods for maintaining interface compatibility
    def reset_envs_idx(self, env_ids: torch.Tensor):
        """Reset environments by indices (compatibility method)."""
        return self.reset_idx(env_ids)

    def __post_init__(self):
        """Post-initialization setup after scene is created."""
        # Initialize essential buffers
        self._init_essential_buffers()

        # Call the post-init callback
        self._post_init_callback()

        logger.info(
            "Motion tracking environment post-initialization completed"
        )

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # Initialize empty observation dict for now
        if not hasattr(self, "obs_buf_dict"):
            self.obs_buf_dict = {}

        # This will be implemented separately with observation functions
        return self.obs_buf_dict

    def _get_dones(self) -> tuple:
        """Get termination and truncation flags."""
        # Check various termination conditions
        terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        truncated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Check timeout
        if hasattr(self, "episode_length_buf"):
            max_episode_length = int(self.cfg.episode_length_s / self.dt)
            truncated |= self.episode_length_buf >= max_episode_length

        # Motion tracking termination conditions will be added later
        return terminated, truncated

    def _get_extras(self) -> dict:
        """Get extra information."""
        extras = {}

        if hasattr(self, "log_dict"):
            extras["to_log"] = self.log_dict

        if hasattr(self, "log_dict_nonreduced"):
            extras["log_nonreduced"] = self.log_dict_nonreduced

        return extras

    def render(self):
        """Render the environment."""
        if self.sim.has_gui():
            self.sim.render()

    def close(self):
        """Close the environment."""
        if hasattr(self, "sim"):
            self.sim.stop()

    def _draw_debug_vis(self):
        """Draw debug visualization."""
        if not self.debug_viz:
            return

        # Draw marker coordinates if available
        if hasattr(self, "marker_coords"):
            for env_id in range(
                min(self.num_envs, 4)
            ):  # Limit to first 4 envs
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]):
                    color = (0.3, 0.3, 0.3)  # Default color
                    if hasattr(self.config, "robot") and hasattr(
                        self.config.robot, "motion"
                    ):
                        if hasattr(self.config.robot.motion, "visualization"):
                            viz_config = self.config.robot.motion.visualization
                            if viz_config.get("customize_color", False):
                                colors = viz_config.get(
                                    "marker_joint_colors", [(0.3, 0.3, 0.3)]
                                )
                                color = colors[pos_id % len(colors)]

                    # Draw sphere at marker position (placeholder for IsaacLab visualization)
                    pass
