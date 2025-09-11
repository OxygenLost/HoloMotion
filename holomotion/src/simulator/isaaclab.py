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
import sys
import weakref
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import torch
from loguru import logger
from pyvirtualdisplay import Display
from rich.progress import Progress

from isaaclab.app import AppLauncher


from holomotion.src.simulator.base_simulator import BaseSimulator


class IsaacLab(BaseSimulator):
    """IsaacLab/IsaacSim simulation backend.

    Provides a framework for simulation setup, environment creation, and
    control over robotic assets and simulation properties.
    """

    def __init__(self, config, device):
        """Initializes the base simulator with configuration settings.

        Args:
            config (dict): Configuration dictionary for the simulation.
            device (str): Device type for simulation ('cpu' or 'cuda').

        """
        self.config = config
        self.sim_device = device
        self.headless = True

        self._rigid_body_pos: torch.Tensor
        self._rigid_body_rot: torch.Tensor
        self._rigid_body_vel: torch.Tensor
        self._rigid_body_ang_vel: torch.Tensor

    def set_headless(self, headless):
        """Sets the headless mode for the simulator.

        Args:
            headless (bool): If True, runs the simulation without
                graphical display.
        """
        self.headless = headless

    def setup_isaacsim_sim(self):
        if self.headless:
            os.environ["ISAAC_SIM_HEADLESS"] = "1"
            os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
            display = Display(visible=0, size=(1024, 768))
            display.start()
            logger.info(
                "Headless setup for IsaacLab/IsaacSim with virtual display !"
            )
        app_launcher_flags = {
            "headless": self.headless,
        }
        self._sim_app_launcher = AppLauncher(app_launcher_flags)
        self._sim_app = self._sim_app_launcher.app

        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        import isaaclab.terrains as terrain_gen
        import isaaclab.utils.math as math_utils
        from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
        from isaaclab.envs import (
            DirectRLEnv,
            DirectRLEnvCfg,
            ManagerBasedEnv,
            ViewerCfg,
        )
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
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
        from isaaclab.utils.timer import Timer
        from isaaclab_assets import H1_CFG

        if SimulationContext.instance() is None:
            self.sim = SimulationContext(
                sim_utils.SimulationCfg(
                    device=str(self.sim_device),
                    dt=1.0 / self.config.sim.fps,
                )
            )
        else:
            raise RuntimeError("SimulationContext already exists !")

        camera_pose = [[2.0, 0.0, 2.5], [-0.5, 0.0, 0.5]]
        self.sim.set_camera_view(*camera_pose)
        logger.info(f"Camera pose set to: {camera_pose} !")

        self.sim_dt = 1.0 / self.simulator_config.sim.fps

        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self.simulator_config.scene.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing,
            replicate_physics=self.simulator_config.scene.replicate_physics,
        )
        self.scene = InteractiveScene(scene_config)
        self._setup_scene()

        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(
                self, viewer_config
            )
        else:
            self.viewport_camera_controller = None

        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros(
            (self.simulator_config.scene.num_envs, 3),
            dtype=torch.float,
            device="cpu",
        )

        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(
                        self.domain_rand_config["link_mass_range"]
                    ),
                    "operation": "scale",
                },
            )

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": tuple(
                        self.domain_rand_config["friction_range"]
                    ),
                    "operation": "scale",
                },
            )

        if self.domain_rand_config.get("randomize_base_com", False):
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            "torso_link",
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor(
                            [
                                self.domain_rand_config["base_com_range"]["x"][
                                    0
                                ],
                                self.domain_rand_config["base_com_range"]["y"][
                                    0
                                ],
                                self.domain_rand_config["base_com_range"]["z"][
                                    0
                                ],
                            ]
                        ),
                        torch.tensor(
                            [
                                self.domain_rand_config["base_com_range"]["x"][
                                    1
                                ],
                                self.domain_rand_config["base_com_range"]["y"][
                                    1
                                ],
                                self.domain_rand_config["base_com_range"]["z"][
                                    1
                                ],
                            ]
                        ),
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)

        self._sim_step_counter = 0
        logger.info("Completed setting up the environment...")

    def _setup_scene(self):
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file

        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(asset_root, asset_path),
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
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        )

        default_joint_angles = copy.deepcopy(
            self.robot_config.init_state.default_joint_angles
        )
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle
                for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping

        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(
                        f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}"
                    )

        actuators = {
            dof_names_list[i]: IdealPDActuatorCfg(
                joint_names_expr=[dof_names_list[i]],
                effort_limit=dof_effort_limit_list[i],
                velocity_limit=dof_vel_limit_list[i],
                stiffness=0,
                damping=0,
                armature=dof_armature_list[i],
                friction=dof_joint_friction_list[i],
            )
            for i in range(len(dof_names_list))
        }

        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
        )

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3,
            update_period=0.005,
            track_air_time=True,
        )

        # Add a height scanner to the torso to detect the height of the terrain mesh
        height_scanner_config = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # Apply a grid pattern that is smaller than the resolution to only return one height value.
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.1, size=[0.05, 0.05]
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        if (self.terrain_config.mesh_type == "heightfield") or (
            self.terrain_config.mesh_type == "trimesh"
        ):
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(
                terrain_types, terrain_proportions
            ):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = (
                            terrain_gen.MeshPlaneTerrainCfg(
                                proportion=proportion
                            )
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

            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(
                    self.terrain_config.terrain_length,
                    self.terrain_config.terrain_width,
                ),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs

        else:
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing

        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(height_scanner_config)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(
            global_prim_paths=[terrain_config.prim_path]
        )

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func(
            "/World/DomeLight", light_config1, translation=(1, 0, 10)
        )

    def setup(self):
        """Initializes the simulator parameters and environment.

        This method should be implemented by subclasses to set specific
        simulator configurations.
        """
        raise NotImplementedError(
            "The 'setup' method must be implemented in subclasses."
        )

    def setup_terrain(self, mesh_type):
        """Configures the terrain based on specified mesh type.

        Args:
            mesh_type (str): Type of terrain mesh ('plane', 'heightfield',
                'trimesh').

        """
        raise NotImplementedError(
            "The 'setup_terrain' method must be implemented in subclasses."
        )

    def load_assets(self, robot_config):
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)

        self.dof_ids, self.dof_names = self._robot.find_joints(
            dof_names_list, preserve_order=True
        )
        self.body_ids, self.body_names = self._robot.find_bodies(
            self.robot_config.body_names, preserve_order=True
        )

        self._body_list = self.body_names.copy()

        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        assert self.dof_ids != list(range(self.num_dof)), (
            "The order of the joint_names in the robot_config does not match "
            "the order of the joint_ids in IsaacSim."
        )
        assert self.num_dof == len(self.robot_config.dof_names), (
            "Number of DOFs must be equal to number of actions"
        )
        assert self.num_bodies == len(self.robot_config.body_names), (
            "Number of bodies must be equal to number of body names"
        )
        assert self.dof_names == self.robot_config.dof_names, (
            "DOF names must match the config"
        )
        assert self.body_names == self.robot_config.body_names, (
            "Body names must match the config"
        )

    def create_envs(self, num_envs, env_origins, base_init_state, env_config):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        return self.scene, self._robot

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof,
            2,
            dtype=torch.float,
            device=self.sim_device,
            requires_grad=False,
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof,
            2,
            dtype=torch.float,
            device=self.sim_device,
            requires_grad=False,
        )
        self.dof_vel_limits = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.sim_device,
            requires_grad=False,
        )
        self.torque_limits = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.sim_device,
            requires_grad=False,
        )
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = (
                self.robot_config.dof_pos_lower_limit_list[i]
            )
            self.hard_dof_pos_limits[i, 1] = (
                self.robot_config.dof_pos_upper_limit_list[i]
            )
            self.dof_pos_limits[i, 0] = (
                self.robot_config.dof_pos_lower_limit_list[i]
            )
            self.dof_pos_limits[i, 1] = (
                self.robot_config.dof_pos_upper_limit_list[i]
            )
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m
                - 0.5
                * r
                * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m
                + 0.5
                * r
                * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            )
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(
                f"Body {body_name} not found in the contact sensor."
            )
            return None
        elif len(indices) == 1:
            return indices[0]
        else:  # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices

    def prepare_sim(self):
        self.refresh_sim_tensors()

    @property
    def dof_state(self):
        return torch.cat(
            [self.dof_pos[..., None], self.dof_vel[..., None]],
            dim=-1,
        )

    def refresh_sim_tensors(self):
        ############################################################################################
        # TODO: currently, we only consider the robot root state, ignore other objects's root states
        ############################################################################################
        self.all_root_states = self._robot.data.root_state_w  # (num_envs, 13)

        self.robot_root_states = self.all_root_states  # (num_envs, 13)
        self.base_quat = self.robot_root_states[
            :, [4, 5, 6, 3]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency

        self.dof_pos = self._robot.data.joint_pos[
            :, self.dof_ids
        ]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        self.contact_forces = (
            self.contact_sensor.data.net_forces_w
        )  # (num_envs, num_bodies, 3)

        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][
            :, :, [1, 2, 3, 0]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_lin_vel_w[
            :, self.body_ids, :
        ]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[
            :, self.body_ids, :
        ]

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        self._robot.write_root_pose_to_sim(
            root_states[set_env_ids, :7], set_env_ids
        )
        self._robot.write_root_velocity_to_sim(
            root_states[set_env_ids, 7:], set_env_ids
        )

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = (
            dof_states[set_env_ids, :, 0],
            dof_states[set_env_ids, :, 1],
        )
        self._robot.write_joint_state_to_sim(
            dof_pos, dof_vel, self.dof_ids, set_env_ids
        )

    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if (
            self._sim_step_counter % self.simulator_config.sim.render_interval
            == 0
            and is_rendering
        ):
            self.sim.render()
        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller

    def render(self, sync_frame_time=True):
        pass

    def draw_sphere(self, pos, radius, color, env_id):
        # draw a big sphere
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)

    def draw_line(self, start_point, end_point, color, env_id):
        # import ipdb; ipdb.set_trace()
        start_point_list = [
            (start_point.x.item(), start_point.y.item(), start_point.z.item())
        ]
        end_point_list = [
            (end_point.x.item(), end_point.y.item(), end_point.z.item())
        ]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(
            start_point_list, end_point_list, color_list, sizes
        )


class ViewportCameraController:
    """This class handles controlling the camera associated with a viewport in the simulator.

    It can be used to set the viewpoint camera to track different origin types:

    - **world**: the center of the world (static)
    - **env**: the center of an environment (static)
    - **asset_root**: the root of an asset in the scene (e.g. tracking a robot moving in the scene)

    On creation, the camera is set to track the origin type specified in the configuration.

    For the :attr:`asset_root` origin type, the camera is updated at each rendering step to track the asset's
    root position. For this, it registers a callback to the post update event stream from the simulation app.
    """

    def __init__(self, env, cfg: ViewerCfg):
        """Initialize the ViewportCameraController.

        Args:
            env: The environment.
            cfg: The configuration for the viewport camera controller.

        Raises:
            ValueError: If origin type is configured to be "env" but :attr:`cfg.env_index` is out of bounds.
            ValueError: If origin type is configured to be "asset_root" but :attr:`cfg.asset_name` is unset.

        """
        # store inputs
        self._env = env
        self._cfg = copy.deepcopy(cfg)
        # cast viewer eye and look-at to numpy arrays
        self.default_cam_eye = np.array(self._cfg.eye)
        self.default_cam_lookat = np.array(self._cfg.lookat)

        # set the camera origins
        if self.cfg.origin_type == "env":
            # check that the env_index is within bounds
            self.set_view_env_index(self.cfg.env_index)
            # set the camera origin to the center of the environment
            self.update_view_to_env()
        elif self.cfg.origin_type == "asset_root":
            # note: we do not yet update camera for tracking an asset origin, as the asset may not yet be
            # in the scene when this is called. Instead, we subscribe to the post update event to update the camera
            # at each rendering step.
            if self.cfg.asset_name is None:
                raise ValueError(
                    f"No asset name provided for viewer with origin type: '{self.cfg.origin_type}'."
                )
        else:
            # set the camera origin to the center of the world
            self.update_view_to_world()

        # subscribe to post update event so that camera view can be updated at each rendering step
        app_interface = omni.kit.app.get_app_interface()
        app_event_stream = app_interface.get_post_update_event_stream()
        self._viewport_camera_update_handle = (
            app_event_stream.create_subscription_to_pop(
                lambda event,
                obj=weakref.proxy(self): obj._update_tracking_callback(event)
            )
        )

    def __del__(self):
        """Unsubscribe from the callback."""
        # use hasattr to handle case where __init__ has not completed before __del__ is called
        if (
            hasattr(self, "_viewport_camera_update_handle")
            and self._viewport_camera_update_handle is not None
        ):
            self._viewport_camera_update_handle.unsubscribe()
            self._viewport_camera_update_handle = None

    """
    Properties
    """

    @property
    def cfg(self) -> ViewerCfg:
        """The configuration for the viewer."""
        return self._cfg

    """
    Public Functions
    """

    def set_view_env_index(self, env_index: int):
        """Sets the environment index for the camera view.

        Args:
            env_index: The index of the environment to set the camera view to.

        Raises:
            ValueError: If the environment index is out of bounds. It should be between 0 and num_envs - 1.
        """
        # check that the env_index is within bounds
        if env_index < 0 or env_index >= self._env.config.scene.num_envs:
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {self._env.config.scene.num_envs - 1} for the current environment."
            )
        # update the environment index
        self.cfg.env_index = env_index
        # update the camera view if the origin is set to env type (since, the camera view is static)
        # note: for assets, the camera view is updated at each rendering step
        if self.cfg.origin_type == "env":
            self.update_view_to_env()

    def update_view_to_world(self):
        """Updates the viewer's origin to the origin of the world which is (0, 0, 0)."""
        # set origin type to world
        self.cfg.origin_type = "world"
        # update the camera origins
        self.viewer_origin = torch.zeros(3)
        # update the camera view
        self.update_view_location()

    def update_view_to_env(self):
        """Updates the viewer's origin to the origin of the selected environment."""
        # set origin type to world
        self.cfg.origin_type = "env"
        # update the camera origins
        self.viewer_origin = self._env.scene.env_origins[self.cfg.env_index]
        # update the camera view
        self.update_view_location()

    def update_view_to_asset_root(self, asset_name: str):
        """Updates the viewer's origin based upon the root of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.

        Raises:
            ValueError: If the asset is not in the scene.
        """
        # check if the asset is in the scene
        if self.cfg.asset_name != asset_name:
            asset_entities = [
                *self._env.scene.rigid_objects.keys(),
                *self._env.scene.articulations.keys(),
            ]
            if asset_name not in asset_entities:
                raise ValueError(
                    f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}."
                )
        # update the asset name
        self.cfg.asset_name = asset_name
        # set origin type to asset_root
        self.cfg.origin_type = "asset_root"
        # update the camera origins
        self.viewer_origin = self._env.scene[
            self.cfg.asset_name
        ].data.root_pos_w[self.cfg.env_index]
        # update the camera view
        self.update_view_location()

    def update_view_location(
        self,
        eye: Sequence[float] | None = None,
        lookat: Sequence[float] | None = None,
    ):
        """Updates the camera view pose based on the current viewer origin and the eye and lookat positions.

        Args:
            eye: The eye position of the camera. If None, the current eye position is used.
            lookat: The lookat position of the camera. If None, the current lookat position is used.
        """
        # store the camera view pose for later use
        if eye is not None:
            self.default_cam_eye = np.asarray(eye)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat)
        # set the camera locations
        viewer_origin = self.viewer_origin.detach().cpu().numpy()
        cam_eye = viewer_origin + self.default_cam_eye
        cam_target = viewer_origin + self.default_cam_lookat

        # set the camera view
        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)

    """
    Private Functions
    """

    def _update_tracking_callback(self, event):
        """Updates the camera view at each rendering step."""
        # update the camera view if the origin is set to asset_root
        # in other cases, the camera view is static and does not need to be updated continuously
        if (
            self.cfg.origin_type == "asset_root"
            and self.cfg.asset_name is not None
        ):
            self.update_view_to_asset_root(self.cfg.asset_name)


@configclass
class IsaacLabCfg(DirectRLEnvCfg):
    # mode = OmniH2OModes.TRAIN

    # env
    episode_length_s = 3600.0
    substeps = 1
    decimation = 4
    action_scale = 0.25

    num_actions = 19
    num_observations = 913
    observation_space = 913
    action_space = 19
    num_self_obs = 342
    num_ref_obs = 552
    num_action_obs = 19

    num_states = 990

    dt = 0.005

    # If we are doing distill
    distill = False
    single_history_dim = 63
    short_history_length = 25
    distill_teleop_selected_keypoints_names = None

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=substeps,
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )
    # TODO(rhua): using flat terrain until RayCaster is fixed
    terrain = TerrainImporterCfg(
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2, env_spacing=4.0, replicate_physics=True
    )

    # robot
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw",
                ".*_hip_roll",
                ".*_hip_pitch",
                ".*_knee",
                "torso",
            ],
            effort_limit={
                ".*_hip_yaw": 200.0,
                ".*_hip_roll": 200.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 300.0,
                "torso": 200.0,
            },
            velocity_limit={
                ".*_hip_yaw": 23.0,
                ".*_hip_roll": 23.0,
                ".*_hip_pitch": 23.0,
                ".*_knee": 14.0,
                "torso": 23.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=40,
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch",
                ".*_shoulder_roll",
                ".*_shoulder_yaw",
                ".*_elbow",
            ],
            effort_limit={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 18.0,
                ".*_elbow": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch": 9.0,
                ".*_shoulder_roll": 9.0,
                ".*_shoulder_yaw": 20.0,
                ".*_elbow": 20.0,
            },
            stiffness=0,
            damping=0,
        ),
    }

    robot: ArticulationCfg = H1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators=actuators,
    )

    body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    joint_names = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
        "torso",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

    base_name = "torso_link"

    feet_name = ".*_ankle_link"
    knee_name = ".*_knee_link"
    # extend links (these ids are ids after reordered, ie. these id are from IsaacGym, TODO: change to new id directly)
    # h1
    extend_body_parent_ids = [15, 19, 0]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])

    teleop_selected_keypoints_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    # control parameters
    stiffness = {
        ".*_hip_yaw": 200.0,
        ".*_hip_roll": 200.0,
        ".*_hip_pitch": 200.0,
        ".*_knee": 300.0,
        ".*_ankle": 40.0,
        ".*_shoulder_pitch": 100.0,
        ".*_shoulder_roll": 100.0,
        ".*_shoulder_yaw": 100.0,
        ".*_elbow": 100.0,
        "torso": 300.0,
    }

    damping = {
        ".*_hip_yaw": 5.0,
        ".*_hip_roll": 5.0,
        ".*_hip_pitch": 5.0,
        ".*_knee": 6.0,
        ".*_ankle": 2.0,
        ".*_shoulder_pitch": 2.0,
        ".*_shoulder_roll": 2.0,
        ".*_shoulder_yaw": 2.0,
        ".*_elbow": 2.0,
        "torso": 6.0,
    }

    # control type: the action type from the policy
    # "Pos": target joint pos, "Vel": target joint vel, "Torque": joint torques
    control_type = "Pos"

    # Control delay step range (min, max): the control will be randomly delayed at least "min" steps and at most
    # "max" steps. If (0,0), then it means no delay happen
    ctrl_delay_step_range = (0, 3)

    # The default control noise limits: we will add noise to the final torques. the default_rfi_lim defines
    # the default limit of the range of the added noise. It represented by the percentage of the control limits.
    # noise = uniform(-rfi_lim * torque_limits, rfi_lim * torque_limits)
    default_rfi_lim = 0.1

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # domain randomization config
    # events: OmniH2OEventCfg

    # Recovery Counter for Pushed robot: Give steps for the robot to stabilize
    recovery_count = 60

    # Termination conditions
    gravity_x_threshold = 0.7
    gravity_y_threshold = 0.7
    max_ref_motion_dist = 1.5

    # reward scales
    # rewards: RewardCfg = RewardCfg()

    # motion and skeleton files
    reference_motion_path = os.path.join(TEST_DATA_DIR, "stable_punch.pkl")
    skeleton_path = os.path.join(TEST_DATA_DIR, "h1.xml")

    # When we resample reference motions
    resample_motions = True  # if we want to resample reference motions
    resample_motions_for_envs_interval_s = (
        1000  # How many seconds between we resample the reference motions
    )

    # observation noise
    add_policy_obs_noise = True
    policy_obs_noise_level = 1.0
    policy_obs_noise_scales = {
        "body_pos": 0.01,  # body pos in cartesian space: 19x3
        "body_rot": 0.01,  # body pos in cartesian space: 19x3
        "body_lin_vel": 0.01,  # body velocity in cartesian space: 19x3
        "body_ang_vel": 0.01,  # body velocity in cartesian space: 19x3
        "ref_body_pos_diff": 0.05,
        "ref_body_rot_diff": 0.01,
        "ref_body_pos": 0.01,
        "ref_body_rot": 0.01,
        "ref_lin_vel": 0.01,
        "ref_ang_vel": 0.01,
    }


@configclass
class EventCfg:
    """Configuration for events."""

    scale_body_mass = None
    random_joint_friction = None


def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float]
    | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,  # number of environments
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(
            asset.num_bodies, dtype=torch.int, device="cpu"
        )
    else:
        body_ids = torch.tensor(
            asset_cfg.body_ids, dtype=torch.int, device="cpu"
        )

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()
    # apply randomization on default values
    # import ipdb; ipdb.set_trace()
    coms[env_ids[:, None], body_ids] = env.default_coms[
        env_ids[:, None], body_ids
    ].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (
            distribution_params[0].to(coms.device),
            distribution_params[1].to(coms.device),
        )

    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params,
        (env_ids.shape[0], env.base_com_bias.shape[1]),
        device=coms.device,
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[
            env_ids[:, None], :
        ]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[
            env_ids[:, None], :
        ]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[
            env_ids[:, None], :
        ]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)


ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="humanoidverse/data/robots/h1/h1.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_joint": -0.52,  # -30 degrees
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
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
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle_joint": 20.0},
            damping={".*_ankle_joint": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
            },
        ),
    },
)
