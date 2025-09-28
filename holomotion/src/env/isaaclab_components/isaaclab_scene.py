import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from loguru import logger
from omegaconf import OmegaConf
import easydict


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

G1_CYLINDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"/home/maiyue01.chen/project3/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf",
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
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
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
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

G1_ACTION_SCALE = {}
for a in G1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


class SceneFunctions:
    """Collection of scene component builders."""

    @staticmethod
    def build_robot_config(config: dict) -> ArticulationCfg:
        """Build robot articulation configuration."""
        urdf_path = config.asset.urdf_file
        init_pos = config.init_state.pos
        default_joint_positions = config.init_state.default_joint_angles
        # prim_path = config.get("prim_path", "/World/envs/env_.*/Robot")
        prim_path="{ENV_REGEX_NS}/Robot"
        actuators = {
            "all_joints": ImplicitActuatorCfg(**config.actuators.all_joints)
        }

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        usd_dir = os.path.join(os.path.dirname(urdf_path), "usd")
        os.makedirs(usd_dir, exist_ok=True)
        logger.info(f"Using URDF path: {urdf_path}")
        logger.info(f"Using USD directory: {usd_dir}")

        # G1_CYLINDER_CFG.prim_path = prim_path
        # G1_CYLINDER_CFG.spawn.asset_path = urdf_path
        # return G1_CYLINDER_CFG

        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.path.abspath(urdf_path),
                usd_dir=os.path.abspath(usd_dir),
                fix_base=False,
                merge_fixed_joints=True,
                root_link_name="pelvis",
                replace_cylinders_with_capsules=True,
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
                        stiffness=0,
                        damping=0,
                    )
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                joint_pos=default_joint_positions,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.9,
            actuators=actuators,
        )

    @staticmethod
    def build_terrain_config(config: dict) -> TerrainImporterCfg:
        """Build terrain configuration."""
        terrain_type = config.get("terrain_type", "plane")
        prim_path = config.get("prim_path", "/World/ground")
        static_friction = config.get("static_friction", 1.0)
        dynamic_friction = config.get("dynamic_friction", 1.0)
        friction_combine_mode = config.get("friction_combine_mode", "multiply")
        restitution_combine_mode = config.get(
            "restitution_combine_mode", "multiply"
        )
        env_spacing = 2.5

        return TerrainImporterCfg(
            prim_path=prim_path,
            terrain_type=terrain_type,
            collision_group=-1,
            env_spacing=env_spacing,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=friction_combine_mode,
                restitution_combine_mode=restitution_combine_mode,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
            ),
        )

    @staticmethod
    def build_lighting_config(
        config: dict,
    ) -> tuple[AssetBaseCfg, AssetBaseCfg]:
        """Build lighting configuration."""
        distant_light_intensity = config.get("distant_light_intensity", 3000.0)
        dome_light_intensity = config.get("dome_light_intensity", 1000.0)
        distant_light_color = config.get(
            "distant_light_color", (0.75, 0.75, 0.75)
        )
        dome_light_color = config.get("dome_light_color", (0.13, 0.13, 0.13))

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=distant_light_color, intensity=distant_light_intensity
            ),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=dome_light_color, intensity=dome_light_intensity
            ),
        )
        return light, sky_light

    @staticmethod
    def build_contact_sensor_config(config: dict) -> ContactSensorCfg:
        """Build contact sensor configuration."""
        prim_path = config.get("prim_path", "{ENV_REGEX_NS}/Robot/.*")
        history_length = config.get("history_length", 3)
        force_threshold = config.get("force_threshold", 10.0)
        track_air_time = config.get("track_air_time", True)
        debug_vis = config.get("debug_vis", False)

        return ContactSensorCfg(
            prim_path=prim_path,
            history_length=history_length,
            track_air_time=track_air_time,
            force_threshold=force_threshold,
            debug_vis=debug_vis,
        )


@configclass
class MotionTrackingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for motion tracking environment."""

    pass


def build_scene_config(scene_config_dict: dict) -> MotionTrackingSceneCfg:
    """Build IsaacLab-compatible scene configuration from config dictionary."""
    scene_cfg = MotionTrackingSceneCfg()

    # Basic scene properties
    scene_cfg.num_envs = scene_config_dict.get("num_envs", MISSING)
    scene_cfg.env_spacing = scene_config_dict.get("env_spacing", 4.0)
    scene_cfg.replicate_physics = scene_config_dict.get(
        "replicate_physics", True
    )

    # Build robot configuration
    if "robot" in scene_config_dict:
        robot_config = scene_config_dict["robot"]
        scene_cfg.robot = SceneFunctions.build_robot_config(robot_config)

    # Build terrain configuration
    if "terrain" in scene_config_dict:
        terrain_config = scene_config_dict["terrain"]
        scene_cfg.terrain = SceneFunctions.build_terrain_config(terrain_config)
        # scene_cfg.terrain.physics_material.improve_patch_friction = True

    # Build lighting configuration
    if "lighting" in scene_config_dict:
        lighting_config = scene_config_dict["lighting"]
        light, sky_light = SceneFunctions.build_lighting_config(
            lighting_config
        )
        scene_cfg.light = light
        scene_cfg.sky_light = sky_light

    # Build contact sensor configuration
    if "contact_sensor" in scene_config_dict:
        contact_config = scene_config_dict["contact_sensor"]
        scene_cfg.contact_forces = SceneFunctions.build_contact_sensor_config(
            contact_config
        )

    return scene_cfg
