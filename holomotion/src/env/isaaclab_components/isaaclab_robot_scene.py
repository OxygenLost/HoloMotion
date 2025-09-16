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
from isaaclab.managers import EventTermCfg as EventTerm
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
import math
import os
from loguru import logger


urdf_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/assets/robots/unitree/G1/23dof/official_g1_23dof_rev_1_0.urdf"
usd_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/assets/robots/unitree/G1/23dof"

if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")

logger.info(f"Using URDF path: {urdf_path}")
logger.info(f"Using USD directory: {usd_dir}")


class PDParamsCalculator:
    def __init__(
        self, natural_freq_hz: float = 10.0, damping_ratio: float = 2.0
    ):
        self.natural_freq = natural_freq_hz * 2.0 * math.pi
        self.damping_ratio = damping_ratio

    def armature_to_stiffness(self, armature: float) -> float:
        return armature * self.natural_freq**2

    def armature_to_damping(self, armature: float) -> float:
        return 2.0 * self.damping_ratio * armature * self.natural_freq


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925

# ignore wrist motors for now
# ARMATURE_4010 = 0.00425
# STIFFNESS_4010 = pd_params_calculator.armature_to_stiffness(
#     ARMATURE_4010
# )
# DAMPING_4010 = pd_params_calculator.armature_to_damping(ARMATURE_4010)

pd_params_calculator = PDParamsCalculator(
    natural_freq_hz=10.0,
    damping_ratio=2.0,
)

STIFFNESS_5020 = pd_params_calculator.armature_to_stiffness(ARMATURE_5020)
DAMPING_5020 = pd_params_calculator.armature_to_damping(ARMATURE_5020)

STIFFNESS_7520_14 = pd_params_calculator.armature_to_stiffness(
    ARMATURE_7520_14
)
DAMPING_7520_14 = pd_params_calculator.armature_to_damping(ARMATURE_7520_14)

STIFFNESS_7520_22 = pd_params_calculator.armature_to_stiffness(
    ARMATURE_7520_22
)
DAMPING_7520_22 = pd_params_calculator.armature_to_damping(ARMATURE_7520_22)


@configclass
class MotionTrackingSceneCfg(InteractiveSceneCfg):
    num_envs = 2048
    env_spacing = 4.0
    replicate_physics = True

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

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
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
                    stiffness=0,
                    damping=0,
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
