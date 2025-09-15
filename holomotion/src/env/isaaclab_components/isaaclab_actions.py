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


@configclass
class ActionsCfg:
    joint_efforts: JointEffortActionCfg = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
    )
