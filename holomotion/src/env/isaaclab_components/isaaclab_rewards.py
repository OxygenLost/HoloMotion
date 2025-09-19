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
import isaaclab.utils.math as isaaclab_math
import torch
from isaaclab.sensors import ContactSensor

from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)


class RewardFunctions:
    @torch.compile
    @staticmethod
    def _get_reward_motion_global_anchor_position_error_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
    ) -> torch.Tensor:
        ref_motion_command: RefMotionCommand = env.command_manager.get_term(
            command_name
        )
        error = torch.sum(
            torch.square(
                ref_motion_command.ref_motion_anchor_bodylink_global_pos_cur
                - ref_motion_command.global_robot_anchor_pos_cur
            ),
            dim=-1,
        )
        return torch.exp(-error / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_global_anchor_orientation_error_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        error = (
            isaaclab_math.quat_error_magnitude(
                command.anchor_quat_w, command.robot_anchor_quat_w
            )
            ** 2
        )
        return torch.exp(-error / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_relative_body_position_error_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        error = torch.sum(
            torch.square(
                command.body_pos_relative_w[:, keybody_idxs]
                - command.robot_body_pos_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_relative_body_orientation_error_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        error = (
            isaaclab_math.quat_error_magnitude(
                command.body_quat_relative_w[:, keybody_idxs],
                command.robot_body_quat_w[:, keybody_idxs],
            )
            ** 2
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_global_body_linear_velocity_error_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        error = torch.sum(
            torch.square(
                command.body_lin_vel_w[:, keybody_idxs]
                - command.robot_body_lin_vel_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_global_body_angular_velocity_error_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        error = torch.sum(
            torch.square(
                command.body_ang_vel_w[:, keybody_idxs]
                - command.robot_body_ang_vel_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_feet_contact_time(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first_air = contact_sensor.compute_first_air(
            env.step_dt, env.physics_dt
        )[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[
            :, sensor_cfg.body_ids
        ]
        reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
        return reward

    @torch.compile
    @staticmethod
    def _get_reward_alive(
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
        return torch.ones(env.num_envs, device=env.device)


def build_rewards_config(reward_config_dict: dict):
    @configclass
    class RewardsCfg:
        pass

    rewards_cfg = RewardsCfg()

    for reward_name, reward_cfg in reward_config_dict.items():
        method_name = f"_get_reward_{reward_name}"

        if not hasattr(RewardFunctions, method_name):
            raise ValueError(f"Unknown reward function: {reward_name}")

        func = getattr(RewardFunctions, method_name)

        setattr(
            rewards_cfg,
            reward_name,
            RewardTermCfg(
                func=func,
                weight=reward_cfg["weight"],
                params=reward_cfg["params"],
            ),
        )
    return rewards_cfg
