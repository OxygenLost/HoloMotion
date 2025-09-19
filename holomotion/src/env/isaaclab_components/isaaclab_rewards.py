import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
import isaaclab.utils.math as isaaclab_math

from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)
import isaaclab.envs.mdp as isaaclab_mdp


class RewardFunctions:
    @torch.compile
    @staticmethod
    def _get_reward_motion_global_anchor_position_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
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
        std: float,
        command_name: str = "ref_motion",
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        error = (
            isaaclab_math.quat_error_magnitude(
                command.ref_motion_anchor_bodylink_global_rot_cur_wxyz,
                command.robot.data.body_quat_w[:, command.anchor_bodylink_idx],
            )
            ** 2
        )
        return torch.exp(-error / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_relative_body_position_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        if keybody_names is None:
            keybody_idxs = None
        else:
            keybody_idxs = [
                i
                for i, name in enumerate(command.robot.body_names)
                if name in keybody_names
            ]
        error = torch.sum(
            torch.square(
                command.ref_motion_bodylink_global_pos_cur[:, keybody_idxs]
                - command.robot.data.body_pos_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_relative_body_orientation_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        if keybody_names is None:
            keybody_idxs = None
        else:
            keybody_idxs = [
                i
                for i, name in enumerate(command.robot.body_names)
                if name in keybody_names
            ]
        error = (
            isaaclab_math.quat_error_magnitude(
                command.ref_motion_bodylink_global_rot_wxyz_cur[:, keybody_idxs],
                command.robot.data.body_quat_w[:, keybody_idxs],
            )
            ** 2
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_global_body_linear_velocity_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        if keybody_names is None:
            keybody_idxs = None
        else:
            keybody_idxs = [
                i
                for i, name in enumerate(command.robot.body_names)
                if name in keybody_names
            ]
        error = torch.sum(
            torch.square(
                command.ref_motion_bodylink_global_lin_vel_cur[:, keybody_idxs]
                - command.robot.data.body_lin_vel_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    @torch.compile
    @staticmethod
    def _get_reward_motion_global_body_angular_velocity_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        if keybody_names is None:
            keybody_idxs = None
        else:
            keybody_idxs = [
                i
                for i, name in enumerate(command.robot.body_names)
                if name in keybody_names
            ]
        error = torch.sum(
            torch.square(
                command.ref_motion_bodylink_global_ang_vel_cur[:, keybody_idxs]
                - command.robot.data.body_ang_vel_w[:, keybody_idxs]
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
        first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[
            :, sensor_cfg.body_ids
        ]
        last_contact_time = contact_sensor.data.last_contact_time[
            :, sensor_cfg.body_ids
        ]
        reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
        return reward

    @torch.compile
    @staticmethod
    def _get_reward_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.ones(env.num_envs, device=env.device)

    @torch.compile
    @staticmethod
    def _get_reward_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize the rate of change of the actions using L2 squared kernel."""
        return torch.sum(
            torch.square(env.action_manager.action - env.action_manager.prev_action),
            dim=1,
        )

    @torch.compile
    @staticmethod
    def _get_reward_joint_pos_limits(
        env: ManagerBasedRLEnv,
        asset_name: str = "robot",
        joint_names=(".*"),
    ) -> torch.Tensor:
        """Penalize joint positions if they cross the soft limits."""
        return isaaclab_mdp.joint_pos_limits(
            env,
            SceneEntityCfg(
                asset_name,
                joint_names=joint_names,
            ),
        )

    @torch.compile
    @staticmethod
    def _get_reward_undesired_contacts(
        env: ManagerBasedRLEnv,
        sensor_name: str,
        body_names: list[str],
        threshold: float,
    ) -> torch.Tensor:
        """Penalize undesired contacts as the number of violations above a threshold."""
        sensor_cfg = SceneEntityCfg(sensor_name, body_names=body_names)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > threshold
        )
        # sum over contacts for each environment
        return torch.sum(is_contact, dim=1)


@configclass
class RewardsCfg:
    pass


def build_rewards_config(reward_config_dict: dict):
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
