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
    @staticmethod
    def _get_body_indices(
        robot: Articulation,
        keybody_names: list[str] | None,
    ) -> list[int] | None:
        """Convert body names to indices.

        Args:
            robot: Robot articulation asset
            keybody_names: List of body names. If None, returns None.

        Returns:
            List of body indices corresponding to the given names, or None if keybody_names is None
        """
        if keybody_names is None:
            return list(range(len(robot.body_names)))

        body_indices = []
        for name in keybody_names:
            if name not in robot.body_names:
                raise ValueError(
                    f"Body '{name}' not found in robot.body_names: {robot.body_names}"
                )
            body_indices.append(robot.body_names.index(name))

        return body_indices

    #  @torch.compile
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

    #  @torch.compile
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

    #  @torch.compile
    @staticmethod
    def _get_reward_motion_relative_body_position_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        keybody_idxs = RewardFunctions._get_body_indices(command.robot, keybody_names)

        # Get reference and robot anchor positions/orientations
        ref_anchor_pos = command.ref_motion_root_global_pos_cur  # [B, 3]
        ref_anchor_quat = (
            command.ref_motion_root_global_rot_quat_wxyz_cur
        )  # [B, 4] (w,x,y,z)
        robot_anchor_pos = command.robot.data.body_pos_w[
            :, command.anchor_bodylink_idx
        ]  # [B, 3]
        robot_anchor_quat = command.robot.data.body_quat_w[
            :, command.anchor_bodylink_idx
        ]  # [B, 4] (w,x,y,z)

        # Get reference body positions in global frame
        ref_body_pos_global = (
            command.ref_motion_bodylink_global_pos_cur
        )  # [B, num_bodies, 3]

        # Transform reference body positions to be relative to robot's current anchor
        # This follows the same logic as the whole_body_tracking implementation

        # Select relevant body indices first
        ref_body_pos_selected = ref_body_pos_global[
            :, keybody_idxs
        ]  # [B, selected_bodies, 3]

        # Expand anchor positions/orientations to match number of selected bodies
        num_bodies = len(keybody_idxs)
        ref_anchor_pos_exp = ref_anchor_pos[:, None, :].expand(
            -1, num_bodies, -1
        )  # [B, num_bodies, 3]
        ref_anchor_quat_exp = ref_anchor_quat[:, None, :].expand(
            -1, num_bodies, -1
        )  # [B, num_bodies, 4]
        robot_anchor_pos_exp = robot_anchor_pos[:, None, :].expand(
            -1, num_bodies, -1
        )  # [B, num_bodies, 3]
        robot_anchor_quat_exp = robot_anchor_quat[:, None, :].expand(
            -1, num_bodies, -1
        )  # [B, num_bodies, 4]

        # Create delta transformation (preserving z from reference, aligning xy to robot)
        delta_pos = robot_anchor_pos_exp.clone()
        delta_pos[..., 2] = ref_anchor_pos_exp[..., 2]  # Keep reference Z height

        delta_ori = isaaclab_math.yaw_quat(
            isaaclab_math.quat_mul(
                robot_anchor_quat_exp,
                isaaclab_math.quat_inv(ref_anchor_quat_exp),
            )
        )

        # Transform reference body positions to relative frame
        ref_body_pos_relative = delta_pos + isaaclab_math.quat_apply(
            delta_ori, ref_body_pos_selected - ref_anchor_pos_exp
        )

        # Get robot body positions
        robot_body_pos = command.robot.data.body_pos_w[:, keybody_idxs]

        # Compute error
        error = torch.sum(
            torch.square(ref_body_pos_relative - robot_body_pos),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    #  @torch.compile
    @staticmethod
    def _get_reward_motion_relative_body_orientation_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        keybody_idxs = RewardFunctions._get_body_indices(command.robot, keybody_names)

        # Get reference and robot anchor orientations
        ref_anchor_quat = (
            command.ref_motion_root_global_rot_quat_wxyz_cur
        )  # [B, 4] (w,x,y,z)
        robot_anchor_quat = command.robot.data.body_quat_w[
            :, command.anchor_bodylink_idx
        ]  # [B, 4] (w,x,y,z)

        # Get reference body orientations in global frame
        ref_body_quat_global = (
            command.ref_motion_bodylink_global_rot_wxyz_cur
        )  # [B, num_bodies, 4]

        # Select relevant body indices
        ref_body_quat_selected = ref_body_quat_global[
            :, keybody_idxs
        ]  # [B, selected_bodies, 4]

        # Expand anchor orientations to match number of selected bodies
        num_bodies = len(keybody_idxs)
        ref_anchor_quat_exp = ref_anchor_quat[:, None, :].expand(
            -1, num_bodies, -1
        )  # [B, num_bodies, 4]
        robot_anchor_quat_exp = robot_anchor_quat[:, None, :].expand(
            -1, num_bodies, -1
        )  # [B, num_bodies, 4]

        # Compute relative orientation transformation (only yaw component)
        delta_ori = isaaclab_math.yaw_quat(
            isaaclab_math.quat_mul(
                robot_anchor_quat_exp,
                isaaclab_math.quat_inv(ref_anchor_quat_exp),
            )
        )

        # Transform reference body orientations to relative frame
        ref_body_quat_relative = isaaclab_math.quat_mul(
            delta_ori, ref_body_quat_selected
        )

        # Get robot body orientations
        robot_body_quat = command.robot.data.body_quat_w[:, keybody_idxs]

        # Compute error
        error = (
            isaaclab_math.quat_error_magnitude(ref_body_quat_relative, robot_body_quat)
            ** 2
        )
        return torch.exp(-error.mean(-1) / std**2)

    #  @torch.compile
    @staticmethod
    def _get_reward_motion_global_body_linear_velocity_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        keybody_idxs = RewardFunctions._get_body_indices(command.robot, keybody_names)

        # Direct comparison of global velocities (no coordinate transformation needed)
        error = torch.sum(
            torch.square(
                command.ref_motion_bodylink_global_lin_vel_cur[:, keybody_idxs]
                - command.robot.data.body_lin_vel_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    #  @torch.compile
    @staticmethod
    def _get_reward_motion_global_body_angular_velocity_error_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        # Get body indexes based on body names (similar to whole_body_tracking implementation)
        keybody_idxs = RewardFunctions._get_body_indices(command.robot, keybody_names)

        # Direct comparison of global angular velocities (no coordinate transformation needed)
        error = torch.sum(
            torch.square(
                command.ref_motion_bodylink_global_ang_vel_cur[:, keybody_idxs]
                - command.robot.data.body_ang_vel_w[:, keybody_idxs]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / std**2)

    #  @torch.compile
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

    #  @torch.compile
    @staticmethod
    def _get_reward_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.ones(env.num_envs, device=env.device)

    #  @torch.compile
    @staticmethod
    def _get_reward_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize the rate of change of the actions using L2 squared kernel."""
        return isaaclab_mdp.action_rate_l2(env)

    #  @torch.compile
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

    #  @torch.compile
    @staticmethod
    def _get_reward_undesired_contacts(
        env: ManagerBasedRLEnv,
        sensor_name: str,
        body_names: list[str],
        threshold: float,
    ) -> torch.Tensor:
        """Penalize undesired contacts as the number of violations above a threshold."""
        return isaaclab_mdp.undesired_contacts(
            env,
            sensor_cfg=SceneEntityCfg(sensor_name, body_names=body_names),
            threshold=threshold,
        )


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
