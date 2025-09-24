from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import torch
from isaaclab.assets import Articulation
from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)
import isaaclab.utils.math as isaaclab_math
import isaaclab.envs.mdp as mdp


class TerminationFunctions:
    @staticmethod
    def _get_body_indices(
        robot: Articulation, keybody_names: list[str] | None
    ) -> list[int] | None:
        """Convert body names to indices.

        Args:
            robot: Robot articulation asset
            keybody_names: List of body names. If None, returns None.

        Returns:
            List of body indices corresponding to the given names, or None if keybody_names is None
        """
        if keybody_names is None:
            return None

        body_indices = []
        for name in keybody_names:
            if name not in robot.body_names:
                raise ValueError(
                    f"Body '{name}' not found in robot.body_names: {robot.body_names}"
                )
            body_indices.append(robot.body_names.index(name))

        return body_indices

    # @torch.compile
    @staticmethod
    def _get_termination_global_bodylink_pos_far(
        env: ManagerBasedRLEnv,
        threshold: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Any body link position deviates more than threshold (world frame)."""
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        ref_pos_w = command.ref_motion_bodylink_global_pos_cur  # [B, Nb, 3]
        robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

        # Convert body names to indices
        keybody_idxs = TerminationFunctions._get_body_indices(
            command.robot, keybody_names
        )

        if keybody_idxs is not None and len(keybody_idxs) > 0:
            idxs = torch.as_tensor(
                keybody_idxs,
                device=ref_pos_w.device,
                dtype=torch.long,
            )
            ref_pos_w = ref_pos_w[:, idxs]
            robot_pos_w = robot_pos_w[:, idxs]

        error = torch.norm(ref_pos_w - robot_pos_w, dim=-1)  # [B, Nb]
        return torch.any(error > threshold, dim=-1)  # [B]

    # @torch.compile
    @staticmethod
    def _get_termination_anchor_ref_z_far(
        env: ManagerBasedRLEnv,
        threshold: float,
        command_name: str = "ref_motion",
    ) -> torch.Tensor:
        """Anchor link z difference exceeds threshold (world frame)."""
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        ref_z = command.ref_motion_anchor_bodylink_global_pos_cur[:, -1]
        robot_z = command.global_robot_anchor_pos_cur[:, -1]
        return (ref_z - robot_z).abs() > threshold

    # @torch.compile
    @staticmethod
    def _get_termination_ref_gravity_projection_far(
        env: ManagerBasedRLEnv,
        threshold: float,
        asset_name: str = "robot",
        command_name: str = "ref_motion",
    ) -> torch.Tensor:
        """Difference in projected gravity z-component between ref and robot exceeds threshold.

        Follows the provided reference: project world gravity into the anchor body frames
        using inverse quaternion rotation and compare z-components.
        """
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        g_w = env.scene[asset_name].data.GRAVITY_VEC_W  # [B, 3]

        # Reference anchor orientation (xyzw) from motion cache
        ref_anchor_quat_xyzw = command.ref_motion_bodylink_global_rot_wxyz_cur[
            :, command.anchor_bodylink_idx
        ]  # [B, 4]

        motion_projected_gravity_b = isaaclab_math.quat_apply_inverse(
            ref_anchor_quat_xyzw, g_w
        )  # [B, 3]

        # Robot anchor orientation (xyzw) from sim
        robot_anchor_quat_wxyz = command.robot.data.body_quat_w[
            :, command.anchor_bodylink_idx
        ]  # [B, 4]

        robot_projected_gravity_b = isaaclab_math.quat_apply_inverse(
            robot_anchor_quat_wxyz, g_w
        )  # [B, 3]

        return (
            motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
        ).abs() > threshold

    # @torch.compile
    @staticmethod
    def _get_termination_keybody_ref_z_far(
        env: ManagerBasedRLEnv,
        threshold: float,
        command_name: str = "ref_motion",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Any key body link z difference exceeds threshold (world frame)."""
        command: RefMotionCommand = env.command_manager.get_term(command_name)
        ref_pos_w = command.ref_motion_bodylink_global_pos_cur  # [B, Nb, 3]
        robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

        # Convert body names to indices
        keybody_idxs = TerminationFunctions._get_body_indices(
            command.robot, keybody_names
        )

        if keybody_idxs is not None and len(keybody_idxs) > 0:
            idxs = torch.as_tensor(
                keybody_idxs,
                device=ref_pos_w.device,
                dtype=torch.long,
            )
            ref_pos_w = ref_pos_w[:, idxs]
            robot_pos_w = robot_pos_w[:, idxs]

        error_z = (ref_pos_w[..., 2] - robot_pos_w[..., 2]).abs()  # [B, Nb]
        return torch.any(error_z > threshold, dim=-1)  # [B]

    @staticmethod
    def _get_termination_motion_end(
        env: ManagerBasedRLEnv,
        command_name: str = "ref_motion",
    ) -> torch.Tensor:
        """Terminate when reference motion frames exceed their end frames.

        Returns a boolean mask of shape [num_envs].
        """

        command: RefMotionCommand = env.command_manager.get_term(command_name)
        result = command.motion_end_mask.clone().bool()

        return result


@configclass
class TerminationsCfg:
    pass


def build_terminations_config(
    termination_config_dict: dict,
) -> TerminationsCfg:
    # Debug logging to check if function is called and what config is passed
    from loguru import logger

    logger.info(
        f"BUILD_TERMINATIONS_CONFIG CALLED: config_keys={list(termination_config_dict.keys()) if termination_config_dict else 'EMPTY_DICT'}"
    )

    terminations_cfg = TerminationsCfg()

    for termination_name, termination_cfg in termination_config_dict.items():
        # Debug logging (remove after debugging)
        from loguru import logger

        # Check if it's an official IsaacLab termination function
        if hasattr(mdp.terminations, termination_name):
            func = getattr(mdp.terminations, termination_name)
            logger.info(
                f"TERMINATION CONFIG: {termination_name} -> official IsaacLab function"
            )
        else:
            # Otherwise, look for custom termination function
            method_name = f"_get_termination_{termination_name}"
            if not hasattr(TerminationFunctions, method_name):
                raise ValueError(
                    f"Unknown termination function: {termination_name}. "
                    f"Not found in TerminationFunctions or isaaclab.envs.mdp.terminations"
                )
            func = getattr(TerminationFunctions, method_name)
            logger.info(
                f"TERMINATION CONFIG: {termination_name} -> custom function {method_name}"
            )

        # Create termination configuration
        term_cfg = TerminationTermCfg(
            func=func,
            params=termination_cfg.get("params", {}),
            time_out=termination_cfg.get("time_out", False),
        )

        # Debug: log the created configuration
        logger.info(
            f"CREATING TERM CFG: {termination_name}, time_out={term_cfg.time_out}, params={term_cfg.params}"
        )

        setattr(terminations_cfg, termination_name, term_cfg)
    return terminations_cfg
