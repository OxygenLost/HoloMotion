import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
import torch
from isaaclab.markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


import isaaclab.utils.math as isaaclab_math
import isaaclab.utils.noise as isaaclab_noise


class ObservationFunctions:
    """Atomic observation functions.

    The most foundamental observation functions are defined here, aiming to
    utize the convenient functions from isaaclab apis. For complex observation
    composition patterns, we'll use the custom observation serizliazer.
    """

    @staticmethod
    def _get_body_indices(
        robot: Articulation, keybody_names: list[str] | None
    ) -> list[int]:
        """Convert body names to indices.

        Args:
            robot: Robot articulation asset
            keybody_names: List of body names. If None, returns all body indices.

        Returns:
            List of body indices corresponding to the given names
        """
        if keybody_names is None:
            return list(range(robot.num_bodies))

        body_indices = []
        for name in keybody_names:
            if name not in robot.body_names:
                raise ValueError(
                    f"Body '{name}' not found in robot.body_names: {robot.body_names}"
                )
            body_indices.append(robot.body_names.index(name))

        return body_indices

    # ------- Robot Root States -------
    @staticmethod
    def _get_obs_global_robot_root_pos(env: ManagerBasedRLEnv):
        """Asset root position in the environment frame."""
        return isaaclab_mdp.root_pos_w(env)

    @staticmethod
    def _get_obs_global_robot_root_rot_wxyz(env: ManagerBasedRLEnv):
        """Asset root orientation (w, x, y, z) in the environment frame."""
        return isaaclab_mdp.root_quat_w(env)

    @staticmethod
    def _get_obs_global_robot_root_rot_xyzw(env: ManagerBasedRLEnv):
        """Asset root orientation (x, y, z, w) in the environment frame."""
        return ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)[
            ..., [1, 2, 3, 0]
        ]

    @staticmethod
    def _get_obs_global_robot_root_rot_mat(env: ManagerBasedRLEnv):
        """Asset root orientation as a 3x3 matrix, flattened to the first two rows (6D)."""
        return isaaclab_math.matrix_from_quat(
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )[..., :2]  # [num_envs, 6]

    @staticmethod
    def _get_obs_global_robot_root_lin_vel(env: ManagerBasedRLEnv):
        """Asset root linear velocity in the environment frame."""
        return isaaclab_mdp.root_lin_vel_w(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_global_robot_root_ang_vel(env: ManagerBasedRLEnv):
        """Asset root angular velocity in the environment frame."""
        return isaaclab_mdp.root_ang_vel_w(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_robot_root_lin_vel(env: ManagerBasedRLEnv):
        """Relative root linear velocity in the root frame."""
        return isaaclab_mdp.base_lin_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_robot_root_ang_vel(env: ManagerBasedRLEnv):
        """Relative root angular velocity in the root frame."""
        return isaaclab_mdp.base_ang_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_global_robot_root_yaw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """Robot's yaw heading in the environment frame (in radians)."""
        robot_ptr = env.scene[robot_asset_name]
        return robot_ptr.data.heading_w  # [num_envs, ]

    # @torch.compile
    @staticmethod
    def _get_obs_robot_root_heading_aligned_quat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """A quaternion representing only the robot's yaw heading."""
        global_yaw = ObservationFunctions._get_obs_global_robot_root_yaw(
            env,
            robot_asset_name,
        )  # [num_envs, ]
        zero_roll = torch.zeros_like(global_yaw, device=env.device)
        zero_pitch = torch.zeros_like(global_yaw, device=env.device)
        heading_aligned_quat = isaaclab_math.quat_from_angle_axis(
            roll=zero_roll,
            pitch=zero_pitch,
            yaw=global_yaw,
        )  # [num_envs, 4]
        return heading_aligned_quat  # [num_envs, 4]

    # @torch.compile
    @staticmethod
    def _get_obs_rel_robot_root_roll_pitch(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """Robot's roll and pitch relative to its heading-aligned frame."""
        heading_aligned_quat = (
            ObservationFunctions._get_obs_robot_root_heading_aligned_quat(
                env,
                robot_asset_name,
            )
        )  # [num_envs, 4]
        robot_quat_in_heading_aligned_frame = isaaclab_math.quat_mul(
            isaaclab_math.quat_inv(heading_aligned_quat),
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env),
        )  # [num_envs, 4]
        rel_roll, rel_pitch, _ = isaaclab_math.get_euler_xyz(
            robot_quat_in_heading_aligned_frame
        )  # [num_envs, 3]
        return torch.stack([rel_roll, rel_pitch], dim=-1)  # [num_envs, 2]

    # ------- Robot Bodylink States -------
    @staticmethod
    def _get_obs_global_robot_bodylink_pos(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Positions of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_pos = robot_ptr.data.body_pos_w[:, keybody_idxs]
        return keybody_global_pos  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_wxyz(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Orientations (w, x, y, z) of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_rot = robot_ptr.data.body_quat_w[:, keybody_idxs]
        return keybody_global_rot  # [num_envs, num_keybodies, 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_xyzw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Orientations (x, y, z, w) of specified bodylinks in the environment frame."""
        return ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
            env,
            robot_asset_name,
            keybody_names,
        )[..., [1, 2, 3, 0]]  # [num_envs, num_keybodies, 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_mat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Orientations of specified bodylinks as a 3x3 matrix, flattened to the first two rows (6D)."""
        keybody_global_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env,
                robot_asset_name,
                keybody_names,
            )
        )
        return isaaclab_math.matrix_from_quat(keybody_global_rot_wxyz)[
            ..., :2
        ]  # [num_envs, num_keybodies, 6]

    @staticmethod
    def _get_obs_global_robot_bodylink_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Linear velocities of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_lin_vel = robot_ptr.data.body_lin_vel_w[:, keybody_idxs]
        return keybody_global_lin_vel  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_ang_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ):
        """Angular velocities of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        keybody_idxs = ObservationFunctions._get_body_indices(
            robot_ptr, keybody_names
        )
        keybody_global_ang_vel = robot_ptr.data.body_ang_vel_w[:, keybody_idxs]
        return keybody_global_ang_vel  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_pos(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Positions of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_pos: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]

        global_root_pos: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_pos(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Transform to root frame
        # Position relative to root
        rel_pos_global: torch.Tensor = (
            keybody_global_pos - global_root_pos[..., None, :]
        )  # [num_envs, num_keybodies, 3]

        # Rotate to root frame using inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        num_bodies = keybody_global_pos.shape[1]
        rel_pos_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot[..., None, :].expand(-1, num_bodies, -1),
            rel_pos_global,
        )  # [num_envs, num_keybodies, 3]

        return rel_pos_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_wxyz(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 4]
        """Orientations (w, x, y, z) of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_rot: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]

        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Transform to root frame by multiplying with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        num_bodies = keybody_global_rot.shape[1]
        rel_rot_root: torch.Tensor = isaaclab_math.quat_mul(
            root_inv_rot[..., None, :].expand(-1, num_bodies, -1),
            keybody_global_rot,
        )  # [num_envs, num_keybodies, 4]

        return rel_rot_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_xyzw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 4]
        """Orientations (x, y, z, w) of specified bodylinks relative to the robot's root frame."""
        return ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
            env, robot_asset_name, keybody_names
        )[
            ..., [1, 2, 3, 0]
        ]  # [num_envs, num_keybodies, 4] - convert WXYZ to XYZW

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_mat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 6]
        """Orientations of specified bodylinks relative to the robot's root frame, as a 3x3 matrix, flattened to the first two rows (6D)."""
        keybody_rel_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]

        return isaaclab_math.matrix_from_quat(keybody_rel_rot_wxyz)[
            ..., :2
        ]  # [num_envs, num_keybodies, 6]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Linear velocities of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_lin_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        root_global_lin_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_lin_vel(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Compute relative velocity in world frame
        rel_lin_vel_w = keybody_global_lin_vel - root_global_lin_vel.unsqueeze(
            1
        )

        # Transform to root frame by rotating with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        rel_lin_vel_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot.unsqueeze(1), rel_lin_vel_w
        )  # [num_envs, num_keybodies, 3]

        return rel_lin_vel_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_ang_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Angular velocities of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_ang_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        root_global_ang_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_ang_vel(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )  # [num_envs, 4]

        # Compute relative angular velocity in world frame
        rel_ang_vel_w = keybody_global_ang_vel - root_global_ang_vel.unsqueeze(
            1
        )

        # Transform to root frame by rotating with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        rel_ang_vel_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot.unsqueeze(1), rel_ang_vel_w
        )  # [num_envs, num_keybodies, 3]

        return rel_ang_vel_root

    # ------- Flat Bodylink Observations -------
    @staticmethod
    def _get_obs_global_robot_bodylink_pos_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened positions of specified bodylinks in the environment frame."""
        bodylink_pos = ObservationFunctions._get_obs_global_robot_bodylink_pos(
            env, robot_asset_name, keybody_names
        )  # [num_envs, num_keybodies, 3]
        return bodylink_pos.reshape(
            bodylink_pos.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_wxyz_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (w, x, y, z) of specified bodylinks in the environment frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_xyzw_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (x, y, z, w) of specified bodylinks in the environment frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_xyzw(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_mat_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 6]
        """Flattened orientation matrices (6D) of specified bodylinks in the environment frame."""
        bodylink_rot_mat = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_mat(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 6]
        return bodylink_rot_mat.reshape(
            bodylink_rot_mat.shape[0], -1
        )  # [num_envs, num_keybodies * 6]

    @staticmethod
    def _get_obs_global_robot_bodylink_lin_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened linear velocities of specified bodylinks in the environment frame."""
        bodylink_lin_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_lin_vel.reshape(
            bodylink_lin_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_ang_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened angular velocities of specified bodylinks in the environment frame."""
        bodylink_ang_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_ang_vel.reshape(
            bodylink_ang_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_pos_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened positions of specified bodylinks relative to the robot's root frame."""
        bodylink_pos = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_pos(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_pos.reshape(
            bodylink_pos.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_wxyz_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (w, x, y, z) of specified bodylinks relative to the robot's root frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_xyzw_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 4]
        """Flattened orientations (x, y, z, w) of specified bodylinks relative to the robot's root frame."""
        bodylink_rot = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_xyzw(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 4]
        return bodylink_rot.reshape(
            bodylink_rot.shape[0], -1
        )  # [num_envs, num_keybodies * 4]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_mat_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 6]
        """Flattened orientation matrices (6D) of specified bodylinks relative to the robot's root frame."""
        bodylink_rot_mat = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_mat(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 6]
        return bodylink_rot_mat.reshape(
            bodylink_rot_mat.shape[0], -1
        )  # [num_envs, num_keybodies * 6]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_lin_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened linear velocities of specified bodylinks relative to the robot's root frame."""
        bodylink_lin_vel = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_lin_vel.reshape(
            bodylink_lin_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_ang_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        """Flattened angular velocities of specified bodylinks relative to the robot's root frame."""
        bodylink_ang_vel = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_ang_vel.reshape(
            bodylink_ang_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    # ------- Robot DoF States -------
    @staticmethod
    def _get_obs_dof_pos(env: ManagerBasedRLEnv):
        """Joint positions relative to the default joint angles."""
        return isaaclab_mdp.joint_pos_rel(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_dof_vel(env: ManagerBasedRLEnv):
        """Joint velocities."""
        return isaaclab_mdp.joint_vel_rel(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_last_actions(env: ManagerBasedRLEnv):
        """Last action output by the policy."""
        return isaaclab_mdp.last_action(env)  # [num_envs, num_actions]

    # ------- Reference Motion States -------
    @staticmethod
    def _get_obs_ref_motion_states(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
    ):
        """Reference motion states.

        This function should return the reference motion command which
        has already been serialized into flattened vectors.
        """
        return isaaclab_mdp.generated_commands(
            env,
            command_name=ref_motion_command_name,
        )  # [num_envs, ref_motion_states_dim]

    # @torch.compile
    @staticmethod
    def _get_obs_global_anchor_diff(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        ref_motion_command_name: str = "ref_motion",
    ):
        command = env.command_manager.get_term(ref_motion_command_name)
        global_ref_motion_anchor_pos = (
            command.ref_motion_anchor_bodylink_global_pos_cur
        )
        global_ref_motino_anchor_rot_wxyz = (
            command.ref_motion_anchor_bodylink_global_rot_cur_wxyz
        )
        global_robot_anchor_pos = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        global_robot_anchor_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, [command.anchor_bodylink_name]
            ).squeeze(1)
        )
        pos_diff, rot_diff = isaaclab_math.subtract_frame_transforms(
            t01=global_robot_anchor_pos,
            q01=global_robot_anchor_rot_wxyz,
            t02=global_ref_motion_anchor_pos,
            q02=global_ref_motino_anchor_rot_wxyz,
        )
        rot_diff_mat = isaaclab_math.matrix_from_quat(rot_diff)
        return torch.cat(
            [
                pos_diff,
                rot_diff_mat[..., :2].reshape(env.num_envs, -1),
            ],
            dim=-1,
        )  # [num_envs, 9]


@configclass
class ObservationsCfg:
    pass


def build_observations_config(obs_config_dict: dict):
    """Build isaaclab-compatible ObservationsCfg from a config dictionary."""

    obs_cfg = ObservationsCfg()

    # Create observation groups dynamically
    for group_name, group_cfg in obs_config_dict.items():
        # Create a dynamic ObsGroup class for this group
        @configclass
        class DynamicObsGroup(ObsGroup):
            def __post_init__(self, group_cfg=group_cfg):
                self.enable_corruption = group_cfg["enable_corruption"]
                self.concatenate_terms = group_cfg["concatenate_terms"]

        isaaclab_obs_group_cfg = DynamicObsGroup()

        # Add observation terms to the group
        for obs_term_dict in group_cfg["atomic_obs_list"]:
            for obs_name, obs_params in obs_term_dict.items():
                # Look for observation function in ObservationFunctions class
                method_name = f"_get_obs_{obs_name}"

                if hasattr(ObservationFunctions, method_name):
                    # Use custom observation function
                    func = getattr(ObservationFunctions, method_name)
                elif hasattr(isaaclab_mdp, obs_name):
                    # Use isaaclab isaaclab_mdp function directly
                    func = getattr(isaaclab_mdp, obs_name)
                else:
                    raise ValueError(
                        f"Unknown observation function: {obs_name}"
                    )

                # Extract parameters
                params = obs_params.get("params", {})
                noise_cfg = obs_params.get("noise", None)

                # Create noise config if specified
                noise = None
                if noise_cfg:
                    noise = getattr(isaaclab_noise, noise_cfg["type"])(
                        **noise_cfg["params"]
                    )

                # Create observation term
                obs_term = (
                    ObsTerm(
                        func=func,
                        params=params,
                        noise=noise,
                    )
                    if noise
                    else ObsTerm(
                        func=func,
                        params=params,
                    )
                )

                # Add observation term to group
                setattr(isaaclab_obs_group_cfg, obs_name, obs_term)

        # Add group to main observations config
        setattr(obs_cfg, group_name, isaaclab_obs_group_cfg)

    return obs_cfg
