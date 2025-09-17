import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.actions import JointEffortActionCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


# @configclass
# class ObservationsCfg:
#     @configclass
#     class ActorCfg(ObsGroup):
#         base_lin_vel = ObsTerm(
#             func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
#         )
#         base_ang_vel = ObsTerm(
#             func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
#         )
#         joint_pos = ObsTerm(
#             func=mdp.joint_pos_rel,
#             noise=Unoise(n_min=-0.01, n_max=0.01),
#         )
#         joint_vel = ObsTerm(
#             func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
#         )
#         actions = ObsTerm(func=mdp.last_action)
#         ref_motion_flat = ObsTerm(
#             func=mdp.generated_commands,
#             params={"command_name": "ref_motion"},
#         )

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     @configclass
#     class PrivilegedCfg(ObsGroup):
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         actions = ObsTerm(func=mdp.last_action)
#         ref_motion_flat = ObsTerm(
#             func=mdp.generated_commands,
#             params={"command_name": "ref_motion"},
#         )

#     # observation groups
#     actor_obs: ActorCfg = ActorCfg()
#     critic_obs: PrivilegedCfg = PrivilegedCfg()


class ObservationFunctions:
    """Atomic observation functions.

    The most foundamental observation functions are defined here, aiming to
    utize the convenient functions from isaaclab apis. For complex observation
    composition patterns, we'll use the custom observation serizliazer.
    """

    # ------- Robot Root States -------
    @staticmethod
    def _get_obs_global_robot_root_pos(env: ManagerBasedRLEnv):
        """Asset root position in the environment frame."""
        return mdp.root_pos_w(env)

    @staticmethod
    def _get_obs_global_robot_root_rot_wxyz(env: ManagerBasedRLEnv):
        """Asset root orientation (w, x, y, z) in the environment frame."""
        return mdp.root_quat_w(env)

    @staticmethod
    def _get_obs_global_robot_root_rot_xyzw(env: ManagerBasedRLEnv):
        """Asset root orientation (x, y, z, w) in the environment frame."""
        return ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)[
            ..., [1, 2, 3, 0]
        ]

    @staticmethod
    def _get_obs_global_robot_root_rot_mat(env: ManagerBasedRLEnv):
        """Asset root orientation as a 3x3 matrix, flattened to the first two rows (6D)."""
        return isaaclab_math.quaternion_to_matrix(
            ObservationFunctions._get_obs_global_robot_root_rot_wxyz(env)
        )[..., :2].reshape(
            env.num_envs,
            -1,
        )  # [num_envs, 6]

    @staticmethod
    def _get_obs_global_robot_root_lin_vel(env: ManagerBasedRLEnv):
        """Asset root linear velocity in the environment frame."""
        return mdp.root_lin_vel_w(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_global_robot_root_ang_vel(env: ManagerBasedRLEnv):
        """Asset root angular velocity in the environment frame."""
        return mdp.root_ang_vel_w(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_root_lin_vel(env: ManagerBasedRLEnv):
        """Relative root linear velocity in the root frame."""
        return mdp.rel_root_lin_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_root_ang_vel(env: ManagerBasedRLEnv):
        """Relative root angular velocity in the root frame."""
        return mdp.rel_root_ang_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_global_root_yaw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """Robot's yaw heading in the environment frame (in radians)."""
        robot_ptr = env.scene[robot_asset_name]
        return robot_ptr.data.heading_w  # [num_envs, ]

    @torch.compile
    @staticmethod
    def _get_obs_root_heading_aligned_quat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """A quaternion representing only the robot's yaw heading."""
        global_yaw = ObservationFunctions._get_obs_global_root_yaw(
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

    @torch.compile
    @staticmethod
    def _get_obs_rel_root_roll_pitch(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
    ):
        """Robot's roll and pitch relative to its heading-aligned frame."""
        heading_aligned_quat = (
            ObservationFunctions._get_obs_root_heading_aligned_quat(
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
        keybody_idxs: list[int] | None = None,
    ):
        """Positions of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        if keybody_idxs is None:
            keybody_idxs = list(range(robot_ptr.num_bodies))
        keybody_global_pos = robot_ptr.data.body_pos_w[:, keybody_idxs]
        return keybody_global_pos  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_wxyz(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ):
        """Orientations (w, x, y, z) of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        if keybody_idxs is None:
            keybody_idxs = list(range(robot_ptr.num_bodies))
        keybody_global_rot = robot_ptr.data.body_rot_w[:, keybody_idxs]
        return keybody_global_rot  # [num_envs, num_keybodies, 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_xyzw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ):
        """Orientations (x, y, z, w) of specified bodylinks in the environment frame."""
        return ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
            env,
            robot_asset_name,
            keybody_idxs,
        )[..., [1, 2, 3, 0]]  # [num_envs, num_keybodies, 4]

    @staticmethod
    def _get_obs_global_robot_bodylink_rot_mat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ):
        """Orientations of specified bodylinks as a 3x3 matrix, flattened to the first two rows (6D)."""
        keybody_global_rot_wxyz = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env,
                robot_asset_name,
                keybody_idxs,
            )
        )
        return isaaclab_math.quaternion_to_matrix(keybody_global_rot_wxyz)[
            ..., :2
        ].reshape(env.num_envs, -1)  # [num_envs, num_keybodies * 6]

    @staticmethod
    def _get_obs_global_robot_bodylink_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ):
        """Linear velocities of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        if keybody_idxs is None:
            keybody_idxs = list(range(robot_ptr.num_bodies))
        keybody_global_lin_vel = robot_ptr.data.body_lin_vel_w[:, keybody_idxs]
        return keybody_global_lin_vel  # [num_envs, num_keybodies, 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_ang_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ):
        """Angular velocities of specified bodylinks in the environment frame."""
        robot_ptr = env.scene[robot_asset_name]
        if keybody_idxs is None:
            keybody_idxs = list(range(robot_ptr.num_bodies))
        keybody_global_ang_vel = robot_ptr.data.body_ang_vel_w[:, keybody_idxs]
        return keybody_global_ang_vel  # [num_envs, num_keybodies, 3]

    # ------- Root-Relative Robot Bodylink States -------
    @staticmethod
    def _get_obs_root_rel_robot_bodylink_pos(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Positions of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_pos: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_pos(
                env, robot_asset_name, keybody_idxs
            )
        )  # [num_envs, num_keybodies, 3]

        root_global_pos: torch.Tensor = (
            ObservationFunctions._get_obs_robot_root_global_pos(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_robot_root_global_rot_wxyz(env)
        )  # [num_envs, 4]

        # Transform to root frame
        # Position relative to root
        rel_pos_global: torch.Tensor = (
            keybody_global_pos - root_global_pos[..., None, :]
        )  # [num_envs, num_keybodies, 3]

        # Rotate to root frame using inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        rel_pos_root: torch.Tensor = isaaclab_math.quat_apply(
            root_inv_rot[..., None, :], rel_pos_global
        )  # [num_envs, num_keybodies, 3]

        return rel_pos_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_wxyz(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 4]
        """Orientations (w, x, y, z) of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_rot: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_idxs
            )
        )  # [num_envs, num_keybodies, 4]

        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_robot_root_global_rot_wxyz(env)
        )  # [num_envs, 4]

        # Transform to root frame by multiplying with inverse root rotation
        root_inv_rot: torch.Tensor = isaaclab_math.quat_inv(
            root_global_rot_wxyz
        )  # [num_envs, 4]
        rel_rot_root: torch.Tensor = isaaclab_math.quat_mul(
            root_inv_rot[..., None, :], keybody_global_rot, w_last=False
        )  # [num_envs, num_keybodies, 4]

        return rel_rot_root

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_xyzw(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 4]
        """Orientations (x, y, z, w) of specified bodylinks relative to the robot's root frame."""
        return ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
            env, robot_asset_name, keybody_idxs
        )[
            ..., [1, 2, 3, 0]
        ]  # [num_envs, num_keybodies, 4] - convert WXYZ to XYZW

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_rot_mat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 6]
        """Orientations of specified bodylinks relative to the robot's root frame, as a 3x3 matrix, flattened to the first two rows (6D)."""
        keybody_rel_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_root_rel_robot_bodylink_rot_wxyz(
                env, robot_asset_name, keybody_idxs
            )
        )  # [num_envs, num_keybodies, 4]

        return isaaclab_math.quaternion_to_matrix(keybody_rel_rot_wxyz)[
            ..., :2
        ].reshape(env.num_envs, -1)  # [num_envs, num_keybodies * 6]

    @staticmethod
    def _get_obs_root_rel_robot_bodylink_lin_vel(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Linear velocities of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_lin_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_idxs
            )
        )  # [num_envs, num_keybodies, 3]
        root_global_lin_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_lin_vel(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_robot_root_global_rot_wxyz(env)
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
        keybody_idxs: list[int] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies, 3]
        """Angular velocities of specified bodylinks relative to the robot's root frame."""
        # Get global states
        keybody_global_ang_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_idxs
            )
        )  # [num_envs, num_keybodies, 3]
        root_global_ang_vel: torch.Tensor = (
            ObservationFunctions._get_obs_global_robot_root_ang_vel(env)
        )  # [num_envs, 3]
        root_global_rot_wxyz: torch.Tensor = (
            ObservationFunctions._get_obs_robot_root_global_rot_wxyz(env)
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

    # ------- Robot DoF States -------
    @staticmethod
    def _get_obs_dof_pos(env: ManagerBasedRLEnv):
        """Joint positions relative to the default joint angles."""
        return mdp.dof_pos(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_dof_vel(env: ManagerBasedRLEnv):
        """Joint velocities."""
        return mdp.dof_vel(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_last_action(env: ManagerBasedRLEnv):
        """Last action output by the policy."""
        return mdp.last_action(env)  # [num_envs, num_actions]

    # ------- Reference Motion States -------
    @staticmethod
    def _get_obs_ref_motion_states(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str,
    ):
        """Reference motion states.

        This function should return the reference motion command which
        has already been serialized into flattened vectors.
        """
        return mdp.generated_commands(
            env,
            params={"command_name": ref_motion_command_name},
        )  # [num_envs, ref_motion_states_dim]
