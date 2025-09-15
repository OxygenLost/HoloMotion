from dataclasses import MISSING
from typing import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
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
# from isaaclab.utils.math import (
#     quat_error_magnitude,
#     quat_from_euler_xyz,
#     sample_uniform,
# )

import isaaclab.utils.math as isaac_math
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib
from holomotion.src.utils.isaac_utils.rotations import (
    calc_heading_quat_inv,
    get_euler_xyz,
    my_quat_rotate,
    quat_inverse,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_matrix,
    wrap_to_pi,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MISSING
    asset_name: str = MISSING
    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(
        self,
        cfg: MotionCommandCfg,
        env: ManagerBasedRLEnv,
        motion_lib: LmdbMotionLib,
        is_evaluating: bool = False,
    ):
        super().__init__(cfg, env)
        self._env = env
        self._motion_lib = motion_lib
        self._is_evaluating = is_evaluating

        self._init_robot_handle()
        self._init_buffers()

    def _init_robot_handle(self):
        self.robot: Articulation = self._env.scene[self.cfg.asset_name]
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(
                self.cfg.body_names,
                preserve_order=True,
            )[0],
            dtype=torch.long,
            device=self.device,
        )

    def _init_buffers(self):
        self.ref_motion_global_frame_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.body_pos_relative_w = torch.zeros(
            self.num_envs,
            len(self.cfg.body_names),
            3,
            device=self.device,
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs,
            len(self.cfg.body_names),
            4,
            device=self.device,
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

    @property
    def command(
        self,
    ) -> torch.Tensor:
        return self._get_obs_fut_ref_motion().reshape(self.num_envs, -1)

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self.command_counter[env_ids] = 0

        self._resample(env_ids)

        return extras

    def compute(self, dt: float):
        self._update_metrics()

        self.ref_motion_global_frame_ids += 1

        env_ids = torch.where(
            self.ref_motion_global_frame_ids
            >= self.ref_motion_global_end_frame_ids
        )[0]
        self._resample_command(env_ids)

        self._update_ref_motion_state()

    def _update_ref_motion_state(self):
        self.ref_motion_state = self._motion_lib.cache.get_motion_state(
            self.ref_motion_global_frame_ids,
            global_offset=self.env_origins,
            n_fut_frames=self.cfg.n_fut_frames,
            target_fps=self.cfg.target_fps,
        )

    @property
    def ref_motion_dof_pos_fut(self):
        return self.ref_motion_state["dof_pos"]

    @property
    def ref_motion_dof_vel_fut(self):
        return self.ref_motion_state["dof_vel"]

    @property
    def ref_motion_root_global_pos_fut(self):
        return self.ref_motion_state["root_pos"]

    @property
    def ref_motion_root_global_rot_quat_xyzw_fut(self):
        return self.ref_motion_state["root_rot"]

    @property
    def ref_motion_root_global_rot_quat_wxyz_fut(self):
        return self.ref_motion_root_global_rot_quat_xyzw_fut[:, [1, 2, 3, 0]]

    @property
    def ref_motion_root_global_rot_mat_fut(self):
        orginal_shape = self.ref_motion_root_global_rot_quat_xyzw_fut.shape
        return quaternion_to_matrix(
            self.ref_motion_root_global_rot_quat_xyzw_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_root_global_lin_vel_fut(self):
        return self.ref_motion_state["root_vel"]

    @property
    def ref_motion_root_global_ang_vel_fut(self):
        return self.ref_motion_state["root_ang_vel"]

    @property
    def ref_motion_bodylink_global_pos_fut(self):
        return self.ref_motion_state["rg_pos_t"]

    @property
    def ref_motion_bodylink_global_rot_xyzw_fut(self):
        return self.ref_motion_state["rg_rot_t"]

    @property
    def ref_motion_bodylink_global_rot_wxyz_fut(self):
        return self.ref_motion_bodylink_global_rot_xyzw_fut[:, [1, 2, 3, 0]]

    @property
    def ref_motion_bodylink_global_rot_mat_fut(self):
        orginal_shape = self.ref_motion_bodylink_global_rot_xyzw_fut.shape
        return quaternion_to_matrix(
            self.ref_motion_bodylink_global_rot_xyzw_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_bodylink_global_lin_vel_fut(self):
        return self.ref_motion_state["body_vel_t"]

    @property
    def ref_motion_bodylink_global_ang_vel_fut(self):
        return self.ref_motion_state["body_ang_vel_t"]

    @property
    def ref_motion_dof_pos_cur(self):
        return self.ref_motion_dof_pos_fut[:, 0]

    @property
    def ref_motion_dof_vel_cur(self):
        return self.ref_motion_dof_vel_fut[:, 0]

    @property
    def ref_motion_root_global_pos_cur(self):
        return self.ref_motion_root_global_pos_fut[:, 0]

    @property
    def ref_motion_root_global_rot_quat_xyzw_cur(self):
        return self.ref_motion_root_global_rot_quat_xyzw_fut[:, 0]

    @property
    def ref_motion_root_global_rot_quat_wxyz_cur(self):
        return self.ref_motion_root_global_rot_quat_wxyz_fut[:, 0]

    @property
    def ref_motion_root_global_rot_mat_cur(self):
        return self.ref_motion_root_global_rot_mat_fut[:, 0]

    @property
    def ref_motion_root_global_lin_vel_cur(self):
        return self.ref_motion_root_global_lin_vel_fut[:, 0]

    @property
    def ref_motion_root_global_ang_vel_cur(self):
        return self.ref_motion_root_global_ang_vel_fut[:, 0]

    @property
    def ref_motion_bodylink_global_pos_cur(self):
        return self.ref_motion_bodylink_global_pos_fut[:, 0]

    @property
    def ref_motion_bodylink_global_rot_xyzw_cur(self):
        return self.ref_motion_bodylink_global_rot_xyzw_fut[:, 0]

    @property
    def ref_motion_bodylink_global_rot_wxyz_cur(self):
        return self.ref_motion_bodylink_global_rot_wxyz_fut[:, 0]

    @property
    def ref_motion_bodylink_global_rot_mat_cur(self):
        return self.ref_motion_bodylink_global_rot_mat_fut[:, 0]

    @property
    def ref_motion_bodylink_global_lin_vel_cur(self):
        return self.ref_motion_bodylink_global_lin_vel_fut[:, 0]

    @property
    def ref_motion_bodylink_global_ang_vel_cur(self):
        return self.ref_motion_bodylink_global_ang_vel_fut[:, 0]

    @property
    def ref_motion_global_start_frame_ids(self):
        return self._motion_lib.cache.sample_cached_global_start_frames(
            torch.arange(self.cfg.num_envs),
            n_fut_frames=self.cfg.n_fut_frames,
            eval=self._is_evaluating,
        ).to(self.device)

    @property
    def ref_motion_global_end_frame_ids(self):
        return self._motion_lib.cache.cached_motion_global_end_frames.to(
            self.device
        )

    @torch.compile
    def _get_obs_fut_ref_motion(self) -> torch.Tensor:
        num_envs, num_fut_timesteps, num_bodies, _ = (
            self.ref_motion_bodylink_global_pos_fut.shape[2]
        )
        assert num_envs == self.num_envs
        assert num_fut_timesteps == self.cfg.n_fut_frames

        fut_ref_root_rot_quat = (
            self.ref_motion_root_global_rot_quat_xyzw_fut
        )  # [B, T, 4]
        fut_ref_root_rot_quat_inv = quat_inverse(
            fut_ref_root_rot_quat, w_last=True
        )  # [B, T, 4]
        fut_ref_root_rot_quat_body_flat = (
            fut_ref_root_rot_quat[:, :, None, :]
            .repeat(1, 1, num_bodies, 1)
            .reshape(-1, 4)
        )
        fut_ref_root_rot_quat_body_flat_inv = quat_inverse(
            fut_ref_root_rot_quat_body_flat, w_last=True
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            fut_ref_root_rot_quat.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            fut_ref_root_rot_quat.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat(
            [ref_fut_roll, ref_fut_pitch], dim=-1
        )  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(num_envs, -1)  # [B, T * 2]
        # ---

        fut_ref_root_quat_inv_fut_flat = fut_ref_root_rot_quat_inv.reshape(
            -1, 4
        )
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            fut_ref_root_quat_inv_fut_flat,  # [B*T, 4]
            self.ref_motion_root_global_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            fut_ref_root_quat_inv_fut_flat,  # [B*T, 4]
            self.ref_motion_root_global_ang_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_dof_pos_flat = self.ref_motion_dof_pos_fut.reshape(
            num_envs, -1
        )
        fut_ref_dof_vel_flat = self.ref_motion_dof_vel_fut.reshape(
            num_envs, -1
        )
        # ---

        # --- calculate the future per frame bodylink position and rotation ---
        fut_ref_global_bodylink_pos = (
            self.ref_motion_bodylink_global_pos_fut
        )  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_rot = (
            self.ref_motion_bodylink_global_rot_xyzw_fut
        )  # [B, T, num_bodies, 4]

        # get root-relative bodylink position
        fut_ref_root_rel_bodylink_pos = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            (
                fut_ref_global_bodylink_pos
                - fut_ref_global_bodylink_pos[:, :, 0:1, :]
            ).reshape(-1, 3),
            w_last=True,
        ).reshape(
            num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink rotation
        fut_ref_root_rel_bodylink_rot = quat_mul(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )
        fut_ref_root_rel_bodylink_rot_mat = quaternion_to_matrix(
            fut_ref_root_rel_bodylink_rot,
            w_last=True,
        )[:, :, :2].reshape(
            num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 6]

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_dof_pos_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_dof_vel_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_root_rel_bodylink_pos.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_rot_mat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*6]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies * (3 + 6)]
        return rel_fut_ref_motion_state_seq

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        # sample the global start frame ids and start from those frames
        self.ref_motion_global_frame_ids[env_ids] = (
            self._motion_lib.cache.sample_cached_global_start_frames(
                env_ids,
                n_fut_frames=self.cfg.n_fut_frames,
                eval=self._is_evaluating,
            ).to(self.device)
        )

        self._align_root_to_ref(env_ids)
        self._align_dof_to_ref(env_ids)

    def _align_root_to_ref(self, env_ids):
        root_pos = self.ref_motion_root_global_pos_cur.clone()
        root_rot = self.ref_motion_root_global_rot_quat_wxyz_cur.clone()
        root_lin_vel = self.ref_motion_root_global_lin_vel_cur.clone()
        root_ang_vel = self.ref_motion_root_global_ang_vel_cur.clone()

        pos_rot_range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        pos_rot_ranges = torch.tensor(pos_rot_range_list, device=self.device)
        pos_rot_rand_deltas = isaac_math.sample_uniform(
            pos_rot_ranges[:, 0],
            pos_rot_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        translation_delta = pos_rot_rand_deltas[:, 0:3]
        rotation_delta = isaac_math.quat_from_euler_xyz(
            pos_rot_rand_deltas[:, 3],
            pos_rot_rand_deltas[:, 4],
            pos_rot_rand_deltas[:, 5],
        )

        root_pos[env_ids] += translation_delta
        root_rot[env_ids] = isaac_math.quat_mul(
            rotation_delta,
            root_rot[env_ids],
        )

        lin_ang_vel_range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        lin_ang_vel_ranges = torch.tensor(
            lin_ang_vel_range_list, device=self.device
        )

        lin_ang_vel_rand_deltas = isaac_math.sample_uniform(
            lin_ang_vel_ranges[:, 0],
            lin_ang_vel_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_lin_vel[env_ids] += lin_ang_vel_rand_deltas[:, :3]
        root_ang_vel[env_ids] += lin_ang_vel_rand_deltas[:, 3:]

        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_rot[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

    def _align_dof_to_ref(self, env_ids):
        dof_pos = self.ref_motion_dof_pos_cur.clone()
        dof_vel = self.ref_motion_dof_vel_cur.clone()

        dof_pos += isaac_math.sample_uniform(
            *self.cfg.joint_position_range,
            dof_pos.shape,
            dof_pos.device,
        )
        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos[env_ids] = torch.clip(
            dof_pos[env_ids],
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )

        dof_vel += isaac_math.sample_uniform(
            *self.cfg.joint_velocity_range,
            dof_vel.shape,
            dof_vel.device,
        )
        soft_dof_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids]
        dof_vel[env_ids] = torch.clip(
            dof_vel[env_ids],
            soft_dof_vel_limits[:, :, 0],
            soft_dof_vel_limits[:, :, 1],
        )

        self.robot.write_joint_state_to_sim(
            dof_pos[env_ids],
            dof_vel[env_ids],
            env_ids=env_ids,
        )

    def _update_command(self):
        self.ref_motion_global_frame_ids += 1
        env_ids = torch.where(
            self.ref_motion_global_frame_ids
            >= self.ref_motion_global_end_frame_ids
        )[0]
        self._resample_command(env_ids)

    # def _update_metrics(self):
    #     self.metrics["error_anchor_pos"] = torch.norm(
    #         self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    #     )
    #     self.metrics["error_anchor_rot"] = quat_error_magnitude(
    #         self.anchor_quat_w, self.robot_anchor_quat_w
    #     )
    #     self.metrics["error_anchor_lin_vel"] = torch.norm(
    #         self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    #     )
    #     self.metrics["error_anchor_ang_vel"] = torch.norm(
    #         self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    #     )

    #     self.metrics["error_body_pos"] = torch.norm(
    #         self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    #     ).mean(dim=-1)
    #     self.metrics["error_body_rot"] = quat_error_magnitude(
    #         self.body_quat_relative_w, self.robot_body_quat_w
    #     ).mean(dim=-1)

    #     self.metrics["error_body_lin_vel"] = torch.norm(
    #         self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    #     ).mean(dim=-1)
    #     self.metrics["error_body_ang_vel"] = torch.norm(
    #         self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    #     ).mean(dim=-1)

    #     self.metrics["error_joint_pos"] = torch.norm(
    #         self.joint_pos - self.robot_joint_pos, dim=-1
    #     )
    #     self.metrics["error_joint_vel"] = torch.norm(
    #         self.joint_vel - self.robot_joint_vel, dim=-1
    #     )

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     if debug_vis:
    #         if not hasattr(self, "current_anchor_visualizer"):
    #             self.current_anchor_visualizer = VisualizationMarkers(
    #                 self.cfg.anchor_visualizer_cfg.replace(
    #                     prim_path="/Visuals/Command/current/anchor"
    #                 )
    #             )
    #             self.goal_anchor_visualizer = VisualizationMarkers(
    #                 self.cfg.anchor_visualizer_cfg.replace(
    #                     prim_path="/Visuals/Command/goal/anchor"
    #                 )
    #             )

    #             self.current_body_visualizers = []
    #             self.goal_body_visualizers = []
    #             for name in self.cfg.body_names:
    #                 self.current_body_visualizers.append(
    #                     VisualizationMarkers(
    #                         self.cfg.body_visualizer_cfg.replace(
    #                             prim_path="/Visuals/Command/current/"
    #                             + name
    #                         )
    #                     )
    #                 )
    #                 self.goal_body_visualizers.append(
    #                     VisualizationMarkers(
    #                         self.cfg.body_visualizer_cfg.replace(
    #                             prim_path="/Visuals/Command/goal/"
    #                             + name
    #                         )
    #                     )
    #                 )

    #         self.current_anchor_visualizer.set_visibility(True)
    #         self.goal_anchor_visualizer.set_visibility(True)
    #         for i in range(len(self.cfg.body_names)):
    #             self.current_body_visualizers[i].set_visibility(True)
    #             self.goal_body_visualizers[i].set_visibility(True)

    #     else:
    #         if hasattr(self, "current_anchor_visualizer"):
    #             self.current_anchor_visualizer.set_visibility(False)
    #             self.goal_anchor_visualizer.set_visibility(False)
    #             for i in range(len(self.cfg.body_names)):
    #                 self.current_body_visualizers[i].set_visibility(
    #                     False
    #                 )
    #                 self.goal_body_visualizers[i].set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     if not self.robot.is_initialized:
    #         return

    #     self.current_anchor_visualizer.visualize(
    #         self.robot_anchor_pos_w, self.robot_anchor_quat_w
    #     )
    #     self.goal_anchor_visualizer.visualize(
    #         self.anchor_pos_w, self.anchor_quat_w
    #     )

    #     for i in range(len(self.cfg.body_names)):
    #         self.current_body_visualizers[i].visualize(
    #             self.robot_body_pos_w[:, i],
    #             self.robot_body_quat_w[:, i],
    #         )
    #         self.goal_body_visualizers[i].visualize(
    #             self.body_pos_relative_w[:, i],
    #             self.body_quat_relative_w[:, i],
    #         )


@configclass
class CommandsCfg:
    motion = MotionCommandCfg(
        class_type=MotionCommand,
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        joint_position_range=(-0.1, 0.1),
        joint_velocity_range=(-1.0, 1.0),
    )
