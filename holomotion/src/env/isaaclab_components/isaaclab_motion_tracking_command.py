from dataclasses import MISSING
from typing import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils


import isaaclab.utils.math as isaaclab_math
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
    TerminationTermCfg,
)
from isaaclab.markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)

from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omegaconf import OmegaConf

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
from loguru import logger


class RefMotionCommand(CommandTerm):
    cfg: CommandTermCfg

    def __init__(
        self,
        cfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)
        self._env = env
        self._is_evaluating = self.cfg.is_evaluating

        self._init_robot_handle()
        self._init_buffers()
        self._init_motion_lib()

    def _init_motion_lib(self):
        self._motion_lib = LmdbMotionLib(
            motion_lib_cfg=OmegaConf.create(self.cfg.motion_lib_cfg),
            cache_device=self.device,
            process_id=self.cfg.process_id,
            num_processes=self.cfg.num_processes,
        )
        self._init_per_env_cache()
        # Initialize motion state after setting up cache
        self._update_ref_motion_state()

    def _init_per_env_cache(self):
        """Initialize per-environment cache with individual motion IDs."""
        # Sample motion IDs for all environments at once
        sampled_motion_ids = self._motion_lib.resample_new_motions(
            self.num_envs, eval=self._is_evaluating
        )
        self._cached_motion_ids[:] = sampled_motion_ids.to(self.device)

        # Initialize start frames
        if self._is_evaluating:
            # Deterministic: start at cached start
            self.ref_motion_global_frame_ids[:] = (
                self.ref_motion_global_start_frame_ids
            )
        else:
            # Training: uniform time sampling within each cached window
            all_env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._uniform_sample_ref_start_frames(all_env_ids)

    def _init_robot_handle(self):
        self.robot: Articulation = self._env.scene[self.cfg.asset_name]
        self.anchor_bodylink_name = self.cfg.anchor_bodylink_name
        self.anchor_bodylink_idx = self.robot.body_names.index(
            self.anchor_bodylink_name
        )
        self.urdf_dof_names = self.cfg.urdf_dof_names
        self.urdf_body_names = self.cfg.urdf_body_names
        self.simulator_dof_names = self.robot.joint_names
        self.simulator_body_names = self.robot.body_names
        # the two dof orders are different, ref motion follows urdf, simulator follows simulator_dof_names
        self.simulator2urdf_dof_idx = [
            self.urdf_dof_names.index(dof) for dof in self.simulator_dof_names
        ]
        self.simulator2urdf_body_idx = [
            self.urdf_body_names.index(body)
            for body in self.simulator_body_names
        ]

        # Create index mappings for metrics computation using unified naming
        # DOF indices for mpjpe metrics
        self.arm_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.arm_dof_names
        ]
        self.torso_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.torso_dof_names
        ]
        self.leg_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.leg_dof_names
        ]

        # Body indices for mpkpe metrics using unified naming
        self.arm_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.arm_body_names
        ]
        self.torso_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.torso_body_names
        ]
        self.leg_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.leg_body_names
        ]

    def _init_buffers(self):
        self.ref_motion_global_frame_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # mark envs that timed out (frame id exceeded end frame) in current step
        self._motion_end_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        # counter for number of motion ends per environment
        self.motion_end_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # per-environment cached motion indices
        self._cached_motion_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # env -> cache row indirection (starts as identity mapping)
        self._env_to_cache_row = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _init_single_motion(self):
        single_id = self._motion_lib.sample_motion_ids_only(1, eval=True)
        motion_key = self._motion_lib.motion_id2key[int(single_id[0].item())]
        self.single_ref_motion = self._motion_lib.export_motion_clip(
            motion_key
        )

    @property
    def command(
        self,
    ) -> torch.Tensor:
        # call the corresponding method based on configured command_obs_name
        return getattr(self, f"_get_obs_{self.cfg.command_obs_name}")()

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:
        # Call parent class reset to handle time_left and command_counter properly
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device)
        self._motion_end_mask[env_ids] = False
        self.motion_end_counter[env_ids] = 0

        # Align behavior with commands.py: perform resampling via unified entrypoint
        self._resample_command(env_ids, eval=self._is_evaluating)

        return extras

    def compute(self, dt: float):
        self._update_metrics()

        self._update_command()

    def _update_ref_motion_state(self):
        # Use env->cache row indirection so resample can be O(1) remap
        self.ref_motion_state = self._motion_lib.cache.get_motion_state(
            self.ref_motion_global_frame_ids,
            global_offset=self._env.scene.env_origins,
            n_fut_frames=self.cfg.n_fut_frames,
            target_fps=self.cfg.target_fps,
            row_indices=self._env_to_cache_row,
        )

    def _uniform_sample_ref_start_frames(self, env_ids: torch.Tensor):
        """Uniformly sample start frames within cached windows for env_ids.

        Sampling range is [start, end - 1 - n_fut_frames] to ensure required
        future frames exist. If that upper bound is < start, it falls back to start.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device).long()

        starts = self.ref_motion_global_start_frame_ids[env_ids]
        ends = self.ref_motion_global_end_frame_ids[env_ids]

        # Ensure room for future frames if requested
        n_fut = (
            int(self.cfg.n_fut_frames)
            if hasattr(self.cfg, "n_fut_frames")
            else 0
        )
        max_start = ends - 1 - n_fut
        max_start = torch.maximum(max_start, starts)

        num_choices = (max_start - starts + 1).clamp(min=1)
        # Sample offsets uniformly
        rand = torch.rand_like(starts, dtype=torch.float32)
        offsets = torch.floor(rand * num_choices.float()).long()
        sampled = starts + offsets

        self.ref_motion_global_frame_ids[env_ids] = sampled

    @property
    def ref_motion_dof_pos_fut(self):
        return self.ref_motion_state["dof_pos"][:, 1:, ...][
            ..., self.simulator2urdf_dof_idx
        ]

    @property
    def ref_motion_dof_vel_fut(self):
        return self.ref_motion_state["dof_vel"][:, 1:, ...][
            ..., self.simulator2urdf_dof_idx
        ]

    @property
    def ref_motion_root_global_pos_fut(self):
        return self.ref_motion_state["root_pos"][:, 1:, ...]

    @property
    def ref_motion_root_global_rot_quat_xyzw_fut(self):
        return self.ref_motion_state["root_rot"][:, 1:, ...]

    @property
    def ref_motion_root_global_rot_quat_wxyz_fut(self):
        return self.ref_motion_root_global_rot_quat_xyzw_fut[..., [3, 0, 1, 2]]

    @property
    def ref_motion_root_global_rot_mat_fut(self):
        orginal_shape = self.ref_motion_root_global_rot_quat_xyzw_fut.shape
        return quaternion_to_matrix(
            self.ref_motion_root_global_rot_quat_xyzw_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_root_global_lin_vel_fut(self):
        return self.ref_motion_state["root_vel"][:, 1:, ...]

    @property
    def ref_motion_root_global_ang_vel_fut(self):
        return self.ref_motion_state["root_ang_vel"][:, 1:, ...]

    @property
    def ref_motion_bodylink_global_pos_fut(self):
        return self.ref_motion_state["rg_pos"][:, 1:, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_xyzw_fut(self):
        return self.ref_motion_state["rb_rot"][:, 1:, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_wxyz_fut(self):
        return self.ref_motion_bodylink_global_rot_xyzw_fut[..., [3, 0, 1, 2]]

    @property
    def ref_motion_bodylink_global_rot_mat_fut(self):
        orginal_shape = self.ref_motion_bodylink_global_rot_xyzw_fut.shape
        return quaternion_to_matrix(
            self.ref_motion_bodylink_global_rot_xyzw_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_bodylink_global_lin_vel_fut(self):
        return self.ref_motion_state["body_vel"][:, 1:, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_ang_vel_fut(self):
        return self.ref_motion_state["body_ang_vel"][:, 1:, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_dof_pos_cur(self):
        return self.ref_motion_state["dof_pos"][:, 0, ...][
            ..., self.simulator2urdf_dof_idx
        ]

    @property
    def ref_motion_dof_vel_cur(self):
        return self.ref_motion_state["dof_vel"][:, 0, ...][
            ..., self.simulator2urdf_dof_idx
        ]

    @property
    def ref_motion_root_global_pos_cur(self):
        return self.ref_motion_state["root_pos"][:, 0, ...]

    @property
    def ref_motion_root_global_rot_quat_xyzw_cur(self):
        return self.ref_motion_state["root_rot"][:, 0, ...]

    @property
    def ref_motion_root_global_rot_quat_wxyz_cur(self):
        return self.ref_motion_root_global_rot_quat_xyzw_cur[..., [3, 0, 1, 2]]

    @property
    def ref_motion_root_global_rot_mat_cur(self):
        orginal_shape = self.ref_motion_root_global_rot_quat_xyzw_cur.shape
        return quaternion_to_matrix(
            self.ref_motion_root_global_rot_quat_xyzw_cur.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_root_global_lin_vel_cur(self):
        return self.ref_motion_state["root_vel"][:, 0, ...]

    @property
    def ref_motion_root_global_ang_vel_cur(self):
        return self.ref_motion_state["root_ang_vel"][:, 0, ...]

    @property
    def ref_motion_bodylink_global_pos_cur(self):
        return self.ref_motion_state["rg_pos"][:, 0, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_xyzw_cur(self):
        return self.ref_motion_state["rb_rot"][:, 0, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_wxyz_cur(self):
        return self.ref_motion_bodylink_global_rot_xyzw_cur[..., [3, 0, 1, 2]]

    @property
    def ref_motion_bodylink_global_rot_mat_cur(self):
        orginal_shape = self.ref_motion_bodylink_global_rot_xyzw_cur.shape
        return quaternion_to_matrix(
            self.ref_motion_bodylink_global_rot_xyzw_cur.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_bodylink_global_lin_vel_cur(self):
        return self.ref_motion_state["body_vel"][:, 0, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_ang_vel_cur(self):
        return self.ref_motion_state["body_ang_vel"][:, 0, ...][
            ..., self.simulator2urdf_body_idx, :
        ]

    @property
    def ref_motion_global_start_frame_ids(self):
        frames = self._motion_lib.cache.cached_motion_global_start_frames.to(
            self.device
        )
        return frames[self._env_to_cache_row]

    @property
    def ref_motion_global_end_frame_ids(self):
        frames = self._motion_lib.cache.cached_motion_global_end_frames.to(
            self.device
        )
        return frames[self._env_to_cache_row]

    @property
    def motion_end_mask(self) -> torch.Tensor:
        """[B] bool: per-step timeout mask.

        Uses the per-step `motion_end_mask` set before resampling so the
        event is observable within the same step, and falls back to a
        direct comparison if not available.
        """
        return self._motion_end_mask

    @property
    def ref_motion_anchor_bodylink_global_pos_cur(self):
        return self.ref_motion_bodylink_global_pos_cur[
            :, self.anchor_bodylink_idx
        ]

    @property
    def ref_motion_anchor_bodylink_global_rot_cur_wxyz(self):
        return self.ref_motion_bodylink_global_rot_wxyz_cur[
            :, self.anchor_bodylink_idx
        ]

    @property
    def global_robot_anchor_pos_cur(self):
        return self.robot.data.body_pos_w[:, self.anchor_bodylink_idx]

    @torch.compile
    def _get_obs_bydmmc_ref_motion(self) -> torch.Tensor:
        assert self.cfg.n_fut_frames == 1, (
            "Only support n_fut_frames = 1 for bydmmc ref motion"
        )
        num_envs = self.ref_motion_dof_pos_cur.shape[0]
        fut_ref_dof_pos_flat = self.ref_motion_dof_pos_cur.reshape(
            num_envs, -1
        )
        fut_ref_dof_vel_flat = self.ref_motion_dof_vel_cur.reshape(
            num_envs, -1
        )
        return torch.cat([fut_ref_dof_pos_flat, fut_ref_dof_vel_flat], dim=-1)

    @torch.compile
    def _get_obs_holomotion_rel_ref_motion_flat(self) -> torch.Tensor:
        num_envs, num_fut_timesteps, num_bodies, _ = (
            self.ref_motion_bodylink_global_pos_fut.shape
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
        return rel_fut_ref_motion_state_seq.reshape(self.num_envs, -1)

    def _resample_command(self, env_ids: Sequence[int], eval=False):
        """Resample command for specified environments.

        This method is called by the parent class when commands need to be resampled.
        It should handle both timeout-based resampling and explicit resampling requests.
        Training uses uniform time sampling within the cached window; evaluation starts at the cached start frame.
        """
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if eval or self._is_evaluating:
            # Deterministic: start at cached start
            self.ref_motion_global_frame_ids[env_ids] = (
                self.ref_motion_global_start_frame_ids[env_ids]
            )
        else:
            # Uniform time sampling within [start, end)
            self._uniform_sample_ref_start_frames(env_ids)
        # Update state and place robot to the resampled reference, mirroring commands.py
        self._update_ref_motion_state()
        self._align_root_to_ref(env_ids)
        self._align_dof_to_ref(env_ids)

    def _resample_when_timeout(self):
        env_ids = torch.where(
            self.ref_motion_global_frame_ids
            >= self.ref_motion_global_end_frame_ids
        )[0]

        if env_ids.numel() > 0:
            self._resample_command(env_ids, eval=self._is_evaluating)

    def _resample_per_env_cache(self, env_ids: torch.Tensor, eval=False):
        """Resample cached motions for specific environments.

        This method samples new motions for the specified environments and updates
        the motion library cache to include the new motions.

        Args:
            env_ids: Tensor of environment IDs to resample cache for
            eval: Whether to use evaluation mode for sampling
        """
        if env_ids.numel() == 0:
            return

        # Minimal refactor: avoid any data copy. Remap envs to existing cache rows.
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device).long()

        # Sample new cache rows uniformly from existing rows [0, num_envs)
        new_rows = torch.randint(
            low=0,
            high=self.num_envs,
            size=(env_ids.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        # Optionally avoid mapping to the same row
        same_mask = new_rows == self._env_to_cache_row[env_ids]
        if same_mask.any():
            alt_rows = (new_rows[same_mask] + 1) % self.num_envs
            new_rows = new_rows.clone()
            new_rows[same_mask] = alt_rows

        # Apply remap
        self._env_to_cache_row[env_ids] = new_rows

    def _rebuild_cache_with_current_motion_ids(self):
        """Rebuild the motion cache with current motion IDs.

        This method registers the current motion IDs and rebuilds the cache.
        Since we always start from frame 0, all start frames are 0.
        """
        # Register current motion IDs in cache
        self._motion_lib.cache.register_motion_ids(
            self._cached_motion_ids.cpu()
        )

        # Create start frames (all zeros since we always start from frame 0)
        start_frames = torch.zeros(self.num_envs, dtype=torch.long)

        # Rebuild cache with new motion IDs
        self._motion_lib._build_online_train_cache(
            self._motion_lib.cache,
            start_frames,
        )

    def _align_root_to_ref(self, env_ids):
        root_pos = self.ref_motion_root_global_pos_cur.clone()
        root_rot = self.ref_motion_root_global_rot_quat_wxyz_cur.clone()
        root_lin_vel = self.ref_motion_root_global_lin_vel_cur.clone()
        root_ang_vel = self.ref_motion_root_global_ang_vel_cur.clone()

        pos_rot_range_list = [
            self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        pos_rot_ranges = torch.tensor(pos_rot_range_list, device=self.device)
        pos_rot_rand_deltas = isaaclab_math.sample_uniform(
            pos_rot_ranges[:, 0],
            pos_rot_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        translation_delta = pos_rot_rand_deltas[:, 0:3]
        rotation_delta = isaaclab_math.quat_from_euler_xyz(
            pos_rot_rand_deltas[:, 3],
            pos_rot_rand_deltas[:, 4],
            pos_rot_rand_deltas[:, 5],
        )

        root_pos[env_ids] += translation_delta
        root_rot[env_ids] = isaaclab_math.quat_mul(
            rotation_delta,
            root_rot[env_ids],
        )

        lin_ang_vel_range_list = [
            self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        lin_ang_vel_ranges = torch.tensor(
            lin_ang_vel_range_list, device=self.device
        )

        lin_ang_vel_rand_deltas = isaaclab_math.sample_uniform(
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

        dof_pos += isaaclab_math.sample_uniform(
            *self.cfg.dof_pos_perturb_range,
            dof_pos.shape,
            dof_pos.device,
        )
        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos[env_ids] = torch.clip(
            dof_pos[env_ids],
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )

        self.robot.write_joint_state_to_sim(
            dof_pos[env_ids],
            dof_vel[env_ids],
            env_ids=env_ids,
        )

    def _update_command(self):
        # advance frame ids and update timeout/resampling
        self.ref_motion_global_frame_ids += 1
        self._resample_when_timeout()
        self._update_ref_motion_state()

    def _update_metrics(self):
        """Update metrics for command progress tracking."""
        if not hasattr(self, "metrics"):
            self.metrics = {}

        self._update_motion_progress_metrics()
        self._update_mpjpe_metrics()
        self._update_mpkpe_metrics()

    def _update_motion_progress_metrics(self):
        # Track motion progress as percentage
        motion_progress = (
            self.ref_motion_global_frame_ids
            - self.ref_motion_global_start_frame_ids
        ).float() / (
            self.ref_motion_global_end_frame_ids
            - self.ref_motion_global_start_frame_ids
        ).float()
        motion_progress = torch.clamp(motion_progress, 0.0, 1.0)

        if "Task/Motion_Progress" not in self.metrics:
            self.metrics["Task/Motion_Progress"] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["Task/Motion_Progress"][:] = motion_progress

    def _update_mpjpe_metrics(self):
        """Update MPJPE (Mean Per Joint Position Error) metrics."""
        # Get current and reference joint positions
        current_dof_pos = self.robot.data.joint_pos  # [B, num_dofs]
        ref_dof_pos = self.ref_motion_dof_pos_cur  # [B, num_dofs]

        # Compute joint position errors
        dof_pos_error = torch.abs(
            current_dof_pos - ref_dof_pos
        )  # [B, num_dofs]

        # MPJPE whole body
        mpjpe_whole = torch.mean(dof_pos_error, dim=-1)  # [B]

        # MPJPE arms (using unified naming)
        mpjpe_arms = torch.mean(
            dof_pos_error[:, self.arm_dof_indices], dim=-1
        )  # [B]

        # MPJPE torso (using unified naming)
        mpjpe_torso = torch.mean(
            dof_pos_error[:, self.torso_dof_indices], dim=-1
        )  # [B]

        # MPJPE legs
        mpjpe_legs = torch.mean(
            dof_pos_error[:, self.leg_dof_indices], dim=-1
        )  # [B]

        # Initialize metric tensors if needed
        for metric_name in [
            "Task/MPJPE_Whole",
            "Task/MPJPE_Arms",
            "Task/MPJPE_Torso",
            "Task/MPJPE_Legs",
        ]:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # Update metric values
        self.metrics["Task/MPJPE_Whole"][:] = mpjpe_whole
        self.metrics["Task/MPJPE_Arms"][:] = mpjpe_arms
        self.metrics["Task/MPJPE_Torso"][:] = mpjpe_torso
        self.metrics["Task/MPJPE_Legs"][:] = mpjpe_legs

    def _update_mpkpe_metrics(self):
        """Update MPKPE (Mean Per Keybody Position Error) metrics."""
        # Get current and reference body positions
        current_body_pos = self.robot.data.body_pos_w  # [B, num_bodies, 3]
        ref_body_pos = (
            self.ref_motion_bodylink_global_pos_cur
        )  # [B, num_bodies, 3]

        # Compute body position errors (L2 norm)
        body_pos_error = torch.norm(
            current_body_pos - ref_body_pos, dim=-1
        )  # [B, num_bodies]

        # MPKPE whole body
        mpkpe_whole = torch.mean(body_pos_error, dim=-1)  # [B]

        # MPKPE arms (using unified naming)
        mpkpe_arms = torch.mean(
            body_pos_error[:, self.arm_body_indices], dim=-1
        )  # [B]

        # MPKPE torso (using unified naming)
        mpkpe_torso = torch.mean(
            body_pos_error[:, self.torso_body_indices], dim=-1
        )  # [B]

        # MPKPE legs
        mpkpe_legs = torch.mean(
            body_pos_error[:, self.leg_body_indices], dim=-1
        )  # [B]

        # Initialize metric tensors if needed
        for metric_name in [
            "Task/MPKPE_Whole",
            "Task/MPKPE_Arms",
            "Task/MPKPE_Torso",
            "Task/MPKPE_Legs",
        ]:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # Update metric values
        self.metrics["Task/MPKPE_Whole"][:] = mpkpe_whole
        self.metrics["Task/MPKPE_Arms"][:] = mpkpe_arms
        self.metrics["Task/MPKPE_Torso"][:] = mpkpe_torso
        self.metrics["Task/MPKPE_Legs"][:] = mpkpe_legs

    def export_motion_data_for_onnx(self) -> dict:
        """Export motion data for ONNX export with proper order mappings.

        This method handles the order conversion from URDF order (motion library)
        to simulator order (IsaacLab expected) for both DOFs and bodies.
        Automatically determines motion length and key bodies from config.

        Returns:
            Dictionary containing motion data in simulator order with shapes:
            - joint_pos: [T, num_dofs] in simulator DOF order
            - joint_vel: [T, num_dofs] in simulator DOF order
            - body_pos_w: [T, num_key_bodies, 3] in simulator body order
            - body_quat_w: [T, num_key_bodies, 4] in simulator body order
            - body_lin_vel_w: [T, num_key_bodies, 3] in simulator body order
            - body_ang_vel_w: [T, num_key_bodies, 3] in simulator body order
        """

        # Always export the full original clip from LMDB starting at frame 0
        # Determine motion key for the first cached row
        if getattr(self._motion_lib, "use_sub_motion_indexing", False):
            sub_id = int(self._motion_lib.cache.cached_motion_ids[0].item())
            motion_key = self._motion_lib.sub_motion_infos[sub_id][
                "original_motion_key"
            ]
        else:
            motion_id = int(self._motion_lib.cache.cached_motion_ids[0].item())
            motion_key = self._motion_lib.motion_id2key[motion_id]

        raw = self._motion_lib.export_motion_clip(motion_key)

        # Map URDF-order arrays to simulator order
        dof_pos_urdf = torch.from_numpy(raw["dof_pos"])  # [T, num_dofs_urdf]
        dof_vel_urdf = torch.from_numpy(raw["dof_vel"])  # [T, num_dofs_urdf]
        dof_pos_simulator = dof_pos_urdf[:, self.simulator2urdf_dof_idx]
        dof_vel_simulator = dof_vel_urdf[:, self.simulator2urdf_dof_idx]

        body_pos_urdf = torch.from_numpy(
            raw["rg_pos"]
        )  # [T, num_bodies_urdf, 3]
        body_quat_urdf = torch.from_numpy(
            raw["rb_rot"]
        )  # [T, num_bodies_urdf, 4]
        body_lin_vel_urdf = torch.from_numpy(
            raw["body_vel"]
        )  # [T, num_bodies_urdf, 3]
        body_ang_vel_urdf = torch.from_numpy(
            raw["body_ang_vel"]
        )  # [T, num_bodies_urdf, 3]

        body_pos_simulator = body_pos_urdf[:, self.simulator2urdf_body_idx]
        body_quat_simulator = body_quat_urdf[:, self.simulator2urdf_body_idx]
        body_lin_vel_simulator = body_lin_vel_urdf[
            :, self.simulator2urdf_body_idx
        ]
        body_ang_vel_simulator = body_ang_vel_urdf[
            :, self.simulator2urdf_body_idx
        ]

        key_bodies = self.cfg.motion_lib_cfg["key_bodies"]
        key_body_indices_simulator = [
            self.simulator_body_names.index(body) for body in key_bodies
        ]

        key_body_pos = body_pos_simulator[:, key_body_indices_simulator]
        key_body_quat = body_quat_simulator[:, key_body_indices_simulator]
        key_body_lin_vel = body_lin_vel_simulator[
            :, key_body_indices_simulator
        ]
        key_body_ang_vel = body_ang_vel_simulator[
            :, key_body_indices_simulator
        ]

        motion_data = {
            "joint_pos": dof_pos_simulator.cpu(),
            "joint_vel": dof_vel_simulator.cpu(),
            "body_pos_w": key_body_pos.cpu(),
            "body_quat_w": key_body_quat.cpu(),
            "body_lin_vel_w": key_body_lin_vel.cpu(),
            "body_ang_vel_w": key_body_ang_vel.cpu(),
        }

        logger.info("Motion data exported with shapes:")
        for key, tensor in motion_data.items():
            logger.info(f"  {key}: {tensor.shape}")

        return motion_data

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Just enable debug mode - visualizers will be created lazily in callback
            self._debug_vis_enabled = True
            # Set visibility if visualizers already exist
            if hasattr(self, "ref_body_visualizers"):
                for visualizer in self.ref_body_visualizers:
                    visualizer.set_visibility(True)
        else:
            self._debug_vis_enabled = False
            # Set visibility to false
            if hasattr(self, "ref_body_visualizers"):
                for visualizer in self.ref_body_visualizers:
                    visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # Check if debug visualization is enabled
        if not getattr(self, "_debug_vis_enabled", False):
            return

        # Check if reference motion state is available
        if (
            not hasattr(self, "ref_motion_state")
            or self.ref_motion_state is None
        ):
            return

        # Create visualizers lazily if they don't exist
        if not hasattr(self, "ref_body_visualizers"):
            self.ref_body_visualizers = []
            # Get number of bodies from the reference motion data
            num_bodies = self.ref_motion_bodylink_global_pos_cur.shape[-2]
            for i in range(num_bodies):
                # Reference bodylinks as red spheres
                self.ref_body_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.body_keypoint_visualizer_cfg.replace(
                            prim_path=f"/Visuals/Command/ref_body_{i}"
                        )
                    )
                )

        # Visualize reference body keypoints
        if len(self.ref_body_visualizers) > 0:
            ref_body_pos = (
                self.ref_motion_bodylink_global_pos_cur
            )  # [B, num_bodies, 3]

            num_bodies = min(
                len(self.ref_body_visualizers), ref_body_pos.shape[1]
            )

            for i in range(num_bodies):
                # Visualize reference bodylinks as spheres (position only)
                self.ref_body_visualizers[i].visualize(
                    ref_body_pos[:, i],  # [B, 3]
                )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = RefMotionCommand

    command_obs_name: str = MISSING
    urdf_dof_names: list[str] = MISSING
    urdf_body_names: list[str] = MISSING

    # DOF name groupings for mpjpe metrics (using unified naming)
    arm_dof_names: list[str] = MISSING
    torso_dof_names: list[str] = MISSING
    leg_dof_names: list[str] = MISSING

    # Body name groupings for mpkpe metrics (using unified naming)
    arm_body_names: list[str] = MISSING
    torso_body_names: list[str] = MISSING
    leg_body_names: list[str] = MISSING

    motion_lib_cfg: dict = MISSING
    process_id: int = MISSING
    num_processes: int = MISSING
    is_evaluating: bool = MISSING
    resample_time_interval_s: float = MISSING

    n_fut_frames: int = MISSING
    target_fps: int = MISSING

    anchor_bodylink_name: str = "pelvis"

    asset_name: str = MISSING
    debug_vis: bool = False

    root_pose_perturb_range: dict[str, tuple[float, float]] = {}
    root_vel_perturb_range: dict[str, tuple[float, float]] = {}
    dof_pos_perturb_range: tuple[float, float] = (-0.1, 0.1)
    dof_vel_perturb_range: tuple[float, float] = (-1.0, 1.0)

    body_keypoint_visualizer_cfg: VisualizationMarkersCfg = (
        SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/ref_keypoint")
    )
    body_keypoint_visualizer_cfg.markers["sphere"].radius = 0.03
    body_keypoint_visualizer_cfg.markers[
        "sphere"
    ].visual_material = PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.0, 1.0)  # blue
    )

    resampling_time_range: tuple[float, float] = (1.0, 1.0)


@configclass
class CommandsCfg:
    pass


def build_commands_config(command_config_dict: dict):
    """Build isaaclab-compatible CommandsCfg from a config dictionary.

    Args:
        command_config_dict: Dictionary mapping command names to command configurations.
                           Each command config should contain the type and parameters.

    Example:
        command_config_dict = {
            "ref_motion": {
                "type": "MotionCommandCfg",
                "params": {
                    "command_obs_name": "bydmmc_ref_motion",
                    "motion_lib_cfg": {...},
                    "process_id": 0,
                    "num_processes": 1,
                    # ... other parameters
                }
            }
        }
    """

    commands_cfg = CommandsCfg()

    # Add command terms dynamically
    for command_name, command_config in command_config_dict.items():
        command_type = command_config.get("type", "MotionCommandCfg")
        command_params = command_config.get("params", {})

        # Get the command class type
        if command_type == "MotionCommandCfg":
            command_cfg = MotionCommandCfg(**command_params)
        else:
            raise ValueError(f"Unknown command type: {command_type}")

        # Add command to config
        setattr(commands_cfg, command_name, command_cfg)

    return commands_cfg
