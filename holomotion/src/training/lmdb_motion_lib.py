# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import pickle
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import lmdb
import numpy as np
import torch
from loguru import logger
from rich.progress import track

from holomotion.src.utils.isaac_utils.rotations import (
    calc_heading_quat_inv,
    my_quat_rotate,
)


def read_motion_array(env, motion_key, array_name, slices=None):
    with env.begin() as txn:
        # Get array data
        array_data = txn.get(f"motion/{motion_key}/{array_name}".encode())

        if array_data is None:
            return None

        # Get shape and dtype
        shape = pickle.loads(
            txn.get(f"motion/{motion_key}/{array_name}_shape".encode())
        )
        dtype = pickle.loads(
            txn.get(f"motion/{motion_key}/{array_name}_dtype".encode())
        )
        # Convert to numpy array
        array = np.frombuffer(array_data, dtype=dtype).reshape(shape)
        # Apply slicing if requested
        if slices is not None:
            return array[slices]
        return array


@dataclass
class OnlineMotionCache:
    """Dataclass for storing cached motion data for training/evaluation."""

    device: torch.device
    num_envs: int

    max_frame_length: int
    n_fut_frames: int
    num_bodies: int
    num_dofs: int
    num_extended_bodies: int
    key_body_indices: List[int]
    fps: float

    cached_motion_ids: Optional[torch.Tensor] = None
    cached_motion_raw_num_frames: Optional[torch.Tensor] = None
    cached_motion_global_start_frames: Optional[torch.Tensor] = None
    cached_motion_global_end_frames: Optional[torch.Tensor] = None
    cached_motion_original_num_frames: Optional[torch.Tensor] = None
    cached_clip_info: Optional[List[dict]] = None

    # Motion data tensors
    global_body_translation: Optional[torch.Tensor] = None
    global_body_rotation: Optional[torch.Tensor] = None
    global_body_velocity: Optional[torch.Tensor] = None
    global_body_angular_velocity: Optional[torch.Tensor] = None
    global_body_translation_extend: Optional[torch.Tensor] = None
    global_body_rotation_extend: Optional[torch.Tensor] = None
    global_body_velocity_extend: Optional[torch.Tensor] = None
    global_body_angular_velocity_extend: Optional[torch.Tensor] = None
    frame_flag: Optional[torch.Tensor] = None

    local_body_rotation: Optional[torch.Tensor] = None

    dof_pos: Optional[torch.Tensor] = None
    dof_vels: Optional[torch.Tensor] = None

    gravity_vec: torch.Tensor = torch.tensor([0.0, 0.0, -1.0])

    def reset(self):
        """Efficiently reset all tensors using zero_() for in-place operation.

        This is faster than creating new tensors when frequently updating the
        cache.
        """
        assert self.num_envs > 0, (
            "num_envs must be set before resetting the cache"
        )
        # Define shape for each tensor
        shapes = {
            "global_body_translation": (
                self.num_envs,
                self.max_frame_length,
                self.num_bodies,
                3,
            ),
            "global_body_rotation": (
                self.num_envs,
                self.max_frame_length,
                self.num_bodies,
                4,
            ),
            "global_body_velocity": (
                self.num_envs,
                self.max_frame_length,
                self.num_bodies,
                3,
            ),
            "global_body_angular_velocity": (
                self.num_envs,
                self.max_frame_length,
                self.num_bodies,
                3,
            ),
            "global_body_translation_extend": (
                self.num_envs,
                self.max_frame_length,
                self.num_extended_bodies,
                3,
            ),
            "global_body_rotation_extend": (
                self.num_envs,
                self.max_frame_length,
                self.num_extended_bodies,
                4,
            ),
            "global_body_velocity_extend": (
                self.num_envs,
                self.max_frame_length,
                self.num_extended_bodies,
                3,
            ),
            "global_body_angular_velocity_extend": (
                self.num_envs,
                self.max_frame_length,
                self.num_extended_bodies,
                3,
            ),
            "local_body_rotation": (
                self.num_envs,
                self.max_frame_length,
                self.num_extended_bodies,
                4,
            ),
            "dof_pos": (self.num_envs, self.max_frame_length, self.num_dofs),
            "dof_vels": (self.num_envs, self.max_frame_length, self.num_dofs),
            "frame_flag": (self.num_envs, self.max_frame_length),
        }

        # Initialize or reset each tensor
        for tensor_name, shape in shapes.items():
            # Get current tensor
            current_tensor = getattr(self, tensor_name)

            # If tensor doesn't exist or shape has changed, create a new one
            if current_tensor is None or current_tensor.shape != shape:
                setattr(
                    self, tensor_name, torch.zeros(shape, device=self.device)
                )
            else:
                # Otherwise, zero out the existing tensor in-place
                current_tensor.zero_()

    def __getitem__(self, key: str) -> torch.Tensor:
        """Allow dict-like access to cache tensors."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: torch.Tensor):
        """Allow dict-like setting of cache tensors."""
        if isinstance(value, torch.Tensor):
            value = value.to(self.device)
        setattr(self, key, value)

    def register_motion_ids(self, motion_ids: torch.Tensor):
        self.cached_motion_ids = motion_ids
        self.num_envs = len(motion_ids)
        self.reset()

    def get_motion_state(
        self,
        motion_global_frame_ids: torch.Tensor,
        global_offset: Union[None, torch.Tensor] = None,
        n_fut_frames: int = 0,
        target_fps: Optional[float] = None,
    ):
        """Obtain the motion state for one or more consecutive frames.

        Handles sequences that extend beyond cached data by padding with zeros.
        Allows for non-continuous future frame fetching based on target_fps.

        Args:
            motion_global_frame_ids: Tensor of global frame IDs (B) for the
                first frame in the sequence.
            global_offset: Optional global position offset to apply (B, 3).
            n_fut_frames: Number of *additional* future frames to fetch after
                motion_global_frame_ids. If 0, fetches only the frame specified
                by motion_global_frame_ids. The total number of frames fetched
                is 1 (current) or n_fut_frames + 1.
            target_fps: Optional target frames per second for sampling future
                frames. If None or invalid, defaults to self.fps.

        Returns:
            Dictionary containing the motion state.
            Tensors will always have a time dimension equal to the number of
                frames fetched. Frames outside the valid cached range will be
                padded with zeros.

        """
        motion_ids = self.cached_motion_ids  # Shape: [B]
        if motion_ids is None:
            raise ValueError("Motion IDs not registered in cache!")
        bs = len(motion_ids)

        assert len(motion_ids) == len(motion_global_frame_ids), (
            "motion_ids and motion_global_frame_ids must have the same length!"
        )
        assert n_fut_frames >= 0, "n_fut_frames cannot be negative."

        motion_global_frame_ids = motion_global_frame_ids.to(self.device)

        # --- Determine frame offsets based on n_fut_frames and target_fps ---
        frame_offsets_list = [0]  # Always include the current frame (offset 0)

        if n_fut_frames > 0:
            frame_offsets_list.append(
                1
            )  # Always include the immediate next frame (offset 1)

            if n_fut_frames > 1:
                num_sparse_fut_frames = n_fut_frames - 1

                effective_target_fps = (
                    self.fps
                )  # Default to continuous sampling
                if target_fps is not None and target_fps > 0:
                    effective_target_fps = target_fps

                if (
                    effective_target_fps <= 0
                ):  # Should not happen if target_fps > 0 or self.fps > 0
                    time_step_ratio = 1.0  # Fallback to continuous
                else:
                    time_step_ratio = self.fps / effective_target_fps

                for k in range(num_sparse_fut_frames):
                    offset = 1 + round((k + 1) * time_step_ratio)
                    frame_offsets_list.append(int(offset))

        frame_offsets = torch.tensor(
            frame_offsets_list, device=self.device, dtype=torch.long
        )
        num_frames_to_fetch = len(frame_offsets)

        # Shape: [B, num_frames_to_fetch]
        motion_global_frame_ids_seq_raw = (
            motion_global_frame_ids[:, None] + frame_offsets[None, :]
        )

        if (
            self.cached_motion_global_start_frames is None
            or self.cached_motion_global_end_frames is None
        ):
            raise ValueError(
                "Ensure cache is populated before calling get_motion_state."
            )

        # Convert global frame IDs to relative indices in the cache
        # Shape: [B, num_frames_to_fetch]
        relative_frame_indices_seq = (
            motion_global_frame_ids_seq_raw
            - self.cached_motion_global_start_frames[:, None]
        )
        relative_indices_long = relative_frame_indices_seq.long()

        num_actually_cached_frames = (
            self.cached_motion_global_end_frames
            - self.cached_motion_global_start_frames
        )  # Shape [B]

        # Mask for valid data reading from cache
        read_mask = (
            (relative_indices_long >= 0)
            & (relative_indices_long < num_actually_cached_frames[:, None])
            & (relative_indices_long < self.max_frame_length)
        )
        # read_mask has shape [B, num_frames_to_fetch]

        batch_indices = torch.arange(
            bs, device=motion_ids.device
        )  # Shape: [B]
        batch_indices_expanded = batch_indices[:, None].expand(
            -1, num_frames_to_fetch
        )

        dof_pos_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_dofs),
            device=self.device,
            dtype=self.dof_pos.dtype
            if self.dof_pos is not None
            else torch.float32,
        )
        dof_vels_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_dofs),
            device=self.device,
            dtype=self.dof_vels.dtype
            if self.dof_vels is not None
            else torch.float32,
        )
        global_body_translation_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_bodies, 3),
            device=self.device,
            dtype=self.global_body_translation.dtype
            if self.global_body_translation is not None
            else torch.float32,
        )
        global_body_rotation_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_bodies, 4),
            device=self.device,
            dtype=self.global_body_rotation.dtype
            if self.global_body_rotation is not None
            else torch.float32,
        )
        global_body_velocity_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_bodies, 3),
            device=self.device,
            dtype=self.global_body_velocity.dtype
            if self.global_body_velocity is not None
            else torch.float32,
        )
        global_body_angular_velocity_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_bodies, 3),
            device=self.device,
            dtype=self.global_body_angular_velocity.dtype
            if self.global_body_angular_velocity is not None
            else torch.float32,
        )

        global_body_translation_extend_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_extended_bodies, 3),
            device=self.device,
            dtype=self.global_body_translation_extend.dtype
            if self.global_body_translation_extend is not None
            else torch.float32,
        )
        global_body_rotation_extend_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_extended_bodies, 4),
            device=self.device,
            dtype=self.global_body_rotation_extend.dtype
            if self.global_body_rotation_extend is not None
            else torch.float32,
        )
        global_body_velocity_extend_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_extended_bodies, 3),
            device=self.device,
            dtype=self.global_body_velocity_extend.dtype
            if self.global_body_velocity_extend is not None
            else torch.float32,
        )
        global_body_angular_velocity_extend_out = torch.zeros(
            (bs, num_frames_to_fetch, self.num_extended_bodies, 3),
            device=self.device,
            dtype=self.global_body_angular_velocity_extend.dtype
            if self.global_body_angular_velocity_extend is not None
            else torch.float32,
        )

        frame_flag_out = torch.zeros(
            (bs, num_frames_to_fetch),
            device=self.device,
            dtype=self.frame_flag.dtype
            if self.frame_flag is not None
            else torch.long,
        )

        # --- Populate output tensors using the mask ---
        src_batch_indices_flat = batch_indices_expanded[read_mask]
        src_frame_indices_flat = relative_indices_long[read_mask]

        output_frame_indices_template = torch.arange(
            num_frames_to_fetch, device=self.device
        )[None, :].expand(bs, -1)
        tgt_batch_indices_flat = batch_indices_expanded[read_mask]
        tgt_frame_indices_flat = output_frame_indices_template[read_mask]

        if (
            src_batch_indices_flat.numel() > 0
        ):  # Only copy if there's valid data to read
            # Ensure source tensors are not None before indexing
            if self.dof_pos is not None:
                dof_pos_out[tgt_batch_indices_flat, tgt_frame_indices_flat] = (
                    self.dof_pos[
                        src_batch_indices_flat, src_frame_indices_flat
                    ]
                )
            if self.dof_vels is not None:
                dof_vels_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.dof_vels[
                    src_batch_indices_flat, src_frame_indices_flat
                ]

            if self.global_body_translation is not None:
                global_body_translation_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_translation[
                    src_batch_indices_flat, src_frame_indices_flat
                ]
            if self.global_body_rotation is not None:
                global_body_rotation_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_rotation[
                    src_batch_indices_flat, src_frame_indices_flat
                ]
            if self.global_body_velocity is not None:
                global_body_velocity_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_velocity[
                    src_batch_indices_flat, src_frame_indices_flat
                ]
            if self.global_body_angular_velocity is not None:
                global_body_angular_velocity_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_angular_velocity[
                    src_batch_indices_flat, src_frame_indices_flat
                ]

            if self.global_body_translation_extend is not None:
                global_body_translation_extend_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_translation_extend[
                    src_batch_indices_flat, src_frame_indices_flat
                ]
            if self.global_body_rotation_extend is not None:
                global_body_rotation_extend_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_rotation_extend[
                    src_batch_indices_flat, src_frame_indices_flat
                ]
            if self.global_body_velocity_extend is not None:
                global_body_velocity_extend_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_velocity_extend[
                    src_batch_indices_flat, src_frame_indices_flat
                ]
            if self.global_body_angular_velocity_extend is not None:
                global_body_angular_velocity_extend_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.global_body_angular_velocity_extend[
                    src_batch_indices_flat, src_frame_indices_flat
                ]

            if self.frame_flag is not None:
                frame_flag_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] = self.frame_flag[
                    src_batch_indices_flat, src_frame_indices_flat
                ]

        # Apply global offset if provided
        if global_offset is not None:
            # Offset shape: [B, 3]
            # Select offsets for valid batch entries: shape (N_valid, 3)
            selected_offsets = global_offset.to(self.device)[
                tgt_batch_indices_flat
            ]

            offset_for_body_tensors = selected_offsets[:, None, :]

            if self.global_body_translation is not None:
                global_body_translation_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] += offset_for_body_tensors

            if self.global_body_translation_extend is not None:
                global_body_translation_extend_out[
                    tgt_batch_indices_flat, tgt_frame_indices_flat
                ] += offset_for_body_tensors

        global_root_translation = global_body_translation_out[..., 0, :]
        global_root_rotation = global_body_rotation_out[..., 0, :]
        global_root_velocity = global_body_velocity_out[..., 0, :]
        global_root_angular_velocity = global_body_angular_velocity_out[
            ..., 0, :
        ]

        # Construct and return the state dictionary
        return_dict = {
            "dof_pos": dof_pos_out,
            "dof_vel": dof_vels_out,
            "root_pos": global_root_translation,
            "root_rot": global_root_rotation,
            "root_vel": global_root_velocity,
            "root_ang_vel": global_root_angular_velocity,
            "rg_pos": global_body_translation_out,
            "rb_rot": global_body_rotation_out,
            "body_vel": global_body_velocity_out,
            "body_ang_vel": global_body_angular_velocity_out,
            "rg_pos_t": global_body_translation_extend_out,
            "rg_rot_t": global_body_rotation_extend_out,
            "body_vel_t": global_body_velocity_extend_out,
            "body_ang_vel_t": global_body_angular_velocity_extend_out,
            "frame_flag": frame_flag_out,
            "valid_frame_flag": read_mask,
        }

        return return_dict

    def sample_cached_global_start_frames(
        self,
        env_ids: torch.Tensor,
        n_fut_frames: int = 0,
        eval: bool = False,
    ) -> torch.Tensor:
        """For each motion, sample a random global start frame.

        Args:
            env_ids (torch.Tensor): Indices of environments/motions.
            n_fut_frames (int): The number of additional future frames
                required after the sampled start frame (default: 0). Total
                frames needed = n_fut_frames + 1.

        Returns:
            torch.Tensor: Sampled global start frame IDs for the
                specified env_ids.

        """
        env_ids = env_ids.cpu()
        global_start_frames = self.cached_motion_global_start_frames[env_ids]
        global_end_frames = self.cached_motion_global_end_frames[env_ids]
        if eval:
            sampled_global_start_frames = global_start_frames
        else:
            cached_duration = global_end_frames - global_start_frames
            valid_duration = cached_duration - n_fut_frames - 1
            rand_factors = torch.rand(
                len(env_ids), device=global_start_frames.device
            )
            sampled_offset = torch.floor(rand_factors * valid_duration).long()
            sampled_global_start_frames = global_start_frames + sampled_offset

        return sampled_global_start_frames.to(self.device)

    def move_cache_to_device(self, device: torch.device):
        """Move tensors to device.

        Moves all torch.Tensor attributes of the given OnlineMotionCache
        instance to the specified device *in-place*.

        Args:
            device: The target torch.device.

        """
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                try:
                    moved_tensor = attr_value.to(device)
                    setattr(self, attr_name, moved_tensor)
                except Exception as e:
                    logger.error(
                        f"Failed to move tensor attribute '{attr_name}'"
                        f"to device {device}: {e}"
                    )

    @property
    def motion_clip_full(self):
        bs, ts = self.dof_pos.shape[:2]
        return torch.cat(
            [
                self.global_body_translation.view(bs, ts, -1),  # NBx3
                self.global_body_rotation.view(bs, ts, -1),  # NBx4
                self.local_body_rotation.view(bs, ts, -1),  # (NB+NE)x4
                self.global_body_velocity.view(bs, ts, -1),  # NBx3
                self.global_body_angular_velocity.view(bs, ts, -1),  # NBx3
                self.global_body_velocity_extend.view(bs, ts, -1),  # (NB+NE)x3
                self.global_body_angular_velocity_extend.view(
                    bs, ts, -1
                ),  # (NB+NE)x3
                self.dof_vels.view(bs, ts, -1),  # ND
                self.dof_pos.view(bs, ts, -1),  # ND
            ],
            dim=-1,
        )

    def sample_demo_seq(self, num_samples: int, seq_len: int):
        sampled_local_motion_ids = torch.randint(
            0, len(self.cached_motion_ids), (num_samples,)
        )

        # get the valid num of frames for each motion
        valid_max_num_frames = (
            self.cached_motion_global_end_frames
            - self.cached_motion_global_start_frames
        )[sampled_local_motion_ids]
        valid_num_frames = valid_max_num_frames - seq_len

        # sample from [0, valid_start_frames)
        start_frames = (
            torch.rand(num_samples, device=self.device) * valid_num_frames
        ).long()

        # Create sequence indices: shape (num_samples, seq_len)
        seq_indices = (
            start_frames[:, None]
            + torch.arange(seq_len, device=self.device)[None, :]
        )
        motion_indices_expanded = sampled_local_motion_ids[:, None]
        # Slice out the sequences using advanced indexing
        dof_pos = self.dof_pos[motion_indices_expanded, seq_indices]
        dof_vels = self.dof_vels[motion_indices_expanded, seq_indices]
        root_pos = self.global_body_translation[
            motion_indices_expanded, seq_indices, 0
        ]
        root_rot = self.global_body_rotation[
            motion_indices_expanded, seq_indices, 0
        ]
        root_vel = self.global_body_velocity[
            motion_indices_expanded, seq_indices, 0
        ]
        root_ang_vel = self.global_body_angular_velocity[
            motion_indices_expanded, seq_indices, 0
        ]
        key_body_pos = self.global_body_translation[
            motion_indices_expanded, seq_indices
        ][
            :, :, self.key_body_indices
        ]  # (num_samples, seq_len, num_key_bodies, 3)

        # --- Start: Process key_body_pos ---
        # Calculate inverse heading rotation for each frame in the sequence
        heading_rot_inv = calc_heading_quat_inv(
            root_rot.reshape(-1, 4), w_last=True
        ).reshape(num_samples, seq_len, 4)  # (num_samples, seq_len, 4)

        # Make key body positions relative to root
        local_key_body_pos = (
            key_body_pos - root_pos[:, :, None, :]
        )  # (num_samples, seq_len, num_key_bodies, 3)

        # Reshape for rotation: flatten batch and seq dims
        num_samples, seq_len, num_key_bodies, _ = local_key_body_pos.shape
        flat_local_key_body_pos = local_key_body_pos.view(
            num_samples * seq_len, num_key_bodies, 3
        )
        flat_heading_rot_inv = heading_rot_inv.view(num_samples * seq_len, 4)[
            :, None, :
        ].expand(-1, num_key_bodies, -1)

        # Apply rotation
        flat_rotated_local_key_pos = my_quat_rotate(
            flat_heading_rot_inv.reshape(-1, 4),
            flat_local_key_body_pos.reshape(-1, 3),
        )

        # Reshape back and flatten key body dim
        flat_local_key_pos = flat_rotated_local_key_pos.view(
            num_samples, seq_len, num_key_bodies * 3
        )

        return {
            "root_pos": root_pos,  # (num_samples, seq_len, 3)
            "root_rot": root_rot,  # (num_samples, seq_len, 4)
            "root_vel": root_vel,  # (num_samples, seq_len, 3)
            "root_ang_vel": root_ang_vel,  # (num_samples, seq_len, 3)
            "dof_pos": dof_pos,  # (num_samples, seq_len, num_dofs)
            "dof_vels": dof_vels,  # (num_samples, seq_len, num_dofs)
            "flat_local_key_pos": flat_local_key_pos,
        }

    def sample_demo_seq_global(self, num_samples: int, seq_len: int):
        sampled_local_motion_ids = torch.randint(
            0, len(self.cached_motion_ids), (num_samples,)
        )
        # get the valid num of frames for each motion
        valid_max_num_frames = (
            self.cached_motion_global_end_frames
            - self.cached_motion_global_start_frames
        )[sampled_local_motion_ids]
        valid_num_frames = valid_max_num_frames - seq_len
        # sample from [0, valid_start_frames)
        start_frames = (
            torch.rand(num_samples, device=self.device) * valid_num_frames
        ).long()
        # Create sequence indices: shape (num_samples, seq_len)
        seq_indices = (
            start_frames[:, None]
            + torch.arange(seq_len, device=self.device)[None, :]
        )
        motion_indices_expanded = sampled_local_motion_ids[:, None]
        # Slice out the sequences using advanced indexing
        dof_pos = self.dof_pos[motion_indices_expanded, seq_indices]
        dof_vels = self.dof_vels[motion_indices_expanded, seq_indices]
        root_pos = self.global_body_translation[
            motion_indices_expanded, seq_indices, 0
        ]
        root_rot = self.global_body_rotation[
            motion_indices_expanded, seq_indices, 0
        ]
        root_vel = self.global_body_velocity[
            motion_indices_expanded, seq_indices, 0
        ]
        root_ang_vel = self.global_body_angular_velocity[
            motion_indices_expanded, seq_indices, 0
        ]
        global_bodylink_pos = self.global_body_translation_extend[
            motion_indices_expanded, seq_indices
        ]
        return {
            "global_root_pos": root_pos,
            "global_root_rot": root_rot,
            "global_root_vel": root_vel,
            "global_root_ang_vel": root_ang_vel,
            "global_bodylink_pos": global_bodylink_pos,
            "dof_pos": dof_pos,
            "dof_vels": dof_vels,
        }


class LmdbMotionLib:
    """LMDB-based Motion Library for humanoid robot training.

    This class manages motion data stored in LMDB format and supports both uniform
    and weighted sampling strategies for motion selection during training.

    Features:
    - Motion data caching for efficient access
    - Score-based motion sampling with configurable base probability
    - Public API for updating motion scores during training
    - Sub-motion indexing for finer-grained curriculum learning

    Sub-motion Indexing:
    When enabled, long motion clips are automatically split into consecutive
    sub-clips of max_frame_length, allowing curriculum learning to operate at
    a finer granularity. This addresses the issue where long motions contain
    both easy and hard subsequences that would otherwise be treated as a single unit.

    Example usage:
        # Initialize with weighted sampling and sub-motion indexing enabled
        motion_lib = LmdbMotionLib(motion_lib_cfg, device)

        # Update scores for specific motions during training
        if motion_lib.use_sub_motion_indexing:
            motion_lib.update_sub_motion_score(sub_motion_id=42, score=0.8)
        else:
            motion_lib.update_motion_score(motion_id=42, score=0.8)

        # Batch update scores
        motion_ids = torch.tensor([10, 20, 30])
        scores = torch.tensor([1.2, 0.5, 1.5])
        if motion_lib.use_sub_motion_indexing:
            motion_lib.update_sub_motion_scores_batch(motion_ids, scores)
        else:
            motion_lib.update_motion_scores_batch(motion_ids, scores)

        # Sample new motions (will use sub-motion or motion sampling based on config)
        sampled_motion_ids = motion_lib.resample_new_motions(num_samples=100)

    Configuration:
        Set in robot config under motion section:
        motion:
            use_linear_weighted_sampling: True   # Enable weighted sampling
            sampling_base_probability: 0.1       # Base probability for all clips
            use_sub_motion_indexing: True        # Enable sub-motion indexing
            sub_motion_overlap_frames: 50        # Overlap between consecutive sub-motions
            dump_sampling_info: False            # Enable JSON dumping of sampling info

        TD Error-based Curriculum Learning (automatic when using weighted sampling):
        - Raw TD error tracking with EMA smoothing for stable logging
        - Absolute TD error used for motion scoring to prioritize difficult clips
        - Simple and effective approach without complex phase classification
    """

    def __init__(
        self,
        motion_lib_cfg,
        cache_device,
        process_id: int = 0,
        num_processes: int = 1,
    ):
        self.m_cfg = motion_lib_cfg

        self.motion_log_dir = self.m_cfg.get("motion_log_dir", None)
        if self.motion_log_dir is not None:
            os.makedirs(self.motion_log_dir, exist_ok=True)

        self.min_frame_length = self.m_cfg.get("min_frame_length", 0)
        self._sim_fps = 1 / self.m_cfg.get("step_dt", 1 / 50)
        self.cache_device = cache_device
        self._lmdb_handle = None  # Initialize handle to None for lazy loading
        self.process_id = process_id
        self.num_processes = num_processes
        # Read metadata once using a temporary handle
        self.handpicked_motion_names = set(
            self.m_cfg.get("handpicked_motion_names", [])
        )
        self.excluded_motion_names = set(
            self.m_cfg.get("excluded_motion_names", [])
        )

        # Motion scoring configuration
        self.use_linear_weighted_sampling = self.m_cfg.get(
            "use_linear_weighted_sampling", False
        )
        self.sampling_base_probability = self.m_cfg.get(
            "sampling_base_probability", 0.1
        )

        # Sub-motion indexing configuration
        self.use_sub_motion_indexing = self.m_cfg.get(
            "use_sub_motion_indexing", False
        )
        self.sub_motion_overlap_frames = self.m_cfg.get(
            "sub_motion_overlap_frames", 0
        )
        logger.info(
            f"Sub-motion indexing enabled: {self.use_sub_motion_indexing}"
        )
        if self.use_sub_motion_indexing:
            logger.info(
                f"Sub-motion overlap frames: {self.sub_motion_overlap_frames}"
            )

        # Sampling info dumping configuration
        self.dump_sampling_info = self.m_cfg.get("dump_sampling_info", False)
        logger.info(
            f"Sampling info dumping enabled: {self.dump_sampling_info}"
        )

        raw_all_motion_keys = []
        try:
            with lmdb.open(
                self.m_cfg.motion_file, readonly=True, lock=False
            ) as temp_env:
                with temp_env.begin() as txn:
                    raw_all_motion_keys = pickle.loads(txn.get(b"all_uuids"))

                    # --- Filter motions based on different criteria ---
                    self.all_motion_keys = []
                    self.train_motion_keys = []
                    self.val_motion_keys = []
                    num_filtered_out = 0
                    total_num_frames = 0
                    total_wallclock_time = 0.0
                    for key in track(
                        raw_all_motion_keys,
                        description="Filtering motions ...",
                    ):
                        # --- Filter motions based on handpicked_motion_names ---
                        if self.handpicked_motion_names:
                            if key not in self.handpicked_motion_names:
                                continue

                        # --- Filter motions based on excluded_motion_names ---
                        if key in self.excluded_motion_names:
                            logger.info(
                                f"Filtering motion {key} dueto excluded motion names."
                            )
                            num_filtered_out += 1
                            continue

                        # --- Filter motions based on min_frame_length ---
                        metadata = pickle.loads(
                            txn.get(f"motion/{key}/metadata".encode())
                        )
                        num_frames = metadata["num_frames"]
                        wallclock_len = metadata["wallclock_len"]

                        # --- Filter motions based on min_frame_length ---
                        filter_flag = False
                        if num_frames < self.min_frame_length:
                            filter_flag = True

                        if not filter_flag:
                            self.all_motion_keys.append(key)
                            # if key in raw_train_motion_keys_set:
                            #     self.train_motion_keys.append(key)
                            # if key in raw_val_motion_keys_set:
                            #     self.val_motion_keys.append(key)
                            total_num_frames += num_frames
                            total_wallclock_time += wallclock_len
                        else:
                            num_filtered_out += 1

                    # --- Filter motions based on motion keys ---

                    # --- Statistics for filtered motions ---
                    logger.info(
                        f"Number of raw clips: {len(raw_all_motion_keys)}"
                    )
                    logger.info(
                        f"Number of filtered-out clips: {num_filtered_out}"
                    )
                    logger.info(
                        f"Number of remaining clips: {len(self.all_motion_keys)}"
                    )
                    logger.info(
                        f"Total frame length after filtering: "
                        f"{total_num_frames} frames."
                    )
                    logger.info(
                        f"Total wall clock time after filtering: "
                        f"{total_wallclock_time:.2f} seconds."
                    )

        except lmdb.Error as e:
            logger.error(
                f"Failed to open or read LMDB metadata from "
                f"{self.m_cfg.motion_file}: {e}"
            )
            raise
        logger.info(f"All motion keys: {self.all_motion_keys[:20]}")
        self.motion_id2key = {
            motion_id: key
            for motion_id, key in enumerate(self.all_motion_keys)
        }
        self.motion_key2id = {
            key: motion_id
            for motion_id, key in enumerate(self.all_motion_keys)
        }
        self.motion_ids = list(self.motion_id2key.keys())

        # Initialize motion scores - start with uniform scores of ones
        self.motion_scores = torch.ones(
            len(self.motion_ids), dtype=torch.float32
        )
        logger.info(
            f"Motion scoring enabled: {self.use_linear_weighted_sampling}"
        )
        if self.use_linear_weighted_sampling:
            logger.info(
                f"Sampling base probability: {self.sampling_base_probability}"
            )

        self.max_frame_length = self.m_cfg.get("max_frame_length", 500)
        self.n_fut_frames = self.m_cfg.get("n_fut_frames", 1)
        self.num_dofs = len(self.m_cfg.dof_names)
        self.num_bodies = len(self.m_cfg.body_names)
        self.num_extended_bodies = self.num_bodies + len(
            self.m_cfg.extend_config
        )
        self.key_bodies = self.m_cfg.get("key_bodies", [])
        self.body_names = self.m_cfg.body_names
        self.extended_body_names = self.body_names + [
            i["joint_name"] for i in self.m_cfg.extend_config
        ]
        self.key_body_indices = [
            self.extended_body_names.index(body) for body in self.key_bodies
        ]

        # Windowed-UCB curriculum learning
        self.ucb_window_size = self.m_cfg.get(
            "ucb_window_size", 1000
        )  # Number of recent samples to consider

        # Build sub-motion indexing system
        if self.use_sub_motion_indexing:
            self._build_sub_motion_index()
        else:
            # For backward compatibility, sub-motion system mirrors original system
            self.sub_motion_infos = None
            self.sub_motion_scores = self.motion_scores.clone()
            self.num_sub_motions = len(self.motion_ids)

            # TD error tracking with EMA for logging
        self.motion_td_ema = torch.zeros(
            len(self.motion_ids), dtype=torch.float32
        )  # EMA of raw TD errors for logging
        self.td_ema_alpha = 0.1  # EMA smoothing factor

        self.motion_sample_counts = torch.zeros(
            len(self.motion_ids), dtype=torch.long
        )  # Track sampling frequency (all-time)
        self.motion_windowed_sample_counts = torch.zeros(
            len(self.motion_ids), dtype=torch.long
        )  # Track sampling frequency (windowed)
        self.motion_td_variance_ema = torch.zeros(
            len(self.motion_ids), dtype=torch.float32
        )  # Track TD error variance

        # Circular buffer for windowed sample tracking
        self.motion_sample_buffer = torch.full(
            (self.ucb_window_size,), -1, dtype=torch.long
        )  # -1 indicates empty slot
        self.buffer_ptr = 0  # Current position in circular buffer
        self.buffer_filled = False  # Whether buffer has been filled once
        self.total_windowed_samples = 0  # Total samples in current window
        # self.train_motion_ids = list(
        #     [self.motion_key2id[key] for key in self.train_motion_keys]
        # )
        # self.val_motion_ids = list(
        #     [self.motion_key2id[key] for key in self.val_motion_keys]
        # )

        if len(self.all_motion_keys) < self.num_processes:
            logger.info(
                f"Fewer motion clips ({len(self.all_motion_keys)}) than "
                f"processes ({self.num_processes}). Will replicate keys."
            )
            # Calculate how many times we need to repeat the keys
            repeat_count = (
                self.num_processes + len(self.all_motion_keys) - 1
            ) // len(self.all_motion_keys)
            # Replicate the keys
            replicated_keys = []
            for _ in range(repeat_count):
                replicated_keys.extend(self.all_motion_keys)
            # Take just what we need
            self.all_motion_keys = replicated_keys[: self.num_processes]
            logger.info(
                f"Replicated motion keys to have {len(self.all_motion_keys)}."
            )

        num_motions_per_process = max(
            1, len(self.all_motion_keys) // self.num_processes
        )
        cur_proc_eval_start_idx = self.process_id * num_motions_per_process
        cur_proc_eval_end_idx = min(
            cur_proc_eval_start_idx + num_motions_per_process,
            len(self.all_motion_keys),
        )
        self.eval_motion_keys = self.all_motion_keys[
            cur_proc_eval_start_idx:cur_proc_eval_end_idx
        ]

        # Handle edge case for last process
        if (
            cur_proc_eval_start_idx >= len(self.all_motion_keys)
            and self.process_id == self.num_processes - 1
        ):
            self.eval_motion_keys = self.all_motion_keys[-1:]
        elif len(self.eval_motion_keys) == 0:
            # This shouldn't happen with the replication, but just in case
            self.eval_motion_keys = [
                self.all_motion_keys[
                    self.process_id % len(self.all_motion_keys)
                ]
            ]
            logger.info(
                f"Process {self.process_id} assigned replicated motion clip."
            )

        # Pre-calculate the evaluation motion clip allocation schedule
        self.eval_allocation_schedule: List[dict] = self._eval_preallocation()
        # Index for evaluation schedule
        self.eval_schedule_idx = 0

        # Initialize the motion cache (will be populated later)
        self.cache = OnlineMotionCache(
            device=cache_device,
            num_envs=0,  # Will be set when populating
            max_frame_length=self.max_frame_length,
            num_bodies=self.num_bodies,
            num_dofs=self.num_dofs,
            num_extended_bodies=self.num_extended_bodies,
            key_body_indices=self.key_body_indices,
            n_fut_frames=self.n_fut_frames,
            fps=self._sim_fps,
        )

        logger.info("MotionLib initialized !")

    def __enter__(self):
        """Context manager entry - ensure handle is open."""
        _ = self.lmdb_handle  # This will open the handle if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the handle."""
        if self._lmdb_handle is not None:
            self._lmdb_handle.close()
            self._lmdb_handle = None

    @property
    def lmdb_handle(self):
        """Lazy opening of the LMDB handle for process safety."""
        if self._lmdb_handle is None:
            try:
                self._lmdb_handle = lmdb.open(
                    self.m_cfg.motion_file,
                    readonly=True,
                    max_readers=2048,
                    max_dbs=0,
                    lock=False,
                )
                logger.debug(f"LMDB handle opened for process {os.getpid()}")
            except lmdb.Error as e:
                logger.error(
                    f"Failed to open LMDB database at {self.m_cfg.motion_file}"
                    f"in process {os.getpid()}: {e}"
                )
                raise
        return self._lmdb_handle

    @property
    def _num_unique_motions(self) -> int:
        return len(self.all_motion_keys)

    def get_motion_wallclock_length(
        self, motion_ids: torch.Tensor
    ) -> torch.Tensor:
        # Use the property to get the handle for this process
        with self.lmdb_handle.begin() as txn:
            motion_lengths = []
            for motion_id in motion_ids:
                key = self.motion_id2key[motion_id.item()]
                motion_lengths.append(
                    pickle.loads(txn.get(f"motion/{key}/metadata".encode()))[
                        "wallclock_len"
                    ]
                )
        return torch.tensor(motion_lengths)

    def get_motion_num_frames(self, motion_ids: List[int]) -> List[int]:
        # Use the property to get the handle for this process
        with self.lmdb_handle.begin() as txn:
            motion_num_frames = []
            for motion_id in motion_ids:
                key = self.motion_id2key[
                    motion_id.item()
                    if isinstance(motion_id, torch.Tensor)
                    else motion_id
                ]
                motion_num_frames.append(
                    pickle.loads(txn.get(f"motion/{key}/metadata".encode()))[
                        "num_frames"
                    ]
                )
        return torch.tensor(motion_num_frames)

    def sample_wallclock_time(
        self,
        motion_ids: torch.Tensor,
        truncate_time: float = None,
    ) -> torch.Tensor:
        motion_phase = torch.rand(len(motion_ids))
        motion_len = (
            self.get_motion_wallclock_length(motion_ids).clone().detach()
        )
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = motion_phase * motion_len
        return motion_time

    def sample_global_start_frames(
        self,
        motion_ids: torch.Tensor,
        eval: bool = False,
    ) -> torch.Tensor:
        motion_global_num_frames_total = (
            self.get_motion_num_frames(motion_ids).clone().detach()
        )
        max_start_frame = torch.where(
            motion_global_num_frames_total > self.max_frame_length,
            self.max_frame_length - self.min_frame_length,
            motion_global_num_frames_total - self.min_frame_length,
        )
        if eval:
            motion_global_start_frames = torch.zeros(
                len(motion_ids), dtype=torch.long
            )
        else:
            rand_factors = torch.rand(len(motion_ids))
            motion_global_start_frames = torch.floor(
                rand_factors * max_start_frame
            ).long()

        return motion_global_start_frames

    def resample_new_motions(
        self, num_samples: int, eval: bool = False
    ) -> torch.Tensor:
        logger.info("Starting resampling new motions ...")
        start_time = time.time()

        # Choose sampling strategy based on configuration
        if self.use_linear_weighted_sampling and not eval:
            sampled_motion_ids = self._sample_motion_ids_weighted(num_samples)
        else:
            sampled_motion_ids = self._sample_motion_ids_uniform(num_samples)

        if self.use_sub_motion_indexing:
            # For sub-motion indexing, we don't need to sample global start frames
            # since they're already defined in the sub-motion information
            sampled_global_start_frames = torch.zeros(
                num_samples, dtype=torch.long
            )
        else:
            sampled_global_start_frames = self.sample_global_start_frames(
                sampled_motion_ids, eval=eval
            )

        self.cache.register_motion_ids(sampled_motion_ids)
        self._build_online_train_cache(
            self.cache,
            sampled_global_start_frames,
        )

        # Dump motion scores and sampling probabilities to JSON file (if enabled)
        # Only dump on main process (process_id 0) to avoid VRAM waste on worker processes
        if self.dump_sampling_info and self.process_id == 0:
            self._dump_motion_sampling_info(eval)

        end_time = time.time()

        # Extract first 16 sampled motion keys for logging
        if self.use_sub_motion_indexing:
            sampled_keys_preview = [
                f"{self.sub_motion_infos[idx.item()]['original_motion_key']}_sub{idx.item()}"
                for idx in sampled_motion_ids[:16]
            ]
        else:
            sampled_keys_preview = [
                self.all_motion_keys[idx.item()]
                for idx in sampled_motion_ids[:16]
            ]
        sampled_keys_str = "\n".join(sampled_keys_preview)

        sampling_method = (
            "weighted"
            if (self.use_linear_weighted_sampling and not eval)
            else "uniform"
        )
        indexing_method = (
            "sub-motion" if self.use_sub_motion_indexing else "motion"
        )

        logger.info(
            f"""
            New start frames sampled using {sampling_method} {indexing_method} sampling !!! Cache updated in:
            {(end_time - start_time):.4f} seconds.
            Sampled motion names:\n{sampled_keys_str}\n...\n
            """
        )
        return sampled_motion_ids

    def _dump_motion_sampling_info(self, eval: bool = False):
        """Dump motion scores and sampling probabilities to JSON file.

        Args:
            eval (bool): Whether this is an evaluation sampling (affects filename)
        """
        try:
            logger.debug("Dumping motion sampling info to JSON file...")
            # Get current sampling probabilities and scores
            sampling_data = self.get_current_sampling_probabilities()

            # Add timestamp and additional metadata
            sampling_data["timestamp"] = time.time()
            sampling_data["eval_mode"] = eval
            sampling_data["resampling_iteration"] = getattr(
                self, "_resampling_counter", 0
            )

            # Increment resampling counter
            self._resampling_counter = (
                getattr(self, "_resampling_counter", 0) + 1
            )

            # Determine save directory
            if self.motion_log_dir is None:
                save_dir = os.getcwd()
            else:
                save_dir = self.motion_log_dir

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Generate filename
            mode_str = "eval" if eval else "train"
            indexing_str = (
                "submotion" if self.use_sub_motion_indexing else "motion"
            )
            filename = f"{mode_str}_{indexing_str}_sampling_info_iter_{sampling_data['resampling_iteration']:06d}.json"
            filepath = os.path.join(save_dir, filename)

            # Save to JSON file
            with open(filepath, "w") as f:
                json.dump(sampling_data, f, indent=2)

            logger.debug(f"Motion sampling info dumped to: {filepath}")

            # Also save a "latest" version for easy access
            latest_filename = (
                f"latest_{mode_str}_{indexing_str}_sampling_info.json"
            )
            latest_filepath = os.path.join(save_dir, latest_filename)
            with open(latest_filepath, "w") as f:
                json.dump(sampling_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to dump motion sampling info: {e}")

    def update_motion_score(self, motion_id: int, score: float):
        """Update the score for a single motion id.

        Args:
            motion_id (int): The motion id to update
            score (float): The new score value
        """
        if 0 <= motion_id < len(self.motion_scores):
            self.motion_scores[motion_id] = score
        else:
            logger.warning(
                f"Invalid motion_id {motion_id}, should be in [0, {len(self.motion_scores) - 1}]"
            )

    def update_motion_scores_batch(
        self, motion_ids: torch.Tensor, scores: torch.Tensor
    ):
        """Update scores for multiple motion ids.

        Args:
            motion_ids (torch.Tensor): Tensor of motion ids to update
            scores (torch.Tensor): Tensor of corresponding scores
        """
        assert len(motion_ids) == len(scores), (
            "motion_ids and scores must have the same length"
        )

        # Filter valid motion ids
        valid_mask = (motion_ids >= 0) & (motion_ids < len(self.motion_scores))
        valid_motion_ids = motion_ids[valid_mask]
        valid_scores = scores[valid_mask]

        if len(valid_motion_ids) < len(motion_ids):
            logger.warning(
                f"Filtered out {len(motion_ids) - len(valid_motion_ids)} invalid motion ids"
            )

        self.motion_scores[valid_motion_ids] = valid_scores

    def get_motion_score(self, motion_id: int) -> float:
        """Get the score for a specific motion id.

        Args:
            motion_id (int): The motion id to query

        Returns:
            float: The score for the motion
        """
        if 0 <= motion_id < len(self.motion_scores):
            return self.motion_scores[motion_id].item()
        else:
            logger.warning(f"Invalid motion_id {motion_id}")
            return 0.0

    def get_motion_scores(self) -> torch.Tensor:
        """Get all motion scores.

        Returns:
            torch.Tensor: Tensor containing all motion scores
        """
        return self.motion_scores.clone()

    def update_sub_motion_score(self, sub_motion_id: int, score: float):
        """Update the score for a single sub-motion id.

        Args:
            sub_motion_id (int): The sub-motion id to update
            score (float): The new score value
        """
        if not self.use_sub_motion_indexing:
            logger.warning(
                "Sub-motion indexing is disabled. Use update_motion_score instead."
            )
            return

        if 0 <= sub_motion_id < len(self.sub_motion_scores):
            self.sub_motion_scores[sub_motion_id] = score
        else:
            logger.warning(
                f"Invalid sub_motion_id {sub_motion_id}, should be in [0, {len(self.sub_motion_scores) - 1}]"
            )

    def update_sub_motion_scores_batch(
        self, sub_motion_ids: torch.Tensor, scores: torch.Tensor
    ):
        """Update scores for multiple sub-motion ids.

        Args:
            sub_motion_ids (torch.Tensor): Tensor of sub-motion ids to update
            scores (torch.Tensor): Tensor of corresponding scores
        """
        if not self.use_sub_motion_indexing:
            logger.warning(
                "Sub-motion indexing is disabled. Use update_motion_scores_batch instead."
            )
            return

        assert len(sub_motion_ids) == len(scores), (
            "sub_motion_ids and scores must have the same length"
        )

        # Filter valid sub-motion ids
        valid_mask = (sub_motion_ids >= 0) & (
            sub_motion_ids < len(self.sub_motion_scores)
        )
        valid_sub_motion_ids = sub_motion_ids[valid_mask]
        valid_scores = scores[valid_mask]

        if len(valid_sub_motion_ids) < len(sub_motion_ids):
            logger.warning(
                f"Filtered out {len(sub_motion_ids) - len(valid_sub_motion_ids)} invalid sub-motion ids"
            )

        self.sub_motion_scores[valid_sub_motion_ids] = valid_scores

    def get_sub_motion_score(self, sub_motion_id: int) -> float:
        """Get the score for a specific sub-motion id.

        Args:
            sub_motion_id (int): The sub-motion id to query

        Returns:
            float: The score for the sub-motion
        """
        if not self.use_sub_motion_indexing:
            logger.warning(
                "Sub-motion indexing is disabled. Use get_motion_score instead."
            )
            return 0.0

        if 0 <= sub_motion_id < len(self.sub_motion_scores):
            return self.sub_motion_scores[sub_motion_id].item()
        else:
            logger.warning(f"Invalid sub_motion_id {sub_motion_id}")
            return 0.0

    def get_sub_motion_scores(self) -> torch.Tensor:
        """Get all sub-motion scores.

        Returns:
            torch.Tensor: Tensor containing all sub-motion scores
        """
        if not self.use_sub_motion_indexing:
            logger.warning(
                "Sub-motion indexing is disabled. Returning motion scores instead."
            )
            return self.motion_scores.clone()
        return self.sub_motion_scores.clone()

    def _update_windowed_sample_counts(self, motion_id: int):
        """Update windowed sample counts using circular buffer."""
        # Remove old sample from window if buffer is full
        if self.motion_sample_buffer[self.buffer_ptr] != -1:
            old_motion_id = self.motion_sample_buffer[self.buffer_ptr]
            if 0 <= old_motion_id < len(self.motion_windowed_sample_counts):
                self.motion_windowed_sample_counts[old_motion_id] -= 1
        else:
            self.total_windowed_samples += 1

        # Add new sample to window
        self.motion_sample_buffer[self.buffer_ptr] = motion_id
        self.motion_windowed_sample_counts[motion_id] += 1

        # Update buffer pointer
        self.buffer_ptr = (self.buffer_ptr + 1) % self.ucb_window_size
        if self.buffer_ptr == 0:
            self.buffer_filled = True

    def _reinitialize_windowed_tracking(self, new_window_size: int):
        """Reinitialize windowed tracking with new window size."""
        self.ucb_window_size = new_window_size
        self.motion_windowed_sample_counts.zero_()
        self.motion_sample_buffer = torch.full(
            (new_window_size,), -1, dtype=torch.long
        )
        self.buffer_ptr = 0
        self.buffer_filled = False
        self.total_windowed_samples = 0

        # For sub-motions if enabled
        if self.use_sub_motion_indexing and hasattr(
            self, "sub_motion_windowed_sample_counts"
        ):
            self.sub_motion_windowed_sample_counts.zero_()
            self.sub_motion_sample_buffer = torch.full(
                (new_window_size,), -1, dtype=torch.long
            )
            self.sub_motion_buffer_ptr = 0
            self.sub_motion_buffer_filled = False
            self.sub_motion_total_windowed_samples = 0

    def update_motion_score_with_td_error(
        self,
        motion_id: int,
        raw_td_error: float,
        momentum: float = 0.9,
        use_ucb: bool = True,
        ucb_confidence: float = 1.0,
        ucb_window_size: int = 1000,
    ):
        """Update motion score using TD error with optional Windowed-UCB curriculum learning.

        Args:
            motion_id (int): Motion id to update
            raw_td_error (float): Raw TD error (can be positive or negative)
            momentum (float): Momentum for score updates
            use_ucb (bool): Whether to use UCB-style scoring
            ucb_confidence (float): UCB confidence parameter
            ucb_window_size (int): Size of sliding window for sample counting
        """
        if not (0 <= motion_id < len(self.motion_scores)):
            logger.warning(f"Invalid motion_id {motion_id}")
            return

        # Update configuration if needed
        if (
            hasattr(self, "ucb_window_size")
            and self.ucb_window_size != ucb_window_size
        ):
            self._reinitialize_windowed_tracking(ucb_window_size)
        self.ucb_window_size = ucb_window_size

        # Update sample counts (both all-time and windowed)
        self.motion_sample_counts[motion_id] += 1
        self._update_windowed_sample_counts(motion_id)

        # Update EMA of raw TD error for logging. This smoothed value represents
        # the average TD error magnitude.
        current_ema = self.motion_td_ema[motion_id].item()
        new_ema = (
            self.td_ema_alpha * raw_td_error
            + (1.0 - self.td_ema_alpha) * current_ema
        )
        self.motion_td_ema[motion_id] = new_ema

        # Update TD error variance EMA
        current_var_ema = self.motion_td_variance_ema[motion_id].item()
        td_error_squared = raw_td_error**2
        new_var_ema = (
            self.td_ema_alpha * td_error_squared
            + (1.0 - self.td_ema_alpha) * current_var_ema
        )
        self.motion_td_variance_ema[motion_id] = new_var_ema

        if use_ucb:
            # Windowed-UCB scoring: mean_td_error + confidence_bonus
            abs_td_error = abs(
                new_ema
            )  # Use temporally-decayed EMA for mean estimate

            # Confidence bonus based on WINDOWED sample count
            windowed_total_samples = max(1, self.total_windowed_samples)
            windowed_sample_count = max(
                1, self.motion_windowed_sample_counts[motion_id].item()
            )

            # Use windowed counts for UCB confidence bonus
            confidence_bonus = (
                ucb_confidence
                * torch.sqrt(
                    torch.log(
                        torch.tensor(
                            windowed_total_samples, dtype=torch.float32
                        )
                    )
                    / windowed_sample_count
                ).item()
            )

            ucb_score = abs_td_error + confidence_bonus
            current_score = self.motion_scores[motion_id].item()
            new_score = momentum * current_score + (1.0 - momentum) * ucb_score
        else:
            # Original approach: use absolute TD error for actual scoring
            abs_td_error = abs(raw_td_error)
            current_score = self.motion_scores[motion_id].item()
            new_score = (
                momentum * current_score + (1.0 - momentum) * abs_td_error
            )

        self.motion_scores[motion_id] = max(
            0.01, new_score
        )  # Ensure minimum score

    def _update_windowed_sub_motion_sample_counts(self, sub_motion_id: int):
        """Update windowed sub-motion sample counts using circular buffer."""
        # Initialize if not exists
        if not hasattr(self, "sub_motion_sample_buffer"):
            self.sub_motion_sample_buffer = torch.full(
                (self.ucb_window_size,), -1, dtype=torch.long
            )
            self.sub_motion_buffer_ptr = 0
            self.sub_motion_buffer_filled = False
            self.sub_motion_total_windowed_samples = 0

        # Remove old sample from window if buffer is full
        if self.sub_motion_sample_buffer[self.sub_motion_buffer_ptr] != -1:
            old_sub_motion_id = self.sub_motion_sample_buffer[
                self.sub_motion_buffer_ptr
            ]
            if (
                0
                <= old_sub_motion_id
                < len(self.sub_motion_windowed_sample_counts)
            ):
                self.sub_motion_windowed_sample_counts[old_sub_motion_id] -= 1
        else:
            self.sub_motion_total_windowed_samples += 1

        # Add new sample to window
        self.sub_motion_sample_buffer[self.sub_motion_buffer_ptr] = (
            sub_motion_id
        )
        self.sub_motion_windowed_sample_counts[sub_motion_id] += 1

        # Update buffer pointer
        self.sub_motion_buffer_ptr = (
            self.sub_motion_buffer_ptr + 1
        ) % self.ucb_window_size
        if self.sub_motion_buffer_ptr == 0:
            self.sub_motion_buffer_filled = True

    def update_sub_motion_score_with_td_error(
        self,
        sub_motion_id: int,
        raw_td_error: float,
        momentum: float = 0.9,
        use_ucb: bool = True,
        ucb_confidence: float = 1.0,
        ucb_window_size: int = 1000,
    ):
        """Update sub-motion score using TD error with optional Windowed-UCB curriculum learning.

        Args:
            sub_motion_id (int): Sub-motion id to update
            raw_td_error (float): Raw TD error (can be positive or negative)
            momentum (float): Momentum for score updates
            use_ucb (bool): Whether to use UCB-style scoring
            ucb_confidence (float): UCB confidence parameter
            ucb_window_size (int): Size of sliding window for sample counting
        """
        if not self.use_sub_motion_indexing:
            logger.warning(
                "Sub-motion indexing is disabled. Use update_motion_score_with_td_error instead."
            )
            return

        if not (0 <= sub_motion_id < len(self.sub_motion_scores)):
            logger.warning(f"Invalid sub_motion_id {sub_motion_id}")
            return

        # Update configuration if needed
        self.ucb_window_size = ucb_window_size

        # Initialize windowed tracking if not exists (for backward compatibility)
        if not hasattr(self, "sub_motion_sample_counts"):
            self.sub_motion_sample_counts = torch.zeros(
                self.num_sub_motions, dtype=torch.long
            )
            self.sub_motion_td_variance_ema = torch.zeros(
                self.num_sub_motions, dtype=torch.float32
            )
            self.sub_motion_windowed_sample_counts = torch.zeros(
                self.num_sub_motions, dtype=torch.long
            )

        # Update sample counts (both all-time and windowed)
        self.sub_motion_sample_counts[sub_motion_id] += 1
        self._update_windowed_sub_motion_sample_counts(sub_motion_id)

        # Update EMA of raw TD error for logging. This smoothed value represents
        # the average TD error magnitude.
        current_ema = self.sub_motion_td_ema[sub_motion_id].item()
        new_ema = (
            self.td_ema_alpha * raw_td_error
            + (1.0 - self.td_ema_alpha) * current_ema
        )
        self.sub_motion_td_ema[sub_motion_id] = new_ema

        # Update TD error variance EMA
        if hasattr(self, "sub_motion_td_variance_ema"):
            current_var_ema = self.sub_motion_td_variance_ema[
                sub_motion_id
            ].item()
            td_error_squared = raw_td_error**2
            new_var_ema = (
                self.td_ema_alpha * td_error_squared
                + (1.0 - self.td_ema_alpha) * current_var_ema
            )
            self.sub_motion_td_variance_ema[sub_motion_id] = new_var_ema

        if use_ucb and hasattr(self, "sub_motion_sample_counts"):
            # Windowed-UCB scoring: mean_td_error + confidence_bonus
            abs_td_error = abs(
                new_ema
            )  # Use temporally-decayed EMA for mean estimate

            # Confidence bonus based on WINDOWED sample count
            windowed_total_samples = max(
                1, getattr(self, "sub_motion_total_windowed_samples", 1)
            )
            windowed_sample_count = max(
                1, self.sub_motion_windowed_sample_counts[sub_motion_id].item()
            )

            # Use windowed counts for UCB confidence bonus
            confidence_bonus = (
                ucb_confidence
                * torch.sqrt(
                    torch.log(
                        torch.tensor(
                            windowed_total_samples, dtype=torch.float32
                        )
                    )
                    / windowed_sample_count
                ).item()
            )

            ucb_score = abs_td_error + confidence_bonus
            current_score = self.sub_motion_scores[sub_motion_id].item()
            new_score = momentum * current_score + (1.0 - momentum) * ucb_score
        else:
            # Original approach: use absolute TD error for actual scoring
            abs_td_error = abs(raw_td_error)
            current_score = self.sub_motion_scores[sub_motion_id].item()
            new_score = (
                momentum * current_score + (1.0 - momentum) * abs_td_error
            )

        self.sub_motion_scores[sub_motion_id] = max(
            0.01, new_score
        )  # Ensure minimum score

    def get_td_error_statistics(self) -> dict:
        """Get basic TD error statistics for logging.

        Returns:
            dict: Simple statistics about TD errors
        """
        if self.use_sub_motion_indexing:
            td_emas = self.sub_motion_td_ema
            scores = self.sub_motion_scores
            prefix = "sub_motion"
        else:
            td_emas = self.motion_td_ema
            scores = self.motion_scores
            prefix = "motion"

        # Calculate basic statistics
        stats = {
            f"{prefix}_td_ema_statistics": {
                "mean": td_emas.mean().item(),
                "std": td_emas.std().item(),
                "min": td_emas.min().item(),
                "max": td_emas.max().item(),
                "positive_count": (td_emas > 0).sum().item(),
                "negative_count": (td_emas < 0).sum().item(),
            },
            f"{prefix}_score_statistics": {
                "mean": scores.mean().item(),
                "std": scores.std().item(),
                "min": scores.min().item(),
                "max": scores.max().item(),
            },
        }

        # Add UCB-related statistics if available
        if self.use_sub_motion_indexing and hasattr(
            self, "sub_motion_sample_counts"
        ):
            sample_counts = self.sub_motion_sample_counts
            windowed_sample_counts = getattr(
                self, "sub_motion_windowed_sample_counts", sample_counts
            )
            stats[f"{prefix}_ucb_statistics"] = {
                "total_samples": sample_counts.sum().item(),
                "mean_samples_per_motion": sample_counts.float().mean().item(),
                "std_samples_per_motion": sample_counts.float().std().item(),
                "min_samples": sample_counts.min().item(),
                "max_samples": sample_counts.max().item(),
                "unsampled_motions": (sample_counts == 0).sum().item(),
                # Windowed statistics
                "windowed_total_samples": getattr(
                    self, "sub_motion_total_windowed_samples", 0
                ),
                "windowed_mean_samples_per_motion": windowed_sample_counts.float()
                .mean()
                .item(),
                "windowed_std_samples_per_motion": windowed_sample_counts.float()
                .std()
                .item(),
                "windowed_unsampled_motions": (windowed_sample_counts == 0)
                .sum()
                .item(),
                "window_size": self.ucb_window_size,
            }
        elif not self.use_sub_motion_indexing and hasattr(
            self, "motion_sample_counts"
        ):
            sample_counts = self.motion_sample_counts
            windowed_sample_counts = getattr(
                self, "motion_windowed_sample_counts", sample_counts
            )
            stats[f"{prefix}_ucb_statistics"] = {
                "total_samples": sample_counts.sum().item(),
                "mean_samples_per_motion": sample_counts.float().mean().item(),
                "std_samples_per_motion": sample_counts.float().std().item(),
                "min_samples": sample_counts.min().item(),
                "max_samples": sample_counts.max().item(),
                "unsampled_motions": (sample_counts == 0).sum().item(),
                # Windowed statistics
                "windowed_total_samples": getattr(
                    self, "total_windowed_samples", 0
                ),
                "windowed_mean_samples_per_motion": windowed_sample_counts.float()
                .mean()
                .item(),
                "windowed_std_samples_per_motion": windowed_sample_counts.float()
                .std()
                .item(),
                "windowed_unsampled_motions": (windowed_sample_counts == 0)
                .sum()
                .item(),
                "window_size": self.ucb_window_size,
            }

        return stats

    def get_sub_motion_info(self, sub_motion_id: int) -> dict:
        """Get information about a specific sub-motion.

        Args:
            sub_motion_id (int): The sub-motion id to query

        Returns:
            dict: Information about the sub-motion including original motion id and frame range
        """
        if not self.use_sub_motion_indexing:
            logger.warning("Sub-motion indexing is disabled.")
            return {}

        if 0 <= sub_motion_id < len(self.sub_motion_infos):
            return self.sub_motion_infos[sub_motion_id].copy()
        else:
            logger.warning(f"Invalid sub_motion_id {sub_motion_id}")
            return {}

    def _sample_motion_ids_uniform(self, num_samples: int) -> torch.Tensor:
        """Sample motion ids uniformly."""
        if self.use_sub_motion_indexing:
            return self._sample_sub_motion_ids_uniform(num_samples)
        else:
            return torch.randint(0, len(self.motion_ids), (num_samples,))

    def _sample_motion_ids_weighted(self, num_samples: int) -> torch.Tensor:
        """Sample motion ids using linear weighted sampling based on scores."""
        if self.use_sub_motion_indexing:
            return self._sample_sub_motion_ids_weighted(num_samples)
        else:
            # Linear weighted sampling: combine base probability with score-based probability
            base_prob = self.sampling_base_probability / len(self.motion_ids)

            # Normalize scores to get score-based probabilities
            score_weights = self.motion_scores / torch.sum(self.motion_scores)
            score_prob = (1.0 - self.sampling_base_probability) * score_weights

            # Combine base and score probabilities
            final_probs = base_prob + score_prob

            # Sample according to the final probabilities
            # Ensure multinomial is executed on CPU to prevent GPU migration of curriculum tensors
            final_probs_cpu = final_probs.detach().cpu()
            sampled_indices = torch.multinomial(
                final_probs_cpu, num_samples, replacement=True
            )

            return sampled_indices

    def _sample_sub_motion_ids_uniform(self, num_samples: int) -> torch.Tensor:
        """Sample sub-motion ids uniformly."""
        return torch.randint(0, self.num_sub_motions, (num_samples,))

    def _sample_sub_motion_ids_weighted(
        self, num_samples: int
    ) -> torch.Tensor:
        """Sample sub-motion ids using linear weighted sampling based on scores.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Sampled sub-motion ids
        """
        # Linear weighted sampling: combine base probability with score-based probability
        base_prob = self.sampling_base_probability / self.num_sub_motions

        # Normalize scores to get score-based probabilities
        score_weights = self.sub_motion_scores / torch.sum(
            self.sub_motion_scores
        )
        score_prob = (1.0 - self.sampling_base_probability) * score_weights

        # Combine base and score probabilities
        final_probs = base_prob + score_prob

        # Sample according to the final probabilities
        # Ensure multinomial is executed on CPU to prevent GPU migration of curriculum tensors
        final_probs_cpu = final_probs.detach().cpu()
        sampled_indices = torch.multinomial(
            final_probs_cpu,
            num_samples,
            replacement=True,
        )

        return sampled_indices

    def _build_online_train_cache(
        self,
        cache_instance: OnlineMotionCache,
        motion_global_start_frames: torch.Tensor,
    ):
        if self.use_sub_motion_indexing:
            self._build_online_train_cache_with_sub_motions(
                cache_instance, motion_global_start_frames
            )
        else:
            self._build_online_train_cache_original(
                cache_instance, motion_global_start_frames
            )

    def _build_online_train_cache_original(
        self,
        cache_instance: OnlineMotionCache,
        motion_global_start_frames: torch.Tensor,
    ):
        """Original cache building logic for backward compatibility."""
        motion_ids = cache_instance.cached_motion_ids

        cache_instance["cached_motion_raw_num_frames"] = (
            self.get_motion_num_frames(motion_ids).clone()
        )
        cache_instance["cached_motion_global_start_frames"] = (
            motion_global_start_frames.clone()
        )
        cache_instance["cached_motion_global_end_frames"] = (
            cache_instance["cached_motion_global_start_frames"]
            + self.max_frame_length
        ).clamp(max=cache_instance["cached_motion_raw_num_frames"])

        # Use context manager for the entire cache building process
        with self.lmdb_handle.begin() as _:
            for i, motion_id in enumerate(motion_ids):
                key = self.motion_id2key[motion_id.item()]
                frame_slice = slice(
                    cache_instance.cached_motion_global_start_frames[i],
                    cache_instance.cached_motion_global_end_frames[i],
                )
                frame_slice_len = frame_slice.stop - frame_slice.start

                self._load_motion_data_to_cache(
                    cache_instance, i, key, frame_slice, frame_slice_len
                )

        # Move cache tensors to the specified device
        cache_instance.move_cache_to_device(self.cache_device)

    def _build_online_train_cache_with_sub_motions(
        self,
        cache_instance: OnlineMotionCache,
        motion_global_start_frames: torch.Tensor,
    ):
        """Cache building logic using sub-motion indexing."""
        sampled_sub_motion_ids = (
            cache_instance.cached_motion_ids
        )  # These are actually sub-motion IDs

        # Convert sub-motion IDs to original motion information
        original_motion_ids = []
        actual_start_frames = []
        actual_end_frames = []
        raw_num_frames = []

        for sub_motion_id in sampled_sub_motion_ids:
            sub_motion_info = self.sub_motion_infos[sub_motion_id.item()]
            original_motion_id = sub_motion_info["original_motion_id"]
            sub_start = sub_motion_info["sub_motion_start_frame"]
            sub_end = sub_motion_info["sub_motion_end_frame"]

            original_motion_ids.append(original_motion_id)
            actual_start_frames.append(sub_start)
            actual_end_frames.append(sub_end)

            # Get the actual length of the original motion
            original_motion_length = self.get_motion_num_frames(
                [original_motion_id]
            )[0].item()
            raw_num_frames.append(original_motion_length)

        cache_instance["cached_motion_raw_num_frames"] = torch.tensor(
            raw_num_frames, dtype=torch.long
        )
        cache_instance["cached_motion_global_start_frames"] = torch.tensor(
            actual_start_frames, dtype=torch.long
        )
        cache_instance["cached_motion_global_end_frames"] = torch.tensor(
            actual_end_frames, dtype=torch.long
        )

        # Use context manager for the entire cache building process
        with self.lmdb_handle.begin() as _:
            for i, sub_motion_id in enumerate(sampled_sub_motion_ids):
                sub_motion_info = self.sub_motion_infos[sub_motion_id.item()]
                key = sub_motion_info["original_motion_key"]
                frame_slice = slice(
                    sub_motion_info["sub_motion_start_frame"],
                    sub_motion_info["sub_motion_end_frame"],
                )
                frame_slice_len = frame_slice.stop - frame_slice.start

                self._load_motion_data_to_cache(
                    cache_instance, i, key, frame_slice, frame_slice_len
                )

        # Move cache tensors to the specified device
        cache_instance.move_cache_to_device(self.cache_device)

    def _load_motion_data_to_cache(
        self, cache_instance, i, key, frame_slice, frame_slice_len
    ):
        """Helper method to load motion data into cache."""
        # Load and cache dof positions
        dof_pos = read_motion_array(
            self.lmdb_handle,
            key,
            "dof_pos",
            slices=frame_slice,
        )
        cache_instance.dof_pos[i, :frame_slice_len, :] = torch.from_numpy(
            dof_pos.copy()
        )

        # Load and cache dof velocities
        dof_vels = read_motion_array(
            self.lmdb_handle,
            key,
            "dof_vels",
            slices=frame_slice,
        )
        cache_instance.dof_vels[i, :frame_slice_len, :] = torch.from_numpy(
            dof_vels.copy()
        )

        # Load and cache body translations
        global_body_translation = read_motion_array(
            self.lmdb_handle,
            key,
            "global_translation",
            slices=frame_slice,
        )
        cache_instance.global_body_translation[i, :frame_slice_len, :] = (
            torch.from_numpy(global_body_translation.copy())
        )

        # Load and cache body rotations
        global_body_rotation = read_motion_array(
            self.lmdb_handle,
            key,
            "global_rotation_quat",
            slices=frame_slice,
        )
        cache_instance.global_body_rotation[i, :frame_slice_len, :] = (
            torch.from_numpy(global_body_rotation.copy())
        )

        # Load and cache body velocities
        global_body_velocity = read_motion_array(
            self.lmdb_handle,
            key,
            "global_velocity",
            slices=frame_slice,
        )
        cache_instance.global_body_velocity[i, :frame_slice_len, :] = (
            torch.from_numpy(global_body_velocity.copy())
        )

        # Load and cache body angular velocities
        global_body_angular_velocity = read_motion_array(
            self.lmdb_handle,
            key,
            "global_angular_velocity",
            slices=frame_slice,
        )
        cache_instance.global_body_angular_velocity[i, :frame_slice_len, :] = (
            torch.from_numpy(global_body_angular_velocity.copy())
        )

        # Load and cache extended body translations
        global_body_translation_extend = read_motion_array(
            self.lmdb_handle,
            key,
            "global_translation_extend",
            slices=frame_slice,
        )
        if global_body_translation_extend is not None:
            cache_instance.global_body_translation_extend[
                i, :frame_slice_len, :
            ] = torch.from_numpy(global_body_translation_extend.copy())

            # Load and cache extended body rotations
            global_body_rotation_extend = read_motion_array(
                self.lmdb_handle,
                key,
                "global_rotation_quat_extend",
                slices=frame_slice,
            )
            cache_instance.global_body_rotation_extend[
                i, :frame_slice_len, :
            ] = torch.from_numpy(global_body_rotation_extend.copy())

            # Load and cache extended body velocities
            global_body_velocity_extend = read_motion_array(
                self.lmdb_handle,
                key,
                "global_velocity_extend",
                slices=frame_slice,
            )
            cache_instance.global_body_velocity_extend[
                i, :frame_slice_len, :
            ] = torch.from_numpy(global_body_velocity_extend.copy())

            # Load and cache extended body angular velocities
            global_body_angular_velocity_extend = read_motion_array(
                self.lmdb_handle,
                key,
                "global_angular_velocity_extend",
                slices=frame_slice,
            )
            cache_instance.global_body_angular_velocity_extend[
                i, :frame_slice_len, :
            ] = torch.from_numpy(global_body_angular_velocity_extend.copy())

        frame_flag = np.ones(frame_slice_len, dtype=np.int64)
        frame_flag[0] = 0
        frame_flag[-1] = 2
        cache_instance.frame_flag[i, :frame_slice_len] = torch.from_numpy(
            frame_flag.copy()
        ).long()

        # Load and cache local body rotations
        local_body_rotation = read_motion_array(
            self.lmdb_handle,
            key,
            "local_rotation_quat",
            slices=frame_slice,
        )
        cache_instance.local_body_rotation[i, :frame_slice_len, :] = (
            torch.from_numpy(local_body_rotation.copy())
        )

    def _build_sub_motion_index(self):
        """Build sub-motion indexing system by splitting long motions into consecutive sub-clips.

        This method creates a mapping from sub-motion IDs to the original motion information,
        enabling curriculum learning at a finer granularity than full motions.
        """
        logger.info("Building sub-motion indexing system...")

        self.sub_motion_infos = []
        sub_motion_id = 0

        # Get motion lengths for all motions at once for efficiency
        motion_lengths = self.get_motion_num_frames(self.motion_ids)

        for motion_id, motion_length in zip(self.motion_ids, motion_lengths):
            motion_key = self.motion_id2key[motion_id]
            motion_length = (
                motion_length.item()
                if isinstance(motion_length, torch.Tensor)
                else motion_length
            )

            # Calculate step size considering overlap
            step_size = max(
                1, self.max_frame_length - self.sub_motion_overlap_frames
            )

            # Split motion into sub-motions
            current_start = 0
            while current_start < motion_length:
                current_end = min(
                    current_start + self.max_frame_length, motion_length
                )
                sub_motion_length = current_end - current_start

                # Only create sub-motion if it meets minimum length requirement
                if sub_motion_length >= self.min_frame_length:
                    sub_motion_info = {
                        "sub_motion_id": sub_motion_id,
                        "original_motion_id": motion_id,
                        "original_motion_key": motion_key,
                        "sub_motion_start_frame": current_start,
                        "sub_motion_end_frame": current_end,
                        "sub_motion_length": sub_motion_length,
                    }
                    self.sub_motion_infos.append(sub_motion_info)
                    sub_motion_id += 1

                # Move to next sub-motion
                current_start += step_size

                # If we've reached the end, break
                if current_end >= motion_length:
                    break

        self.num_sub_motions = len(self.sub_motion_infos)

        # Initialize sub-motion scores - start with uniform scores of ones
        self.sub_motion_scores = torch.ones(
            self.num_sub_motions, dtype=torch.float32
        )

        # TD error tracking with EMA for logging
        self.sub_motion_td_ema = torch.zeros(
            self.num_sub_motions, dtype=torch.float32
        )  # EMA of raw TD errors for logging
        self.td_ema_alpha = 0.1  # EMA smoothing factor

        # Windowed-UCB curriculum learning for sub-motions
        self.sub_motion_sample_counts = torch.zeros(
            self.num_sub_motions, dtype=torch.long
        )  # Track sampling frequency (all-time)
        self.sub_motion_windowed_sample_counts = torch.zeros(
            self.num_sub_motions, dtype=torch.long
        )  # Track sampling frequency (windowed)
        self.sub_motion_td_variance_ema = torch.zeros(
            self.num_sub_motions, dtype=torch.float32
        )  # Track TD error variance

        # Circular buffer for windowed sub-motion sample tracking
        self.sub_motion_sample_buffer = torch.full(
            (self.ucb_window_size,), -1, dtype=torch.long
        )  # -1 indicates empty slot
        self.sub_motion_buffer_ptr = 0  # Current position in circular buffer
        self.sub_motion_buffer_filled = (
            False  # Whether buffer has been filled once
        )
        self.sub_motion_total_windowed_samples = (
            0  # Total samples in current window
        )

        logger.info(
            f"Sub-motion indexing complete: "
            f"{len(self.motion_ids)} original motions -> "
            f"{self.num_sub_motions} sub-motions "
            f"(avg {self.num_sub_motions / len(self.motion_ids):.1f} sub-motions per original motion)"
        )

    def _eval_preallocation(self) -> List[Dict[str, Union[int, str]]]:
        allocation_schedule = []
        logger.info(
            f"Pre-calculating evaluation allocation for "
            f"{len(self.eval_motion_keys)} motion keys..."
        )

        # Convert keys to IDs first
        eval_motion_ids_list = []
        valid_eval_motion_keys = []
        for key in self.eval_motion_keys:
            if key in self.motion_key2id:
                eval_motion_ids_list.append(self.motion_key2id[key])
                valid_eval_motion_keys.append(key)

        if not eval_motion_ids_list:
            return allocation_schedule

        # Fetch all required frame lengths at once for efficiency
        # get_motion_num_frames takes List[int] and returns List[int]
        try:
            # Pass IDs to get_motion_num_frames
            motion_lengths = self.get_motion_num_frames(
                torch.tensor(eval_motion_ids_list)
            )
            if len(motion_lengths) != len(eval_motion_ids_list):
                logger.error(
                    f"Mismatch in length returned by get_motion_num_frames. "
                    f"Expected {len(eval_motion_ids_list)}, "
                    f"got {len(motion_lengths)}"
                )
                return []
            motion_id_to_length = {
                id: length
                for id, length in zip(eval_motion_ids_list, motion_lengths)  # noqa: B905
            }
        except Exception as e:
            logger.error(f"Error fetching motion lengths: {e}")
            return []

        for motion_id in track(
            eval_motion_ids_list, description="Processing eval motions"
        ):
            motion_key = self.motion_id2key.get(
                motion_id
            )  # Use .get for safety
            if motion_key is None:
                continue

            num_frames = motion_id_to_length.get(
                motion_id
            ).item()  # Use .get for safety

            if num_frames is None:
                continue

            if num_frames <= 0:  # Check for <= 0
                continue

            current_start_frame = 0
            while current_start_frame < num_frames:
                clip_end_frame = min(
                    current_start_frame + self.max_frame_length, num_frames
                )
                clip_length = clip_end_frame - current_start_frame

                # filter out the clip length that is too short
                if clip_length < self.min_frame_length:
                    break  # Avoid infinite loop

                allocation_schedule.append(
                    {
                        "motion_key": motion_key,
                        "motion_id": motion_id,
                        "start_frame": current_start_frame,
                        "end_frame": clip_end_frame,
                        "length": clip_length,
                    }
                )
                current_start_frame = clip_end_frame

        logger.info(f"Generated {len(allocation_schedule)} evaluation clips.")
        return allocation_schedule

    def load_next_eval_batch(self, num_envs_to_load: int):
        if not self.eval_allocation_schedule:
            logger.warning("Evaluation allocation schedule is empty.")
            return False  # Cannot be the last batch if schedule is empty

        num_total_clips = len(self.eval_allocation_schedule)
        if num_envs_to_load <= 0:
            logger.warning(
                f"Requested to load {num_envs_to_load} envs. Skipping."
            )
            return False  # Cannot be the last batch if loading zero

        is_last_batch = (
            self.eval_schedule_idx + num_envs_to_load
        ) >= num_total_clips

        # Ensure cache is reset before loading new data, but keep original size
        self.cache.reset()

        # Determine clip indices to load, handling wrap-around
        indices_to_load = [
            (self.eval_schedule_idx + i) % num_total_clips
            for i in range(num_envs_to_load)
        ]
        selected_clips = [
            self.eval_allocation_schedule[idx] for idx in indices_to_load
        ]
        self.cache.cached_clip_info = selected_clips
        # Prepare metadata tensors for the cache
        batch_motion_ids = torch.tensor(
            [clip["motion_id"] for clip in selected_clips], dtype=torch.long
        )
        batch_start_frames = torch.tensor(
            [clip["start_frame"] for clip in selected_clips], dtype=torch.long
        )
        batch_clip_lengths = torch.tensor(
            [clip["length"] for clip in selected_clips], dtype=torch.long
        )

        original_motion_ids = (
            batch_motion_ids.tolist()
        )  # Need list for get_motion_num_frames
        original_num_frames = self.get_motion_num_frames(original_motion_ids)

        # Update cache metadata (use .clone() for safety)
        self.cache.cached_motion_ids = batch_motion_ids.clone()
        self.cache.cached_motion_global_start_frames = (
            batch_start_frames.clone()
        )
        self.cache.cached_motion_global_end_frames = (
            self.cache.cached_motion_global_start_frames + batch_clip_lengths
        )
        # For eval, raw_num_frames in cache refers to the clip length loaded
        self.cache.cached_motion_raw_num_frames = batch_clip_lengths.clone()
        self.cache.cached_motion_original_num_frames = (
            original_num_frames.clone()
        )

        # Load motion data clip by clip
        with self.lmdb_handle.begin() as _:
            for i, clip in enumerate(selected_clips):
                motion_key = clip["motion_key"]
                frame_slice = slice(clip["start_frame"], clip["end_frame"])
                frame_slice_len = clip["length"]

                if frame_slice_len <= 0:
                    logger.warning(
                        f"clip {i} for motion {motion_key} has zero length. "
                        f"Skipping data load for this clip."
                    )
                    continue

                # --- Load all data arrays for the current clip ---
                dof_pos = read_motion_array(
                    self.lmdb_handle, motion_key, "dof_pos", slices=frame_slice
                )
                dof_vels = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "dof_vels",
                    slices=frame_slice,
                )
                global_body_translation = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_translation",
                    slices=frame_slice,
                )
                global_body_rotation = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_rotation_quat",
                    slices=frame_slice,
                )
                global_body_velocity = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_velocity",
                    slices=frame_slice,
                )
                global_body_angular_velocity = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_angular_velocity",
                    slices=frame_slice,
                )
                local_body_rotation = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "local_rotation_quat",
                    slices=frame_slice,
                )

                # Extended bodies (check if they exist for this motion)
                global_body_translation_extend = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_translation_extend",
                    slices=frame_slice,
                )
                global_body_rotation_extend = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_rotation_quat_extend",
                    slices=frame_slice,
                )
                global_body_velocity_extend = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_velocity_extend",
                    slices=frame_slice,
                )
                global_body_angular_velocity_extend = read_motion_array(
                    self.lmdb_handle,
                    motion_key,
                    "global_angular_velocity_extend",
                    slices=frame_slice,
                )

                # --- Fill cache tensors ---
                # Make sure tensors are not None before filling
                if dof_pos is not None:
                    self.cache.dof_pos[i, :frame_slice_len, :] = (
                        torch.from_numpy(dof_pos.copy())
                    )
                if dof_vels is not None:
                    self.cache.dof_vels[i, :frame_slice_len, :] = (
                        torch.from_numpy(dof_vels.copy())
                    )
                if global_body_translation is not None:
                    self.cache.global_body_translation[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_translation.copy())
                if global_body_rotation is not None:
                    self.cache.global_body_rotation[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_rotation.copy())
                if global_body_velocity is not None:
                    self.cache.global_body_velocity[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_velocity.copy())
                if global_body_angular_velocity is not None:
                    self.cache.global_body_angular_velocity[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_angular_velocity.copy())
                if local_body_rotation is not None:
                    self.cache.local_body_rotation[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(local_body_rotation.copy())

                # Fill extended body data if available
                if global_body_translation_extend is not None:
                    self.cache.global_body_translation_extend[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_translation_extend.copy())
                if global_body_rotation_extend is not None:
                    self.cache.global_body_rotation_extend[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_rotation_extend.copy())
                if global_body_velocity_extend is not None:
                    self.cache.global_body_velocity_extend[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(global_body_velocity_extend.copy())
                if global_body_angular_velocity_extend is not None:
                    self.cache.global_body_angular_velocity_extend[
                        i, :frame_slice_len, :, :
                    ] = torch.from_numpy(
                        global_body_angular_velocity_extend.copy()
                    )

                # Build frame flag (0=start, 1=middle, 2=end of clip)
                frame_flag = np.ones(frame_slice_len, dtype=np.int64)
                if frame_slice_len > 0:
                    frame_flag[0] = 0
                    frame_flag[-1] = 2
                self.cache.frame_flag[i, :frame_slice_len] = torch.from_numpy(
                    frame_flag.copy()
                ).long()

        # Move all updated cache data to the target device
        self.cache.move_cache_to_device(self.cache_device)

        # Update the schedule index for the next call
        self.eval_schedule_idx = (
            self.eval_schedule_idx + num_envs_to_load
        ) % num_total_clips

        return is_last_batch

    def export_motion_clip(self, motion_key: str) -> Dict[str, np.ndarray]:
        if motion_key not in self.motion_key2id:
            logger.error(
                f"Motion key {motion_key} not found in motion library."
            )
            return {}

        with self.lmdb_handle.begin() as txn:
            # Get number of frames for the motion
            metadata_bytes = txn.get(f"motion/{motion_key}/metadata".encode())
            if metadata_bytes is None:
                logger.error(
                    f"Metadata not found for motion key {motion_key}."
                )
                return {}
            metadata = pickle.loads(metadata_bytes)
            num_frames = metadata["num_frames"]

            if num_frames == 0:
                logger.warning(f"Motion key {motion_key} has 0 frames.")
                return {}

            # LMDB array names we need to fetch
            lmdb_keys_to_fetch = [
                "dof_pos",
                "dof_vels",
                "global_translation",
                "global_rotation_quat",
                "global_velocity",
                "global_angular_velocity",
                "global_translation_extend",
                "global_rotation_quat_extend",
                "global_velocity_extend",
                "global_angular_velocity_extend",
            ]

            raw_motion_arrays = {}
            for lmdb_key in lmdb_keys_to_fetch:
                array_data = read_motion_array(
                    self.lmdb_handle, motion_key, lmdb_key, slices=None
                )
                raw_motion_arrays[lmdb_key] = (
                    array_data  # array_data can be None
                )
                if (
                    array_data is not None
                    and array_data.shape[0] != num_frames
                ):
                    logger.warning(
                        f"Mismatch in frame count for {motion_key}/{lmdb_key}."
                        f"Expected {num_frames}, got {array_data.shape[0]}. "
                        "Data might be partial or inconsistent."
                    )

            output_dict = {}

            # DOF data
            dof_pos_data = raw_motion_arrays.get("dof_pos")
            if dof_pos_data is not None:
                output_dict["dof_pos"] = dof_pos_data

            dof_vels_data = raw_motion_arrays.get("dof_vels")
            if dof_vels_data is not None:
                output_dict["dof_vel"] = dof_vels_data

            # Root and Rigid Body (rb) data
            global_translation_data = raw_motion_arrays.get(
                "global_translation"
            )
            if global_translation_data is not None:
                output_dict["rg_pos"] = global_translation_data
                output_dict["root_pos"] = global_translation_data[:, 0, :]

            global_rotation_data = raw_motion_arrays.get(
                "global_rotation_quat"
            )
            if global_rotation_data is not None:
                output_dict["rb_rot"] = global_rotation_data
                output_dict["root_rot"] = global_rotation_data[:, 0, :]

            global_velocity_data = raw_motion_arrays.get("global_velocity")
            if global_velocity_data is not None:
                output_dict["body_vel"] = global_velocity_data
                output_dict["root_vel"] = global_velocity_data[:, 0, :]

            global_angular_velocity_data = raw_motion_arrays.get(
                "global_angular_velocity"
            )
            if global_angular_velocity_data is not None:
                output_dict["body_ang_vel"] = global_angular_velocity_data
                output_dict["root_ang_vel"] = global_angular_velocity_data[
                    :, 0, :
                ]

            # Extended Rigid Body (rg_pos_t, etc.) data
            g_trans_ext_data = raw_motion_arrays.get(
                "global_translation_extend"
            )
            if g_trans_ext_data is not None:
                output_dict["rg_pos_t"] = g_trans_ext_data

            g_rot_ext_data = raw_motion_arrays.get(
                "global_rotation_quat_extend"
            )
            if g_rot_ext_data is not None:
                output_dict["rg_rot_t"] = g_rot_ext_data

            g_vel_ext_data = raw_motion_arrays.get("global_velocity_extend")
            if g_vel_ext_data is not None:
                output_dict["body_vel_t"] = g_vel_ext_data

            g_ang_vel_ext_data = raw_motion_arrays.get(
                "global_angular_velocity_extend"
            )
            if g_ang_vel_ext_data is not None:
                output_dict["body_ang_vel_t"] = g_ang_vel_ext_data

            # Frame flag
            frame_flag_np = np.ones(num_frames, dtype=np.int64)
            if num_frames > 0:  # Should always be true due to earlier check
                frame_flag_np[0] = 0
            if num_frames > 1:
                frame_flag_np[-1] = 2
            output_dict["frame_flag"] = frame_flag_np

            # Determine and add FPS
            fps_value = 0.0
            motion_fps_stored = metadata.get("fps")
            motion_dt = metadata.get("dt")
            wallclock_len = metadata.get("wallclock_len")

            if (
                motion_fps_stored is not None
                and isinstance(motion_fps_stored, (int, float))
                and motion_fps_stored > 0
            ):
                fps_value = float(motion_fps_stored)
            elif (
                motion_dt is not None
                and isinstance(motion_dt, (int, float))
                and motion_dt > 0
            ):
                fps_value = 1.0 / float(motion_dt)
            elif (
                wallclock_len is not None
                and isinstance(wallclock_len, (int, float))
                and wallclock_len > 0
                and num_frames > 0
            ):
                fps_value = num_frames / float(wallclock_len)
            else:
                logger.warning(
                    f"Could not reliably determine FPS for motion {motion_key}"
                    f"from metadata "
                )
            output_dict["fps"] = int(fps_value)

            return output_dict

    def get_current_sampling_probabilities(self) -> Dict:
        """Get current sampling probabilities for motions or sub-motions.

        Returns:
            dict: Dictionary containing scores and probabilities for logging
        """
        if self.use_sub_motion_indexing:
            scores = self.sub_motion_scores.clone()
            indexing_type = "sub_motion"
            num_items = self.num_sub_motions
        else:
            scores = self.motion_scores.clone()
            indexing_type = "motion"
            num_items = len(self.motion_ids)

        if self.use_linear_weighted_sampling:
            # Calculate sampling probabilities using the same logic as weighted sampling
            base_prob = self.sampling_base_probability / num_items
            score_weights = scores / torch.sum(scores)
            score_prob = (1.0 - self.sampling_base_probability) * score_weights
            final_probs = base_prob + score_prob
            sampling_method = "weighted"
        else:
            # Uniform sampling
            final_probs = torch.ones(num_items) / num_items
            sampling_method = "uniform"

        # Convert tensors to lists for JSON serialization
        result = {
            "indexing_type": indexing_type,
            "sampling_method": sampling_method,
            "num_items": num_items,
            "scores": scores.tolist(),
            "probabilities": final_probs.tolist(),
            "score_statistics": {
                "mean": scores.mean().item(),
                "std": scores.std().item(),
                "min": scores.min().item(),
                "max": scores.max().item(),
            },
            "probability_statistics": {
                "mean": final_probs.mean().item(),
                "std": final_probs.std().item(),
                "min": final_probs.min().item(),
                "max": final_probs.max().item(),
            },
        }

        # Add TD error EMA statistics
        try:
            td_stats = self.get_td_error_statistics()
            result["td_error_statistics"] = td_stats
        except Exception as e:
            logger.warning(
                f"Failed to include TD error statistics in JSON: {e}"
            )

        # Add sub-motion specific information if enabled
        if self.use_sub_motion_indexing:
            result["sub_motion_config"] = {
                "max_frame_length": self.max_frame_length,
                "overlap_frames": self.sub_motion_overlap_frames,
                "num_original_motions": len(self.motion_ids),
            }

            # Add mapping from sub-motion IDs to original motion info
            sub_motion_mapping = []
            for i, sub_motion_info in enumerate(self.sub_motion_infos):
                mapping_info = {
                    "sub_motion_id": i,
                    "original_motion_id": sub_motion_info[
                        "original_motion_id"
                    ],
                    "original_motion_key": sub_motion_info[
                        "original_motion_key"
                    ],
                    "start_frame": sub_motion_info["sub_motion_start_frame"],
                    "end_frame": sub_motion_info["sub_motion_end_frame"],
                    "length": sub_motion_info["sub_motion_length"],
                    "score": scores[i].item(),
                    "probability": final_probs[i].item(),
                }

                # Add TD error EMA for logging
                if hasattr(self, "sub_motion_td_ema") and i < len(
                    self.sub_motion_td_ema
                ):
                    mapping_info["td_error_ema"] = self.sub_motion_td_ema[
                        i
                    ].item()

                sub_motion_mapping.append(mapping_info)
            result["sub_motion_mapping"] = sub_motion_mapping

            # Add top and bottom 100 sub-motion keys by score
            sorted_sub_motion_mapping = sorted(
                sub_motion_mapping, key=lambda x: x["score"], reverse=True
            )
            result["top_100_sub_motion_keys"] = [
                {
                    "sub_motion_id": item["sub_motion_id"],
                    "original_motion_key": item["original_motion_key"],
                    "score": item["score"],
                    "probability": item["probability"],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                }
                for item in sorted_sub_motion_mapping[:100]
            ]
            result["bottom_100_sub_motion_keys"] = [
                {
                    "sub_motion_id": item["sub_motion_id"],
                    "original_motion_key": item["original_motion_key"],
                    "score": item["score"],
                    "probability": item["probability"],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                }
                for item in sorted_sub_motion_mapping[-100:]
            ]
        else:
            # Add original motion mapping for regular motion indexing
            motion_mapping = []
            for i, motion_id in enumerate(self.motion_ids):
                motion_key = self.motion_id2key[motion_id]
                mapping_info = {
                    "motion_id": motion_id,
                    "motion_key": motion_key,
                    "score": scores[i].item(),
                    "probability": final_probs[i].item(),
                }

                # Add TD error EMA for logging
                if hasattr(self, "motion_td_ema") and i < len(
                    self.motion_td_ema
                ):
                    mapping_info["td_error_ema"] = self.motion_td_ema[i].item()

                motion_mapping.append(mapping_info)
            result["motion_mapping"] = motion_mapping

            # Add top and bottom 100 motion keys by score
            sorted_motion_mapping = sorted(
                motion_mapping, key=lambda x: x["score"], reverse=True
            )
            result["top_100_motion_keys"] = [
                {
                    "motion_id": item["motion_id"],
                    "motion_key": item["motion_key"],
                    "score": item["score"],
                    "probability": item["probability"],
                }
                for item in sorted_motion_mapping[:100]
            ]
            result["bottom_100_motion_keys"] = [
                {
                    "motion_id": item["motion_id"],
                    "motion_key": item["motion_key"],
                    "score": item["score"],
                    "probability": item["probability"],
                }
                for item in sorted_motion_mapping[-100:]
            ]

        return result
