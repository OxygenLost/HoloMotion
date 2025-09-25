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

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from loguru import logger

from holomotion.src.training.lmdb_motion_lib import OnlineMotionCache
from holomotion.src.training.async_motion_prefetcher import AsyncMotionPrefetcher, MotionClipData


@dataclass
class CacheRowMetrics:
    """Metrics for tracking cache row usage."""
    row_id: int
    motion_id: int
    motion_key: str
    times_completed: int = 0
    times_accessed: int = 0
    last_access_step: int = 0
    is_stale: bool = False


class EnhancedMotionCache(OnlineMotionCache):
    """Enhanced motion cache with async prefetching and intelligent replacement."""
    
    def __init__(
        self,
        device: torch.device,
        num_envs: int,
        max_frame_length: int,
        min_frame_length: int,
        num_bodies: int,
        num_dofs: int,
        num_extended_bodies: int,
        key_body_indices: List[int],
        n_fut_frames: int,
        fps: float,
        
        # Enhanced cache parameters
        motion_lib_cfg: Dict,
        replacement_threshold: int = 5,  # Replace after N completions
        prefetch_queue_size: int = 1000,
        num_prefetch_loaders: int = 4,
        process_id: int = 0,
        num_processes: int = 1,
    ):
        """Initialize enhanced motion cache.
        
        Args:
            device: Device for cache tensors
            num_envs: Number of environments
            max_frame_length: Maximum frame length
            min_frame_length: Minimum frame length
            num_bodies: Number of bodies
            num_dofs: Number of DOFs
            num_extended_bodies: Number of extended bodies
            key_body_indices: Key body indices
            n_fut_frames: Number of future frames
            fps: Frames per second
            motion_lib_cfg: Motion library configuration
            replacement_threshold: Number of completions before replacing cache row
            prefetch_queue_size: Size of prefetch queue
            num_prefetch_loaders: Number of parallel prefetch loaders
            process_id: Process ID
            num_processes: Total number of processes
        """
        super().__init__(
            device=device,
            num_envs=num_envs,
            max_frame_length=max_frame_length,
            min_frame_length=min_frame_length,
            num_bodies=num_bodies,
            num_dofs=num_dofs,
            num_extended_bodies=num_extended_bodies,
            key_body_indices=key_body_indices,
            n_fut_frames=n_fut_frames,
            fps=fps,
        )
        
        self.replacement_threshold = replacement_threshold
        self.current_step = 0
        
        # Row metrics tracking
        self.row_metrics: List[CacheRowMetrics] = []
        
        # Initialize async prefetcher
        self.prefetcher = AsyncMotionPrefetcher(
            motion_lib_cfg=motion_lib_cfg,
            prefetch_queue_size=prefetch_queue_size,
            num_loaders=num_prefetch_loaders,
            process_id=process_id,
            num_processes=num_processes,
        )
        
        logger.info(f"Enhanced motion cache initialized with {num_envs} environments")
        logger.info(f"Replacement threshold: {replacement_threshold} completions")
        logger.info(f"Prefetch queue size: {prefetch_queue_size}")
        
    def start_prefetching(self):
        """Start the async prefetching process."""
        self.prefetcher.start_prefetching()
        
    def stop_prefetching(self):
        """Stop the async prefetching process."""
        self.prefetcher.stop_prefetching()
        
    def register_motion_ids(self, motion_ids: torch.Tensor):
        """Register motion IDs and initialize row metrics."""
        super().register_motion_ids(motion_ids)
        
        # Initialize row metrics
        self.row_metrics = []
        for i, motion_id in enumerate(motion_ids):
            metrics = CacheRowMetrics(
                row_id=i,
                motion_id=motion_id.item(),
                motion_key=f"motion_{motion_id.item()}",  # Placeholder
            )
            self.row_metrics.append(metrics)
            
        logger.info(f"Registered {len(motion_ids)} motion IDs with row metrics")
        
    def record_motion_completion(self, env_ids: torch.Tensor):
        """Record motion completion for environments.
        
        Args:
            env_ids: Environment IDs that completed their motions
        """
        self.current_step += 1
        
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        # Update completion counters for affected rows
        for env_id in env_ids:
            env_id_cpu = env_id.item()
            if env_id_cpu < len(self.row_metrics):
                self.row_metrics[env_id_cpu].times_completed += 1
                self.row_metrics[env_id_cpu].last_access_step = self.current_step
                
        # Check for rows that need replacement
        self._check_for_replacements()
        
    def record_motion_access(self, env_ids: torch.Tensor):
        """Record motion access for environments.
        
        Args:
            env_ids: Environment IDs that accessed their motions
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
            
        for env_id in env_ids:
            env_id_cpu = env_id.item()
            if env_id_cpu < len(self.row_metrics):
                self.row_metrics[env_id_cpu].times_accessed += 1
                self.row_metrics[env_id_cpu].last_access_step = self.current_step
                
    def _check_for_replacements(self):
        """Check which cache rows need replacement and replace them."""
        rows_to_replace = []
        
        for metrics in self.row_metrics:
            if (metrics.times_completed >= self.replacement_threshold and 
                not metrics.is_stale):
                rows_to_replace.append(metrics.row_id)
                metrics.is_stale = True
                
        if rows_to_replace:
            logger.info(f"Replacing {len(rows_to_replace)} cache rows due to completion threshold")
            self._replace_cache_rows(rows_to_replace)
            
    def _replace_cache_rows(self, row_ids: List[int]):
        """Replace specified cache rows with prefetched clips.
        
        Args:
            row_ids: List of cache row IDs to replace
        """
        for row_id in row_ids:
            # Get prefetched clip
            clip_data = self.prefetcher.get_prefetched_clip(timeout=0.1)
            if clip_data is None:
                logger.warning(f"No prefetched clip available for row {row_id}, skipping replacement")
                continue
                
            # Replace cache row
            self._load_clip_to_cache_row(row_id, clip_data)
            
            # Update row metrics
            if row_id < len(self.row_metrics):
                self.row_metrics[row_id] = CacheRowMetrics(
                    row_id=row_id,
                    motion_id=clip_data.motion_id,
                    motion_key=clip_data.motion_key,
                    times_completed=0,
                    times_accessed=0,
                    last_access_step=self.current_step,
                    is_stale=False,
                )
                
            logger.debug(f"Replaced cache row {row_id} with motion {clip_data.motion_key}")
            
    def _load_clip_to_cache_row(self, row_id: int, clip_data: MotionClipData):
        """Load a motion clip to a specific cache row.
        
        Args:
            row_id: Cache row ID to load to
            clip_data: Motion clip data to load
        """
        # Move tensors to cache device and load into cache
        if clip_data.dof_pos is not None:
            self.dof_pos[row_id, :clip_data.clip_length, :] = clip_data.dof_pos.to(self.device)
            
        if clip_data.dof_vel is not None:
            self.dof_vels[row_id, :clip_data.clip_length, :] = clip_data.dof_vel.to(self.device)
            
        if clip_data.global_body_translation is not None:
            self.global_body_translation[row_id, :clip_data.clip_length, :] = clip_data.global_body_translation.to(self.device)
            
        if clip_data.global_body_rotation is not None:
            self.global_body_rotation[row_id, :clip_data.clip_length, :] = clip_data.global_body_rotation.to(self.device)
            
        if clip_data.global_body_velocity is not None:
            self.global_body_velocity[row_id, :clip_data.clip_length, :] = clip_data.global_body_velocity.to(self.device)
            
        if clip_data.global_body_angular_velocity is not None:
            self.global_body_angular_velocity[row_id, :clip_data.clip_length, :] = clip_data.global_body_angular_velocity.to(self.device)
            
        # Extended body data
        if clip_data.global_body_translation_extend is not None:
            self.global_body_translation_extend[row_id, :clip_data.clip_length, :] = clip_data.global_body_translation_extend.to(self.device)
            
        if clip_data.global_body_rotation_extend is not None:
            self.global_body_rotation_extend[row_id, :clip_data.clip_length, :] = clip_data.global_body_rotation_extend.to(self.device)
            
        if clip_data.global_body_velocity_extend is not None:
            self.global_body_velocity_extend[row_id, :clip_data.clip_length, :] = clip_data.global_body_velocity_extend.to(self.device)
            
        if clip_data.global_body_angular_velocity_extend is not None:
            self.global_body_angular_velocity_extend[row_id, :clip_data.clip_length, :] = clip_data.global_body_angular_velocity_extend.to(self.device)
            
        if clip_data.local_body_rotation is not None:
            self.local_body_rotation[row_id, :clip_data.clip_length, :] = clip_data.local_body_rotation.to(self.device)
            
        if clip_data.frame_flag is not None:
            self.frame_flag[row_id, :clip_data.clip_length] = clip_data.frame_flag.to(self.device)
            
        # Update cache metadata
        if self.cached_motion_ids is not None:
            self.cached_motion_ids[row_id] = clip_data.motion_id
            
        if self.cached_motion_global_start_frames is not None:
            self.cached_motion_global_start_frames[row_id] = clip_data.start_frame
            
        if self.cached_motion_global_end_frames is not None:
            self.cached_motion_global_end_frames[row_id] = clip_data.end_frame
            
        if self.cached_motion_raw_num_frames is not None:
            # This should be the actual motion length, not clip length
            # For now, use clip length as approximation
            self.cached_motion_raw_num_frames[row_id] = clip_data.clip_length
            
    def get_replacement_candidates(self) -> List[int]:
        """Get list of cache rows that are candidates for replacement.
        
        Returns:
            List of row IDs that can be replaced
        """
        candidates = []
        for metrics in self.row_metrics:
            if metrics.times_completed >= self.replacement_threshold:
                candidates.append(metrics.row_id)
        return candidates
        
    def get_cache_statistics(self) -> Dict[str, Union[int, float, List]]:
        """Get cache statistics including replacement info.
        
        Returns:
            Dictionary with cache statistics
        """
        prefetch_stats = self.prefetcher.get_statistics()
        
        completion_counts = [m.times_completed for m in self.row_metrics]
        access_counts = [m.times_accessed for m in self.row_metrics]
        stale_count = sum(1 for m in self.row_metrics if m.is_stale)
        
        stats = {
            "num_cache_rows": len(self.row_metrics),
            "replacement_threshold": self.replacement_threshold,
            "stale_rows": stale_count,
            "avg_completions": sum(completion_counts) / len(completion_counts) if completion_counts else 0,
            "max_completions": max(completion_counts) if completion_counts else 0,
            "min_completions": min(completion_counts) if completion_counts else 0,
            "avg_accesses": sum(access_counts) / len(access_counts) if access_counts else 0,
            "replacement_candidates": len(self.get_replacement_candidates()),
            "current_step": self.current_step,
        }
        
        # Add prefetch statistics
        stats.update(prefetch_stats)
        
        return stats
        
    def force_replacement(self, row_ids: List[int]):
        """Force replacement of specific cache rows.
        
        Args:
            row_ids: List of row IDs to force replace
        """
        logger.info(f"Force replacing {len(row_ids)} cache rows")
        self._replace_cache_rows(row_ids)
        
    def __del__(self):
        """Cleanup when cache is destroyed."""
        self.stop_prefetching()
