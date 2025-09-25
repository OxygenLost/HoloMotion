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

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from loguru import logger

from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib


@dataclass
class MotionClipData:
    """Dataclass for motion clip data."""
    motion_id: int
    motion_key: str
    start_frame: int
    end_frame: int
    clip_length: int
    
    # Motion data tensors (CPU)
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    global_body_translation: torch.Tensor
    global_body_rotation: torch.Tensor
    global_body_velocity: torch.Tensor
    global_body_angular_velocity: torch.Tensor
    global_body_translation_extend: Optional[torch.Tensor] = None
    global_body_rotation_extend: Optional[torch.Tensor] = None
    global_body_velocity_extend: Optional[torch.Tensor] = None
    global_body_angular_velocity_extend: Optional[torch.Tensor] = None
    local_body_rotation: Optional[torch.Tensor] = None
    frame_flag: Optional[torch.Tensor] = None


@ray.remote
class RayMotionLoader:
    """Ray actor for loading motion clips asynchronously."""
    
    def __init__(self, motion_lib_cfg: Dict, process_id: int = 0, num_processes: int = 1):
        """Initialize the Ray motion loader.
        
        Args:
            motion_lib_cfg: Configuration for motion library
            process_id: Process ID for this loader
            num_processes: Total number of processes
        """
        self.motion_lib = LmdbMotionLib(
            motion_lib_cfg=motion_lib_cfg,
            cache_device=torch.device("cpu"),  # Keep on CPU for prefetching
            process_id=process_id,
            num_processes=num_processes,
        )
        
    def load_motion_clip(self, motion_id: int, start_frame: int = None) -> MotionClipData:
        """Load a single motion clip.
        
        Args:
            motion_id: ID of the motion to load
            start_frame: Optional start frame. If None, randomly samples one.
            
        Returns:
            MotionClipData containing the loaded clip
        """
        # Validate motion_id exists
        if motion_id not in self.motion_lib.motion_id2key:
            available_ids = list(self.motion_lib.motion_id2key.keys())
            raise KeyError(
                f"Motion ID {motion_id} not found in motion library. "
                f"Available IDs: {available_ids[:10]}... (total: {len(available_ids)})"
            )
            
        motion_key = self.motion_lib.motion_id2key[motion_id]
        
        # Get motion metadata
        motion_lengths = self.motion_lib.get_motion_num_frames([motion_id])
        motion_length = motion_lengths[0].item()
        
        # Determine frame range
        if start_frame is None:
            # Random sampling within motion
            max_start = max(0, motion_length - self.motion_lib.max_frame_length)
            start_frame = np.random.randint(0, max_start + 1)
        
        end_frame = min(start_frame + self.motion_lib.max_frame_length, motion_length)
        clip_length = end_frame - start_frame
        
        # Load motion data from LMDB
        with self.motion_lib.lmdb_handle.begin() as txn:
            frame_slice = slice(start_frame, end_frame)
            
            # Read all motion arrays
            from holomotion.src.training.lmdb_motion_lib import read_motion_array
            
            dof_pos = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "dof_pos", slices=frame_slice
            )
            dof_vel = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "dof_vels", slices=frame_slice
            )
            global_body_translation = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_translation", slices=frame_slice
            )
            global_body_rotation = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_rotation_quat", slices=frame_slice
            )
            global_body_velocity = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_velocity", slices=frame_slice
            )
            global_body_angular_velocity = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_angular_velocity", slices=frame_slice
            )
            
            # Extended body data (optional)
            global_body_translation_extend = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_translation_extend", slices=frame_slice
            )
            global_body_rotation_extend = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_rotation_quat_extend", slices=frame_slice
            )
            global_body_velocity_extend = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_velocity_extend", slices=frame_slice
            )
            global_body_angular_velocity_extend = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "global_angular_velocity_extend", slices=frame_slice
            )
            local_body_rotation = read_motion_array(
                self.motion_lib.lmdb_handle, motion_key, "local_rotation_quat", slices=frame_slice
            )
            
            # Create frame flag
            frame_flag = np.ones(clip_length, dtype=np.int64)
            if clip_length > 0:
                frame_flag[0] = 0
                frame_flag[-1] = 2
        
        # Convert to tensors (CPU)
        clip_data = MotionClipData(
            motion_id=motion_id,
            motion_key=motion_key,
            start_frame=start_frame,
            end_frame=end_frame,
            clip_length=clip_length,
            dof_pos=torch.from_numpy(dof_pos.copy()) if dof_pos is not None else None,
            dof_vel=torch.from_numpy(dof_vel.copy()) if dof_vel is not None else None,
            global_body_translation=torch.from_numpy(global_body_translation.copy()) if global_body_translation is not None else None,
            global_body_rotation=torch.from_numpy(global_body_rotation.copy()) if global_body_rotation is not None else None,
            global_body_velocity=torch.from_numpy(global_body_velocity.copy()) if global_body_velocity is not None else None,
            global_body_angular_velocity=torch.from_numpy(global_body_angular_velocity.copy()) if global_body_angular_velocity is not None else None,
            global_body_translation_extend=torch.from_numpy(global_body_translation_extend.copy()) if global_body_translation_extend is not None else None,
            global_body_rotation_extend=torch.from_numpy(global_body_rotation_extend.copy()) if global_body_rotation_extend is not None else None,
            global_body_velocity_extend=torch.from_numpy(global_body_velocity_extend.copy()) if global_body_velocity_extend is not None else None,
            global_body_angular_velocity_extend=torch.from_numpy(global_body_angular_velocity_extend.copy()) if global_body_angular_velocity_extend is not None else None,
            local_body_rotation=torch.from_numpy(local_body_rotation.copy()) if local_body_rotation is not None else None,
            frame_flag=torch.from_numpy(frame_flag.copy()) if frame_flag is not None else None,
        )
        
        return clip_data
    
    def sample_motion_ids(self, num_samples: int, eval: bool = False) -> List[int]:
        """Sample motion IDs using the motion library's sampling strategy."""
        # Safety check: ensure motion library has valid motion IDs
        if not hasattr(self.motion_lib, 'motion_ids') or len(self.motion_lib.motion_ids) == 0:
            raise RuntimeError("Motion library has no valid motion IDs")
        
        # Debug info about motion library state
        num_motions = len(self.motion_lib.motion_ids)
        valid_motion_ids = self.motion_lib.motion_ids[:10]  # First 10 for debugging
        
        sampled_ids = self.motion_lib.sample_motion_ids_only(num_samples, eval=eval)
        
        # Validate sampled IDs exist in motion_id2key mapping
        for motion_id in sampled_ids:
            if motion_id.item() not in self.motion_lib.motion_id2key:
                raise KeyError(
                    f"Sampled motion_id {motion_id.item()} not in motion_id2key mapping. "
                    f"Total motions: {num_motions}, Valid IDs: {valid_motion_ids}"
                )
        
        return sampled_ids.tolist()




class AsyncMotionPrefetcher:
    """Async motion prefetcher with CPU queue and parallel workers."""
    
    def __init__(
        self, 
        motion_lib_cfg: Dict,
        prefetch_queue_size: int = 1000,
        num_loaders: int = 4,
        process_id: int = 0,
        num_processes: int = 1,
        batch_size: int = 8,  # Batch size for parallel loading
        max_pending_futures: int = 32,  # Max pending Ray futures
    ):
        """Initialize the async motion prefetcher.
        
        Args:
            motion_lib_cfg: Configuration for motion library
            prefetch_queue_size: Size of the prefetch queue
            num_loaders: Number of parallel loaders
            process_id: Process ID for this prefetcher
            num_processes: Total number of processes
            batch_size: Number of clips to load per batch
            max_pending_futures: Maximum number of pending Ray futures
        """
        self.motion_lib_cfg = motion_lib_cfg
        self.prefetch_queue_size = prefetch_queue_size
        self.num_loaders = num_loaders
        self.process_id = process_id
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.max_pending_futures = max_pending_futures
        
        # CPU-based prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=prefetch_queue_size)
        self.is_running = False
        self.prefetch_threads = []
        
        # Ray-specific: futures tracking for parallel execution
        self.pending_futures = []
        self.futures_lock = threading.Lock()
        
        # Initialize Ray loaders
        logger.info(f"Initializing {num_loaders} Ray motion loaders...")
        self.loaders = [
            RayMotionLoader.remote(
                motion_lib_cfg=motion_lib_cfg,
                process_id=process_id,
                num_processes=num_processes,
            )
            for _ in range(num_loaders)
        ]
        
        self.loader_idx = 0
        
        # Statistics
        self.total_prefetched = 0
        self.total_consumed = 0
        self.total_batches_submitted = 0
        self.total_batches_completed = 0
        
    def start_prefetching(self):
        """Start the prefetching process."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info(f"Starting async motion prefetching with {self.num_loaders} Ray loaders...")
        logger.info(f"Using Ray with batch size {self.batch_size} and max {self.max_pending_futures} pending futures")
        
        # Start batch submitter thread
        batch_thread = threading.Thread(
            target=self._ray_batch_submitter,
            daemon=True,
        )
        batch_thread.start()
        self.prefetch_threads.append(batch_thread)
        
        # Start future collector thread
        collector_thread = threading.Thread(
            target=self._ray_future_collector,
            daemon=True,
        )
        collector_thread.start()
        self.prefetch_threads.append(collector_thread)
            
        logger.info(f"Prefetching started with queue size {self.prefetch_queue_size}")
    
    def stop_prefetching(self):
        """Stop the prefetching process."""
        self.is_running = False
        logger.info("Stopping async motion prefetching...")
        
        # Wait for threads to finish
        for thread in self.prefetch_threads:
            thread.join(timeout=1.0)
        
        self.prefetch_threads.clear()
        
        # Clear pending futures
        with self.futures_lock:
            self.pending_futures.clear()
            
        logger.info("Prefetching stopped")
    
    def _ray_batch_submitter(self):
        """Ray batch submitter that dispatches batches of motion loading tasks."""
        while self.is_running:
            # Check if we have room for more futures
            with self.futures_lock:
                if len(self.pending_futures) >= self.max_pending_futures:
                    time.sleep(0.01)
                    continue
            
            # Check if queue has enough space
            remaining_queue_space = self.prefetch_queue_size - self.prefetch_queue.qsize()
            if remaining_queue_space < self.batch_size:
                time.sleep(0.01)
                continue
            
            # Submit batch to workers
            batch_futures = []
            for i in range(self.batch_size):
                # Round-robin loader selection
                loader = self.loaders[self.loader_idx % self.num_loaders]
                self.loader_idx += 1
                
                # Sample motion ID
                motion_ids_future = loader.sample_motion_ids.remote(1, eval=False)
                batch_futures.append(("sample", motion_ids_future, loader))
            
            # Submit the batch for tracking
            with self.futures_lock:
                self.pending_futures.extend(batch_futures)
            
            self.total_batches_submitted += 1
    
    def _ray_future_collector(self):
        """Ray future collector that processes completed futures."""
        while self.is_running:
            with self.futures_lock:
                if not self.pending_futures:
                    time.sleep(0.01)
                    continue
                
                # Check for completed futures (non-blocking)
                completed_indices = []
                for i, (stage, future, loader) in enumerate(self.pending_futures):
                    ready, _ = ray.wait([future], timeout=0)
                    if ready:  # If future is ready
                        completed_indices.append(i)
                
                if not completed_indices:
                    continue
                
                # Process completed futures in reverse order to maintain indices
                for i in reversed(completed_indices):
                    stage, future, loader = self.pending_futures.pop(i)
                    
                    if stage == "sample":
                        # Motion ID sampling completed, now load the clip
                        motion_ids = ray.get(future)
                        motion_id = motion_ids[0]
                        clip_future = loader.load_motion_clip.remote(motion_id)
                        # Re-add as loading stage
                        self.pending_futures.append(("load", clip_future, loader))
                    elif stage == "load":
                        # Clip loading completed
                        clip_data = ray.get(future)
                        # Add to queue if there's space
                        if not self.prefetch_queue.full():
                            self.prefetch_queue.put(clip_data, block=False)
                            self.total_prefetched += 1
            
            # Update batch completion stats
            if len(completed_indices) > 0:
                self.total_batches_completed += 1
    
    def get_prefetched_clip(self, timeout: float = 0.1) -> Optional[MotionClipData]:
        """Get a prefetched motion clip from the queue.
        
        Args:
            timeout: Timeout for getting from queue
            
        Returns:
            MotionClipData if available, None if queue is empty
        """
        if self.prefetch_queue.empty():
            return None
            
        clip_data = self.prefetch_queue.get(block=True, timeout=timeout)
        self.total_consumed += 1
        return clip_data
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.prefetch_queue.qsize()
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get prefetching statistics."""
        return {
            "queue_size": self.get_queue_size(),
            "max_queue_size": self.prefetch_queue_size,
            "total_prefetched": self.total_prefetched,
            "total_consumed": self.total_consumed,
            "total_batches_submitted": self.total_batches_submitted,
            "total_batches_completed": self.total_batches_completed,
            "pending_futures": len(self.pending_futures),
            "queue_utilization": self.get_queue_size() / self.prefetch_queue_size,
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_prefetching()
