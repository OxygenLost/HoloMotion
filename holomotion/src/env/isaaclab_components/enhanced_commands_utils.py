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

"""Utilities for creating enhanced commands configurations."""

from isaaclab.utils import configclass

from .enhanced_motion_tracking_command import EnhancedMotionCommandCfg

try:
    from .isaaclab_motion_tracking_command import MotionCommandCfg
except ImportError:
    # Handle case where original command isn't available
    MotionCommandCfg = None


@configclass
class EnhancedCommandsCfg:
    """Enhanced commands configuration with async prefetching."""
    pass


def build_enhanced_commands_config(command_config_dict: dict) -> EnhancedCommandsCfg:
    """Build isaaclab-compatible EnhancedCommandsCfg from a config dictionary.

    This creates an enhanced commands configuration that uses Ray-based async
    prefetching for efficient motion clip loading during training.

    Args:
        command_config_dict: Dictionary mapping command names to command configurations.
                           Each command config should contain the type and parameters.

    Example:
        command_config_dict = {
            "ref_motion": {
                "type": "EnhancedMotionCommandCfg",
                "params": {
                    "command_obs_name": "bydmmc_ref_motion",
                    "motion_lib_cfg": {...},
                    "process_id": 0,
                    "num_processes": 1,
                    "replacement_threshold": 5,  # Replace after 5 completions
                    "prefetch_queue_size": 1000,  # 1K clips in queue
                    "num_prefetch_loaders": 8,   # 8 Ray workers
                    # ... other parameters
                }
            }
        }

    Returns:
        EnhancedCommandsCfg: Configuration object with async prefetching enabled
    """
    
    commands_cfg = EnhancedCommandsCfg()

    # Add command terms dynamically
    for command_name, command_config in command_config_dict.items():
        # Handle both old format (direct params) and new format (type + params)
        if "type" in command_config:
            command_type = command_config["type"]
            command_params = command_config.get("params", {})
        else:
            # Old format: assume MotionCommandCfg and use entire config as params
            command_type = "MotionCommandCfg"
            command_params = command_config

        # Get the command class type
        if command_type == "EnhancedMotionCommandCfg":
            command_cfg = EnhancedMotionCommandCfg(**command_params)
        elif command_type == "MotionCommandCfg":
            # Convert original MotionCommandCfg to EnhancedMotionCommandCfg
            # Add default enhanced parameters if not present
            enhanced_params = command_params.copy()
            enhanced_params.setdefault("replacement_threshold", 5)
            enhanced_params.setdefault("prefetch_queue_size", 1000)
            enhanced_params.setdefault("num_prefetch_loaders", 4)
            
            # Log the conversion for debugging
            from loguru import logger
            logger.info(f"🔄 Converting {command_type} to EnhancedMotionCommandCfg for command '{command_name}'")
            logger.info(f"✅ Added async prefetching with {enhanced_params['num_prefetch_loaders']} workers and {enhanced_params['prefetch_queue_size']} queue size")
            
            command_cfg = EnhancedMotionCommandCfg(**enhanced_params)
        else:
            raise ValueError(f"Unknown enhanced command type: {command_type}")

        # Add command to config
        setattr(commands_cfg, command_name, command_cfg)

    return commands_cfg


def get_default_enhanced_motion_config() -> dict:
    """Get default configuration for enhanced motion command.
    
    Returns:
        Default configuration dictionary for enhanced motion tracking
    """
    return {
        "ref_motion": {
            "type": "EnhancedMotionCommandCfg",
            "params": {
                # Basic motion command parameters
                "command_obs_name": "holomotion_rel_ref_motion_flat",
                "n_fut_frames": 4,
                "target_fps": 30,
                "anchor_bodylink_name": "pelvis",
                "asset_name": "robot",
                
                # Enhanced prefetching parameters
                "replacement_threshold": 5,      # Replace after 5 completions
                "prefetch_queue_size": 1000,     # 1K clips in prefetch queue
                "num_prefetch_loaders": 8,       # 8 Ray workers
                
                # Perturbation ranges for domain randomization
                "root_pose_perturb_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05), 
                    "z": (-0.02, 0.02),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
                },
                "root_vel_perturb_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
                "dof_pos_perturb_range": (-0.1, 0.1),
                "dof_vel_perturb_range": (-1.0, 1.0),
                
                # Debug visualization
                "debug_vis": False,
                
                # Resampling parameters
                "resampling_time_range": (1.0, 1.0),
            }
        }
    }


def log_enhanced_cache_statistics(env, command_name: str = "ref_motion"):
    """Log statistics about the enhanced cache performance.
    
    Args:
        env: Environment instance with enhanced commands
        command_name: Name of the command to get statistics from
    """
    from loguru import logger
    
    if hasattr(env, '_commands') and hasattr(env._commands, command_name):
        command = getattr(env._commands, command_name)
        if hasattr(command, 'get_cache_statistics'):
            stats = command.get_cache_statistics()
            
            logger.info("Enhanced Motion Cache Statistics:")
            logger.info(f"  Queue Utilization: {stats['queue_utilization']:.2%}")
            logger.info(f"  Total Prefetched: {stats['total_prefetched']}")
            logger.info(f"  Total Consumed: {stats['total_consumed']}")
            logger.info(f"  Pending Futures: {stats['pending_futures']}")
            logger.info(f"  Stale Cache Rows: {stats['stale_rows']}")
            logger.info(f"  Avg Completions: {stats['avg_completions']:.1f}")
            logger.info(f"  Replacement Candidates: {stats['replacement_candidates']}")
        else:
            logger.warning(f"Command {command_name} does not support cache statistics")
    else:
        logger.warning(f"Command {command_name} not found in environment")


class EnhancedMotionTrainingLogger:
    """Logger for enhanced motion training with async prefetching metrics."""
    
    def __init__(self, env, log_interval: int = 100):
        """Initialize the training logger.
        
        Args:
            env: Environment instance
            log_interval: How often to log statistics (in steps)
        """
        self.env = env
        self.log_interval = log_interval
        self.step_count = 0
        
    def log_step(self):
        """Log training step information."""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self._log_cache_performance()
            self._log_training_metrics()
    
    def _log_cache_performance(self):
        """Log cache and prefetching performance."""
        log_enhanced_cache_statistics(self.env)
    
    def _log_training_metrics(self):
        """Log training-specific metrics."""
        from loguru import logger
        
        # Log step count and timing
        logger.info(f"Training Step: {self.step_count}")
        
        # Additional training metrics can be added here
        if hasattr(self.env, 'episode_length_buf'):
            avg_episode_length = self.env.episode_length_buf.float().mean()
            logger.info(f"Average Episode Length: {avg_episode_length:.1f}")
