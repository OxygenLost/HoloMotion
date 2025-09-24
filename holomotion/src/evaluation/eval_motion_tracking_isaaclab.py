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

import copy
import os
import sys
from pathlib import Path

import hydra
import onnx
import torch
import torch.nn as nn
from hydra.utils import get_class
from isaaclab.app import AppLauncher
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.utils.config import compile_config
from holomotion.src.modules.network_modules import RunningMeanStdNormalizer


def setup_logging():
    """Configure logging for the evaluation process."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
        colorize=True,
    )


def export_motion_policy_to_onnx(algo, checkpoint_path: str):
    """Export the motion tracking policy to ONNX format."""
    checkpoint = Path(checkpoint_path)
    export_dir = checkpoint.parent / "exported"
    export_dir.mkdir(exist_ok=True)

    onnx_name = checkpoint.name.replace(".pt", ".onnx")
    onnx_path = export_dir / onnx_name

    logger.info("Starting ONNX motion tracking policy export...")

    # Set models to evaluation mode
    algo._eval_mode()

    # Get motion command and setup motion data for motion tracking export
    motion_cmd = algo.env._env.command_manager.get_term("ref_motion")
    logger.info(
        "RefMotionCommand detected - will export motion tracking policy"
    )

    motion_data = motion_cmd.export_motion_data_for_onnx()

    # Debug: Show motion data info to verify it's the full length and from frame 0
    actual_frames_retrieved = motion_data["joint_pos"].shape[0]
    logger.info(f"ONNX Export - Retrieved {actual_frames_retrieved} frames")
    logger.info(
        f"ONNX Export - First frame joint pos sample: {motion_data['joint_pos'][0, :5].tolist()}"
    )
    logger.info(
        f"ONNX Export - Last frame joint pos sample: {motion_data['joint_pos'][-1, :5].tolist()}"
    )
    logger.info(
        f"ONNX Export - First frame body pos sample: {motion_data['body_pos_w'][0, 0, :].tolist()}"
    )

    # Create motion tracking policy wrapper
    class _OnnxMotionPolicyExporter(nn.Module):
        def __init__(self, ppo_algo, motion_data):
            super().__init__()
            if ppo_algo.use_accelerate and hasattr(ppo_algo.actor, "module"):
                self.actor = copy.deepcopy(ppo_algo.actor.module)
            else:
                self.actor = copy.deepcopy(ppo_algo.actor)

            self.actor.to("cpu")
            self.actor.eval()

            # Copy normalizer state if enabled
            self.obs_norm_enabled = bool(
                getattr(ppo_algo, "obs_norm_enabled", False)
            )
            self.actor_obs_normalizer = None
            if (
                self.obs_norm_enabled
                and getattr(ppo_algo, "actor_obs_normalizer", None) is not None
            ):
                # Create a CPU normalizer and load state
                feature_dim = int(ppo_algo.actor_obs_normalizer.feature_dim)
                self.actor_obs_normalizer = RunningMeanStdNormalizer(
                    feature_dim=feature_dim
                )
                self.actor_obs_normalizer.set_state(
                    ppo_algo.actor_obs_normalizer.get_state(),
                    new_buffer_device="cpu",
                )
                logger.info(
                    "Actor observation normalizer applied to ONNX successfully !"
                )

            for key, value in motion_data.items():
                self.register_buffer(key, value, persistent=False)

            self.time_step_total = self.joint_pos.shape[0]

        def forward(self, obs, time_step):
            # Clamp time step to valid range
            time_step_clamped = torch.clamp(
                time_step.long().squeeze(-1), max=self.time_step_total - 1
            )

            # Get policy action
            if self.obs_norm_enabled and self.actor_obs_normalizer is not None:
                obs = self.actor_obs_normalizer.normalize(obs)
            action = self.actor.act_inference(obs)

            return (
                action,
                self.joint_pos[time_step_clamped],
                self.joint_vel[time_step_clamped],
                self.body_pos_w[time_step_clamped],
                self.body_quat_w[time_step_clamped],
                self.body_lin_vel_w[time_step_clamped],
                self.body_ang_vel_w[time_step_clamped],
            )

    # Move exporter to CPU for ONNX export
    exporter = _OnnxMotionPolicyExporter(algo, motion_data).to("cpu")

    # Get example inputs
    obs_example = torch.zeros(
        1, algo.obs_serializer.obs_flat_dim, device="cpu"
    )
    time_step_example = torch.zeros(1, 1, device="cpu")

    # Export with motion outputs
    torch.onnx.export(
        exporter,
        (obs_example, time_step_example),
        onnx_path,
        export_params=True,
        opset_version=11,
        verbose=False,
        input_names=["obs", "time_step"],
        output_names=[
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ],
        dynamic_axes={},
    )

    # Attach metadata
    _attach_onnx_metadata(algo, onnx_path)

    logger.info(
        f"Successfully exported motion tracking policy to: {onnx_path}"
    )

    # Return to training mode
    algo._train_mode()


def _attach_onnx_metadata(algo, onnx_path: Path):
    """Attach metadata to the ONNX model file."""

    metadata = {}
    robot_data = algo.env._env.scene["robot"].data
    metadata["joint_names"] = ",".join(robot_data.joint_names)
    metadata["joint_stiffness"] = ",".join(
        [f"{x:.3f}" for x in robot_data.joint_stiffness[0].cpu().tolist()]
    )
    metadata["joint_damping"] = ",".join(
        [f"{x:.3f}" for x in robot_data.joint_damping[0].cpu().tolist()]
    )
    default_pos = robot_data.default_joint_pos_nominal.cpu().tolist()
    metadata["default_joint_pos"] = ",".join([f"{x:.3f}" for x in default_pos])
    metadata["command_names"] = "motion"
    metadata["observation_names"] = ",".join(
        [
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    )

    # Add action scale
    action_scale = (
        algo.env._env.action_manager.get_term("dof_pos")
        ._scale[0]
        .cpu()
        .tolist()
    )
    metadata["action_scale"] = ",".join([f"{x:.3f}" for x in action_scale])

    # Add motion command metadata
    motion_cmd = algo.env._env.command_manager.get_term("ref_motion")
    metadata["anchor_body_name"] = motion_cmd.anchor_bodylink_name

    # Add key bodies from robot config
    metadata["body_names"] = ",".join(algo.env.config.robot.key_bodies)

    # Load and modify ONNX model
    model = onnx.load(str(onnx_path))

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = str(v)
        model.metadata_props.append(entry)

    onnx.save(model, str(onnx_path))
    logger.info("Successfully attached metadata to ONNX model")


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaaclab",
    version_base=None,
)
def main(config: OmegaConf):
    """Evaluate the motion tracking model.

    Args:
        config: OmegaConf object containing the evaluation configuration.

    """
    setup_logging()
    if os.environ.get("TORCH_COMPILE_DISABLE", "0") != "1":
        logger.info(
            "Tip: If you encounter Triton/compilation errors during evaluation,"
        )
        logger.info(
            "     set environment variable: export TORCH_COMPILE_DISABLE=1"
        )

    config = compile_config(config, accelerator=None)
    headless = config.headless
    log_dir = config.experiment_save_dir
    device = "cuda"

    app_launcher_flags = {
        "headless": headless,
        "enable_cameras": not headless,
    }
    _sim_app_launcher = AppLauncher(**app_launcher_flags)
    _sim_app = _sim_app_launcher.app

    # Setup environment
    env_class = get_class(config.env._target_)
    env = env_class(
        config=config.env.config,
        device=device,
        headless=headless,
        log_dir=log_dir,
    )

    # Setup algorithm
    algo_class = get_class(config.algo.algo._target_)
    algo = algo_class(
        env=env,
        config=config.algo.algo.config,
        log_dir=log_dir,
        device=device,
    )
    algo.setup()

    # Load checkpoint if provided
    if hasattr(config, "checkpoint") and config.checkpoint is not None:
        logger.info(f"Loading checkpoint for evaluation: {config.checkpoint}")
        algo.load(config.checkpoint)
    else:
        logger.warning("No checkpoint provided for evaluation!")

    motion_cmd = algo.env._env.command_manager.get_term("ref_motion")

    algo.env._env.reset()

    motion_cmd.ref_motion_global_frame_ids[:] = (
        motion_cmd._motion_lib.cache.cached_motion_global_start_frames.clone()
    )
    motion_cmd._update_ref_motion_state()

    if config.get("export_policy", True):
        export_motion_policy_to_onnx(algo, config.checkpoint)

    # Run evaluation
    logger.info("Starting evaluation...")

    # You can customize the number of evaluation steps here
    max_eval_steps = config.get("max_eval_steps", 1000)
    eval_metrics = algo.evaluate_policy(max_eval_steps=max_eval_steps)

    if eval_metrics is not None:
        logger.info("Evaluation completed successfully!")
        logger.info(f"Evaluation metrics: {eval_metrics}")
    else:
        logger.info("Evaluation completed on non-main process")


if __name__ == "__main__":
    main()
