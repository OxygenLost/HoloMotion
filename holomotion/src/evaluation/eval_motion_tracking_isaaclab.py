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
import sys

import hydra
import torch
from hydra.utils import get_class
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.utils.config import compile_config
from isaaclab.app import AppLauncher


def setup_logging():
    """Configure logging for the evaluation process."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
        colorize=True,
    )


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_base",
    version_base=None,
)
def main(config: OmegaConf):
    """Evaluate the motion tracking model.

    Args:
        config: OmegaConf object containing the evaluation configuration.

    """
    setup_logging()

    # Check for compilation issues early and provide guidance
    import os

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
