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
"""Evaluate a checkpointed policy and save per-motion results as separate files.

This script mirrors the setup in `src/evaluation/eval_motion_tracking.py` but
modifies the evaluation loop so that for each motion in the LMDB we save an
individual JSON named after the motion key (one file per motion). If a motion
is split into multiple clips (due to `max_frame_length`), we aggregate clips in
chronological order and export both per-clip stats and concatenated per-frame
curves for the entire motion.

Usage:
  python -m holomotion.src.evaluation.eval_motion_tracking_per_motion \
    checkpoint=/path/to/xxx.pt

Hydra config is reused from `evaluation/eval_isaacgym.yaml`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import hydra
import isaacgym  # noqa: F401
import torch
from accelerate import Accelerator
from hydra.utils import get_class
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path
from omegaconf import open_dict
from tqdm import tqdm

from holomotion.src.evaluation.eval_motion_tracking import (
    load_training_config,
    setup_eval_directories,
    setup_logging,
)
from holomotion.src.env.motion_tracking import (
    compute_metrics_lite,
    MotionTrackingEnvironment,
)
from holomotion.src.utils.config import (
    setup_hydra_resolvers,
    compile_config_hf_accelerate,
    compile_config_directories,
)
# compile_config_obs 需要在存在 obs 字段时再调用
from holomotion.src.utils import config as config_utils
from holomotion.src.algo.ppo import PPO


def _sequence_record_from_metrics(metrics: Dict[str, List], motion_key: str):
    """Flatten compute_metrics_lite (concatenate=False) output to dict."""
    def _first(name: str):
        arr_list = metrics.get(name)
        if arr_list is None or len(arr_list) == 0:
            return None
        return arr_list[0]

    mpjpe_g = _first("mpjpe_g")  # (T, J)
    mpjpe_l = _first("mpjpe_l")  # (T, J)
    mpjpe_pa = _first("mpjpe_pa")  # (T, J)
    vel_dist = _first("vel_dist")  # (T-1,)
    accel_dist = _first("accel_dist")  # (T-2,)

    out = {"motion_key": motion_key}
    if mpjpe_g is not None:
        out["mpjpe_g_mean_mm"] = float(mpjpe_g.mean())
        out["mpjpe_g_per_frame_mm"] = mpjpe_g.mean(axis=1).tolist()
        out["num_frames"] = int(mpjpe_g.shape[0])
        out["num_joints"] = int(mpjpe_g.shape[1])
    if mpjpe_l is not None:
        out["mpjpe_l_mean_mm"] = float(mpjpe_l.mean())
        out["mpjpe_l_per_frame_mm"] = mpjpe_l.mean(axis=1).tolist()
    if mpjpe_pa is not None:
        out["mpjpe_pa_mean_mm"] = float(mpjpe_pa.mean())
        out["mpjpe_pa_per_frame_mm"] = mpjpe_pa.mean(axis=1).tolist()
    if vel_dist is not None:
        out["vel_dist_mean_mm_s"] = float(vel_dist.mean())
        out["vel_dist_per_frame_mm_s"] = vel_dist.tolist()
    if accel_dist is not None:
        out["accel_dist_mean_mm_s2"] = float(accel_dist.mean())
        out["accel_dist_per_frame_mm_s2"] = accel_dist.tolist()
    return out


def _concat_motion_curves(per_clip: List[dict]) -> dict:
    """Concatenate per-frame curves across clips in chronological order."""
    # Sort by clip start_frame if present
    per_clip_sorted = sorted(
        per_clip, key=lambda x: x.get("start_frame", 0)
    )
    concat = {}
    def _append(key: str, values: Optional[List[float]]):
        if values is None:
            return
        concat.setdefault(key, [])
        concat[key].extend(values)

    for rec in per_clip_sorted:
        _append("mpjpe_g_per_frame_mm", rec.get("mpjpe_g_per_frame_mm"))
        _append("mpjpe_l_per_frame_mm", rec.get("mpjpe_l_per_frame_mm"))
        _append("mpjpe_pa_per_frame_mm", rec.get("mpjpe_pa_per_frame_mm"))
        _append("vel_dist_per_frame_mm_s", rec.get("vel_dist_per_frame_mm_s"))
        _append(
            "accel_dist_per_frame_mm_s2",
            rec.get("accel_dist_per_frame_mm_s2"),
        )
    return concat


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaacgym",
    version_base=None,
)
def main(config: OmegaConf):
    if config.checkpoint is None:
        raise ValueError("Checkpoint path must be provided for evaluation")

    # Merge training config to get the exact env/algo settings
    config = load_training_config(config.checkpoint, config)

    accelerator = Accelerator()
    setup_logging(accelerator)

    # Temporarily disable struct to allow setting extra runtime keys
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(config, False)

    # Force num_envs = 16
    try:
        with open_dict(config):
            config.num_envs = 1
        with open_dict(config.env.config):
            config.env.config.num_envs = 1
        logger.info("Override num_envs to 16 for per-motion evaluation")
    except Exception as e:
        logger.warning(f"Failed to set num_envs=16: {e}")

    # Provide a default experiment_dir if missing to satisfy directories setup
    if not hasattr(config, "experiment_dir") or config.get("experiment_dir") is None:
        ckpt_path = Path(config.checkpoint)
        default_exp_dir = ckpt_path.parent / "eval_experiments" / ckpt_path.stem
        config.experiment_dir = str(default_exp_dir)

    # Fine-grained compile: resolvers + accelerate + directories; obs only if present
    setup_hydra_resolvers()
    config = compile_config_hf_accelerate(config, accelerator)
    config = compile_config_directories(config)

    # Conditionally compile obs to avoid Missing key obs error
    try:
        has_obs = (
            hasattr(config, "env")
            and hasattr(config.env, "config")
            and hasattr(config.env.config, "obs")
            and config.env.config.get("obs") is not None
        )
    except Exception:
        has_obs = False

    if has_obs:
        try:
            config = config_utils.compile_config_obs(config)
        except Exception as e:
            logger.warning(f"compile_config_obs failed, continue without it: {e}")
    else:
        logger.warning("config.env.config.obs missing, skip compile_config_obs().")

    # 确保 env.config.robot 存在：若缺失则从顶层 config.robot 映射
    with open_dict(config.env.config):
        if not hasattr(config.env.config, "robot") or config.env.config.get("robot") is None:
            if hasattr(config, "robot") and config.get("robot") is not None:
                config.env.config.robot = config.robot
                logger.info("Injected env.config.robot from top-level config.robot for compatibility")
            else:
                logger.warning("Top-level config.robot missing; env.config.robot remains unset")

        # 强制无渲染运行
        config.env.config.headless = True

    # Re-enable struct after compilation
    _OC.set_struct(config, True)

    # Prepare directories (open env.config for mutation inside)
    with open_dict(config.env.config):
        eval_log_dir = setup_eval_directories(config, config.checkpoint)
    per_motion_dir = os.path.join(eval_log_dir, "per_motion")
    os.makedirs(per_motion_dir, exist_ok=True)

    # Create environment class with fallback
    try:
        env_target = config.env._target_
        env_class = get_class(env_target)
    except Exception:
        logger.warning("config.env._target_ missing or invalid, fallback to MotionTrackingEnvironment")
        env_class = MotionTrackingEnvironment

    env = env_class(
        config=config.env.config,
        device=accelerator.device,
    )

    # Create algorithm class with fallback
    try:
        algo_target = config.algo.algo._target_
        algo_class = get_class(algo_target)
    except Exception:
        logger.warning("config.algo.algo._target_ missing or invalid, fallback to PPO")
        algo_class = PPO

    algo = algo_class(
        env=env,
        config=config.algo.algo.config if hasattr(config.algo.algo, "config") else config.algo.algo,
        log_dir=str(eval_log_dir),
        device=accelerator.device,
    )

    # Load checkpoint
    algo.setup()
    algo.load(config.checkpoint)

    # Prepare inference policy
    eval_policy = algo._get_inference_policy(device=accelerator.device)

    # Put env into eval mode and reset
    env.set_is_evaluating()

    # Storage for per-motion aggregated results
    motion_to_clips: Dict[str, List[dict]] = defaultdict(list)

    # Outer progress bar for total evaluation clips (motions may have multiple clips)
    try:
        total_eval_clips = len(getattr(env._motion_lib, "eval_allocation_schedule", []))
    except Exception:
        total_eval_clips = 0

    # Pre-compute how many clips per motion to allow saving per-motion immediately once done
    motion_total_clips: Optional[Dict[str, int]] = None
    try:
        alloc = getattr(env._motion_lib, "eval_allocation_schedule", [])
        counts: Dict[str, int] = defaultdict(int)
        for clip in alloc:
            key = clip.get("motion_key")
            if key is not None:
                counts[key] += 1
        motion_total_clips = dict(counts)
    except Exception:
        motion_total_clips = None

    outer_pbar = tqdm(
        total=total_eval_clips if total_eval_clips > 0 else None,
        desc="Eval motions",
        disable=not accelerator.is_main_process,
    )

    # Iterate evaluation batches prepared by motion lib
    last_eval_batch = False

    # We will keep stepping per batch for `cached_max_frame_len` frames
    while not last_eval_batch:
        cached_max_frame_len = env._motion_lib.cache.max_frame_length
        last_eval_batch = env.resample_motion_eval()  # also does reset_all()

        # Identify current clips and motion keys for each env index
        clips = env._motion_lib.cache.cached_clip_info or []
        num_envs = len(clips)
        if num_envs == 0:
            continue

        motion_keys = [clip.get("motion_key", f"motion_{i}") for i, clip in enumerate(clips)]
        if accelerator.is_main_process:
            print(f"Evaluating motions: {', '.join(motion_keys)}")

        # Check if all motions in this batch already have JSON outputs; if so, skip simulation
        skip_flags = []
        for mk in motion_keys:
            safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in mk)
            json_path = os.path.join(per_motion_dir, f"{safe_name}.json")
            skip_flags.append(os.path.exists(json_path))
        if all(skip_flags):
            if accelerator.is_main_process:
                print(f"Skip batch, results already exist: {', '.join(motion_keys)}")
            if outer_pbar is not None:
                try:
                    outer_pbar.update(num_envs)
                except Exception:
                    pass
            # 防止显存碎片，轻量清理
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue

        # Prepare per-env accumulation lists (CPU lists of numpy arrays)
        pred_pos_seq = [[] for _ in range(num_envs)]  # each: list of (nb, 3)
        gt_pos_seq = [[] for _ in range(num_envs)]
        pred_rot_seq = [[] for _ in range(num_envs)]  # each: list of (nb, 4)
        gt_rot_seq = [[] for _ in range(num_envs)]

        # First observation is available after reset_all()
        obs_dict = env.obs_buf_dict

        # Inner progress bar for steps in this batch
        inner_desc = f"Steps: {motion_keys[0]}" if len(motion_keys) == 1 else f"Steps: {motion_keys[0]} (+{len(motion_keys)-1})"

        # Run the stepping loop under inference_mode to avoid autograd overhead/memory
        with torch.no_grad():
            step_iter = tqdm(
                range(cached_max_frame_len),
                desc=inner_desc,
                leave=False,
                disable=not accelerator.is_main_process,
            )
            for step in step_iter:
                # Inference action
                actions = eval_policy(obs_dict["actor_obs"])  # should be CUDA tensor
                # Step environment (physics and observation update)
                obs_dict, rewards, dones, extras = env.step({"actions": actions})

                # Collect positions/rotations at this step (moved to CPU numpy)
                cur_pos = env._rigid_body_pos_extend.detach().cpu().numpy()
                ref_pos = env.ref_body_pos_t.detach().cpu().numpy()
                cur_rot = env._rigid_body_rot_extend.detach().cpu().numpy()
                ref_rot = env.ref_body_rot_t.detach().cpu().numpy()

                for i in range(num_envs):
                    pred_pos_seq[i].append(cur_pos[i])
                    gt_pos_seq[i].append(ref_pos[i])
                    pred_rot_seq[i].append(cur_rot[i])
                    gt_rot_seq[i].append(ref_rot[i])

                # Explicitly drop step-local GPU refs (actions kept minimal)
                del rewards, dones, extras

        # End of batch: compute per-env metrics and dump into per-motion buckets
        for i, clip in enumerate(clips):
            motion_key = clip.get("motion_key", f"motion_{i}")
            start_frame = int(clip.get("start_frame", 0))
            end_frame = int(clip.get("end_frame", start_frame))

            # If this motion already has output json, skip collecting/saving
            safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in motion_key)
            out_json_path = os.path.join(per_motion_dir, f"{safe_name}.json")
            if os.path.exists(out_json_path):
                continue

            # Stack sequences to arrays (T, J, C) on CPU
            jpos_pred = torch.tensor(pred_pos_seq[i]).numpy()
            jpos_gt = torch.tensor(gt_pos_seq[i]).numpy()
            jrot_pred = torch.tensor(pred_rot_seq[i]).numpy()
            jrot_gt = torch.tensor(gt_rot_seq[i]).numpy()

            # Shape safety (T, J, 3/4)
            T = min(jpos_pred.shape[0], jpos_gt.shape[0])
            J = min(jpos_pred.shape[1], jpos_gt.shape[1])
            jpos_pred = jpos_pred[:T, :J]
            jpos_gt = jpos_gt[:T, :J]
            jrot_pred = jrot_pred[:T, :J]
            jrot_gt = jrot_gt[:T, :J]

            metrics = compute_metrics_lite(
                pred_pos_all=[jpos_pred],
                gt_pos_all=[jpos_gt],
                pred_rot_all=[jrot_pred],
                gt_rot_all=[jrot_gt],
                root_idx=0,
                use_tqdm=False,
                concatenate=False,
                pred_vel=None,
                gt_vel=None,
                pred_accel=None,
                gt_accel=None,
            )

            rec = _sequence_record_from_metrics(metrics, motion_key)
            rec.update({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "clip_length": end_frame - start_frame,
            })
            motion_to_clips[motion_key].append(rec)

            # If we've collected all clips for this motion, save immediately
            try:
                if motion_total_clips is not None:
                    if len(motion_to_clips[motion_key]) >= motion_total_clips.get(motion_key, 0):
                        concat_curves = _concat_motion_curves(motion_to_clips[motion_key])
                        out = {
                            "motion_key": motion_key,
                            "num_clips": len(motion_to_clips[motion_key]),
                            "clips": motion_to_clips[motion_key],
                            "concat_curves": concat_curves,
                        }
                        out_path = os.path.join(per_motion_dir, f"{safe_name}.json")
                        if not os.path.exists(out_path):
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(out, f, ensure_ascii=False, indent=2)
                            logger.info(f"Saved per-motion result: {out_path}")
                        else:
                            logger.info(f"Per-motion result exists, skip save: {out_path}")
                        # Free memory for this motion
                        del motion_to_clips[motion_key]
            except Exception as e:
                logger.warning(f"Failed to save per-motion json for {motion_key}: {e}")

        # Batch-level cleanup to mitigate CUDA memory growth
        try:
            # Ensure actions tensor is normal (not inference-marked) for future in-place writes
            if hasattr(env, "actions") and env.actions is not None:
                env.actions = env.actions.clone()
            # Drop large CPU lists
            del pred_pos_seq, gt_pos_seq, pred_rot_seq, gt_rot_seq
            # Encourage Python GC and free cached CUDA memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Update outer progress bar by number of clips processed in this batch
        if outer_pbar is not None:
            try:
                outer_pbar.update(num_envs)
            except Exception:
                pass

    # Close outer progress bar
    try:
        if outer_pbar is not None:
            outer_pbar.close()
    except Exception:
        pass

    # After all batches: write one JSON per motion
    for motion_key, clip_recs in motion_to_clips.items():
        # Concatenate curves across clips
        concat_curves = _concat_motion_curves(clip_recs)
        # Compute motion-level simple means from concatenated curves (optional)
        out = {
            "motion_key": motion_key,
            "num_clips": len(clip_recs),
            "clips": clip_recs,
            "concat_curves": concat_curves,
        }
        # Save as motion_key.json (sanitize filename)
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_"
            for c in motion_key
        )
        out_path = os.path.join(per_motion_dir, f"{safe_name}.json")
        if os.path.exists(out_path):
            logger.info(f"Per-motion result exists, skip save: {out_path}")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved per-motion result: {out_path}")


if __name__ == "__main__":
    main() 