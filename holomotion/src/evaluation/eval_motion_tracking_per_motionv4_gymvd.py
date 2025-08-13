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
import math
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

# 性能参数（可通过环境变量覆盖）
import os as _os
FRAME_STRIDE = int(_os.environ.get("GYMVD_FRAME_STRIDE", "1"))         # 跨帧采样间隔，>=1；2表示每2帧采一帧
MAX_STEPS_PER_CLIP = int(_os.environ.get("GYMVD_MAX_STEPS", "0"))      # 每个clip最多仿真步数；0表示不限制
SAVE_JSON = _os.environ.get("GYMVD_SAVE_JSON", "0") == "1"             # 是否保存 per_motion json，默认关闭以提速
DISABLE_VIEWER_SYNC = _os.environ.get("GYMVD_DISABLE_SYNC", "1") == "0" # 是否禁用viewer同步（提速）

# 并行设置（可通过Hydra或环境变量控制）
DEFAULT_NUM_WORKERS = int(_os.environ.get("GYMVD_NUM_WORKERS", "1"))
DEFAULT_WORKER_IDX = int(_os.environ.get("GYMVD_WORKER_IDX", "-1"))

# Sorted names file (full path provided by user)
SORTED_TXT_PATH = \
    "/home/bo07.zhang/bo07.zhang/Gitlab2/HoloMotion/logs/ckpt/" \
    "20250807_111658-train_unitree_g1_21dof_teacher_phc/eval_logs/ckpt_11000/sorted_name.txt"


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
    # 若未显式传入 checkpoint，则使用默认目录
    DEFAULT_CKPT_DIR = (
        "/home/bo07.zhang/bo07.zhang/Gitlab2/HoloMotion/logs/ckpt/"
        "20250807_111658-train_unitree_g1_21dof_teacher_phc"
    )

    # 安全读取 checkpoint，避免 OmegaConf 的 MissingMandatoryValue 异常
    from omegaconf.errors import MissingMandatoryValue
    ck_value = None
    try:
        ck_value = config.checkpoint
    except MissingMandatoryValue:
        ck_value = None
    except Exception:
        ck_value = None

    # 读取 sorted_name_txt：优先环境变量 SORTED_TXT_PATH，其次默认常量
    sorted_txt_path = _os.environ.get("SORTED_TXT_PATH", "") or SORTED_TXT_PATH

    if ck_value is None or (isinstance(ck_value, str) and ck_value.strip() == ""):
        with open_dict(config):
            config.checkpoint = DEFAULT_CKPT_DIR
        ck_value = config.checkpoint

    # 如果传入的是目录，则自动选择目录下合适的 ckpt 文件
    ckpt_path = Path(ck_value)
    if ckpt_path.is_dir():
        # 选择命名中含有 "ckpt_" 的最新 .pt 文件；若无，则回退到任意 .pt
        pt_files = sorted(ckpt_path.glob("**/*.pt"))
        ckpt_like = sorted(
            [p for p in pt_files if "ckpt_" in p.name], key=lambda p: p.stat().st_mtime, reverse=True
        )
        chosen = ckpt_like[0] if ckpt_like else (pt_files[-1] if pt_files else None)
        if chosen is None:
            raise FileNotFoundError(f"No .pt checkpoint found under directory: {ckpt_path}")
        with open_dict(config):
            config.checkpoint = str(chosen)
        logger.info(f"Auto-selected checkpoint: {config.checkpoint}")

    if config.checkpoint is None:
        raise ValueError("Checkpoint path must be provided for evaluation")

    # Merge training config to get the exact env/algo settings
    config = load_training_config(config.checkpoint, config)

    accelerator = Accelerator()
    setup_logging(accelerator)

    # Temporarily disable struct to allow setting extra runtime keys
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(config, False)

    # Force num_envs = 4 (per requirement: 4 个一组)
    try:
        with open_dict(config):
            config.num_envs = 1
        with open_dict(config.env.config):
            config.env.config.num_envs = 1
        logger.info("Override num_envs to 4 for grouped rendering (4 per group)")
    except Exception as e:
        logger.warning(f"Failed to set num_envs=4: {e}")

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

        # 本脚本需要渲染，设置为非 headless
        config.env.config.headless = False

    # Re-enable struct after compilation
    _OC.set_struct(config, True)

    # Prepare directories (open env.config for mutation inside)
    with open_dict(config.env.config):
        eval_log_dir = setup_eval_directories(config, config.checkpoint)
    per_motion_dir = os.path.join(eval_log_dir, "per_motion")
    os.makedirs(per_motion_dir, exist_ok=True)

    # 加载 sorted_name.txt 并按顺序分组（每组4个）
    ordered_names: List[str] = []
    try:
        with open(sorted_txt_path, "r", encoding="utf-8") as f:
            ordered_names = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(ordered_names)} motion names from sorted list: {sorted_txt_path}")
    except Exception as e:
        logger.error(f"Failed to load sorted names from {sorted_txt_path}: {e}")
        return

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

    # Viewer性能优化：禁用同步以加速（不限帧率）
    try:
        if DISABLE_VIEWER_SYNC and hasattr(env, "simulator"):
            env.simulator.enable_viewer_sync = False
    except Exception:
        pass

    # 将所有名称按4个一组分组
    groups: List[List[str]] = [ordered_names[i:i+4] for i in range(0, len(ordered_names), 4)]

    # 并行控制：支持多个进程分别处理不同的组份额
    # 读取/设置 parallel.num_workers 与 parallel.worker_idx
    try:
        num_workers = int(getattr(config, "parallel", {}).get("num_workers", DEFAULT_NUM_WORKERS))
    except Exception:
        num_workers = DEFAULT_NUM_WORKERS
    try:
        worker_idx = int(getattr(config, "parallel", {}).get("worker_idx", DEFAULT_WORKER_IDX))
    except Exception:
        worker_idx = DEFAULT_WORKER_IDX

    # 如果需要并行且当前未指定worker_idx，则作为主控进程spawn子进程
    if num_workers > 1 and worker_idx < 0:
        import subprocess, sys
        children = []
        for wi in range(num_workers):
            cmd = [
                sys.executable,
                "-m",
                "holomotion.src.evaluation.eval_motion_tracking_per_motionv4_gymvd",
                f"checkpoint={config.checkpoint}",
            ]
            # 继承当前环境，便于沿用GYMVD_*参数；通过环境变量传递worker信息
            env = dict(_os.environ)
            env["GYMVD_NUM_WORKERS"] = str(num_workers)
            env["GYMVD_WORKER_IDX"] = str(wi)
            p = subprocess.Popen(cmd, env=env)
            children.append(p)
        # 等待所有子进程结束
        exit_code = 0
        for p in children:
            rc = p.wait()
            if rc != 0:
                exit_code = rc
        # 主进程完成
        _os._exit(exit_code)

    # 若指定了worker_idx，则过滤组列表（按顺序连续分片）
    if num_workers > 1 and worker_idx >= 0:
        total_groups = len(groups)
        if total_groups == 0:
            logger.info(f"Worker {worker_idx}/{num_workers} no groups to process")
            return
        chunk_size = (total_groups + num_workers - 1) // num_workers
        start_idx = worker_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_groups)
        groups = groups[start_idx:end_idx]
        logger.info(
            f"Worker {worker_idx}/{num_workers} processing groups [{start_idx}:{end_idx}) "
            f"(count={len(groups)})"
        )

    # 外层总进度（与 v3_render 一致样式）
    try:
        total_eval_clips_all = 0
        # 粗略统计：各组 allocation_schedule 的和
        # 仅用于显示，获取失败则显示不定
        for g in groups:
            present_tmp = [n for n in g if n in getattr(env._motion_lib, "motion_key2id", {})]
            if not present_tmp:
                continue
            prev_keys = env._motion_lib.eval_motion_keys
            env._motion_lib.eval_motion_keys = present_tmp
            sch = env._motion_lib._eval_preallocation()
            total_eval_clips_all += len(sch)
            env._motion_lib.eval_motion_keys = prev_keys
    except Exception:
        total_eval_clips_all = 0

    outer_pbar = tqdm(
        total=total_eval_clips_all if total_eval_clips_all > 0 else None,
        desc="Eval motions",
        disable=not accelerator.is_main_process,
    )

    # 小工具：生成安全的组名
    import re
    def _safe(s: str) -> str:
        return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s)
    def _short_label(names: List[str]) -> str:
        # 尝试提取 douyinxxx
        parts = []
        for n in names:
            m = re.search(r"douyin\d+", n)
            parts.append(m.group(0) if m else n[:12])
        return "-".join(parts)

    # 遍历每一组，设置调度，开启录制，评测，到组结束后停止录制并生成 mp4
    for gi, group_names in enumerate(groups, start=1):
        # 过滤出在 lmdb 中存在的 key
        mlib = env._motion_lib
        present = [n for n in group_names if n in getattr(mlib, "motion_key2id", {})]
        if len(present) == 0:
            logger.warning(f"Group {gi} has no valid motions in LMDB, skip: {group_names}")
            continue

        # 按组设置评测队列顺序
        mlib.eval_motion_keys = present
        mlib.eval_allocation_schedule = mlib._eval_preallocation()
        mlib.eval_schedule_idx = 0

        # 配置录制输出路径
        render_root = Path(config.env.config.save_rendering_dir)
        group_label = _safe(_short_label(present))
        video_stem = render_root / f"group_{gi:03d}_{group_label}"

        # 通过 isaacgym 的 API 自动开启录制（在 render() 中生效）
        try:
            sim = env.simulator
            sim.user_recording_video_path = str(video_stem) + "-%s"
            sim.user_is_recording = True
            sim.user_recording_state_change = True
        except Exception as e:
            logger.warning(f"Failed to arm video recorder: {e}")

        # 外层循环：直到当前组的 allocation schedule 完成
        last_eval_batch = False
        while not last_eval_batch:
            cached_max_frame_len = env._motion_lib.cache.max_frame_length
            if MAX_STEPS_PER_CLIP > 0:
                cached_max_frame_len = min(cached_max_frame_len, MAX_STEPS_PER_CLIP)
            last_eval_batch = env.resample_motion_eval()  # also does reset_all()

            # Identify current clips and motion keys for each env index
            clips = env._motion_lib.cache.cached_clip_info or []
            num_envs = len(clips)
            if num_envs == 0:
                continue

            motion_keys = [clip.get("motion_key", f"motion_{i}") for i, clip in enumerate(clips)]
            if accelerator.is_main_process:
                print(f"Evaluating motions: {', '.join(motion_keys)}")

            selected_indices = [i for i, mk in enumerate(motion_keys) if mk in present]
            if len(selected_indices) == 0:
                # 更新外层进度（与 v3_render 一致：按本批clips数）
                try:
                    outer_pbar.update(num_envs)
                except Exception:
                    pass
                continue

            # First observation is available after reset_all()
            obs_dict = env.obs_buf_dict

            # 内层进度条样式统一为 v3_render：Steps: 第一个motion名(+N)
            inner_desc = (
                f"Steps: {motion_keys[0]}"
                if len(motion_keys) == 1
                else f"Steps: {motion_keys[0]} (+{len(motion_keys)-1})"
            )

            if SAVE_JSON:
                pred_pos_seq = [[] for _ in range(len(selected_indices))]
                gt_pos_seq = [[] for _ in range(len(selected_indices))]
                pred_rot_seq = [[] for _ in range(len(selected_indices))]
                gt_rot_seq = [[] for _ in range(len(selected_indices))]

            with torch.no_grad():
                step_iter = tqdm(
                    range(cached_max_frame_len),
                    desc=inner_desc,
                    leave=False,
                    disable=not accelerator.is_main_process,
                )
                for step in step_iter:
                    actions = eval_policy(obs_dict["actor_obs"])  # CUDA tensor
                    obs_dict, rewards, dones, extras = env.step({"actions": actions})

                    if FRAME_STRIDE > 1 and (step % FRAME_STRIDE) != 0:
                        del rewards, dones, extras
                        continue

                    if SAVE_JSON:
                        cur_pos = env._rigid_body_pos_extend.detach().cpu().numpy()
                        ref_pos = env.ref_body_pos_t.detach().cpu().numpy()
                        cur_rot = env._rigid_body_rot_extend.detach().cpu().numpy()
                        ref_rot = env.ref_body_rot_t.detach().cpu().numpy()
                        for local_idx, global_i in enumerate(selected_indices):
                            pred_pos_seq[local_idx].append(cur_pos[global_i])
                            gt_pos_seq[local_idx].append(ref_pos[global_i])
                            pred_rot_seq[local_idx].append(cur_rot[global_i])
                            gt_rot_seq[local_idx].append(ref_rot[global_i])

                    del rewards, dones, extras

            # 结束batch：保存或跳过
            if SAVE_JSON:
                for local_idx, global_i in enumerate(selected_indices):
                    clip = clips[global_i]
                    motion_key = clip.get("motion_key", f"motion_{global_i}")
                    start_frame = int(clip.get("start_frame", 0))
                    end_frame = int(clip.get("end_frame", start_frame))

                    safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in motion_key)
                    out_json_path = os.path.join(per_motion_dir, f"{safe_name}.json")
                    if os.path.exists(out_json_path):
                        continue

                    jpos_pred = torch.tensor(pred_pos_seq[local_idx]).numpy()
                    jpos_gt = torch.tensor(gt_pos_seq[local_idx]).numpy()
                    jrot_pred = torch.tensor(pred_rot_seq[local_idx]).numpy()
                    jrot_gt = torch.tensor(gt_rot_seq[local_idx]).numpy()

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

                    try:
                        concat_curves = _concat_motion_curves([rec])
                        out = {
                            "motion_key": motion_key,
                            "num_clips": 1,
                            "clips": [rec],
                            "concat_curves": concat_curves,
                        }
                        with open(out_json_path, "w", encoding="utf-8") as f:
                            json.dump(out, f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved per-motion result: {out_json_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save per-motion json for {motion_key}: {e}")

            # Batch-level cleanup与外层进度（与 v3_render 一致）
            try:
                if hasattr(env, "actions") and env.actions is not None:
                    env.actions = env.actions.clone()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                outer_pbar.update(num_envs)
            except Exception:
                pass

        # 结束当前组的录制并触发写入 mp4
        try:
            sim.user_is_recording = False
            sim.user_recording_state_change = True
            env.render()
            logger.info(f"Finished recording group {gi}: {video_stem}*.mp4")
        except Exception as e:
            logger.warning(f"Failed to finalize recording for group {gi}: {e}")

    # 关闭外层进度
    try:
        outer_pbar.close()
    except Exception:
        pass


if __name__ == "__main__":
    main() 