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
"""Sort per-motion JSONs by MPJPE (global) and output motion names.

Default per_motion directory:
  /home/bo07.zhang/bo07.zhang/Gitlab2/HoloMotion/logs/ckpt/20250807_111658-train_unitree_g1_21dof_teacher_phc/eval_logs/ckpt_11000/per_motion

Usage:
  python -m holomotion.src.evaluation.sort_per_motion_by_mpjpe \
    --per_motion_dir /path/to/per_motion \
    --output_txt /path/to/sorted_name.txt

If --output_txt is not provided, the file will be created as 'sorted_name.txt'
under the parent directory of per_motion.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Optional

DEFAULT_PER_MOTION_DIR = (
    "/home/bo07.zhang/bo07.zhang/Gitlab2/HoloMotion/logs/ckpt/"
    "20250807_111658-train_unitree_g1_21dof_teacher_phc/eval_logs/ckpt_11000/per_motion"
)


def _safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _extract_mpjpe_g_mean(data: dict) -> Optional[float]:
    # 1) Prefer top-level mpjpe_g_mean_mm
    if isinstance(data, dict) and "mpjpe_g_mean_mm" in data:
        try:
            return float(data["mpjpe_g_mean_mm"])
        except Exception:
            pass

    # 2) If concat_curves exists, compute mean over mpjpe_g_per_frame_mm
    try:
        concat = data.get("concat_curves", {})
        per_frame = concat.get("mpjpe_g_per_frame_mm")
        if isinstance(per_frame, list) and len(per_frame) > 0:
            return _safe_mean([float(x) for x in per_frame])
    except Exception:
        pass

    # 3) Fallback: look into clips list; compute weighted or simple mean
    try:
        clips = data.get("clips", [])
        if isinstance(clips, list) and len(clips) > 0:
            weighted_sum = 0.0
            total_weight = 0
            simple_vals: List[float] = []
            for clip in clips:
                if not isinstance(clip, dict):
                    continue
                mean_val = clip.get("mpjpe_g_mean_mm")
                per_frame = clip.get("mpjpe_g_per_frame_mm")
                try:
                    if per_frame and isinstance(per_frame, list) and len(per_frame) > 0:
                        w = len(per_frame)
                        weighted_sum += float(sum(per_frame) / w) * w
                        total_weight += w
                    elif mean_val is not None:
                        simple_vals.append(float(mean_val))
                except Exception:
                    continue
            if total_weight > 0:
                return float(weighted_sum / total_weight)
            if simple_vals:
                return _safe_mean(simple_vals)
    except Exception:
        pass

    return None


def _gather_scores(per_motion_dir: str) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for name in os.listdir(per_motion_dir):
        if not name.endswith(".json"):
            continue
        json_path = os.path.join(per_motion_dir, name)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        # motion_key fallback: filename without extension
        motion_key = data.get("motion_key") if isinstance(data, dict) else None
        if not motion_key:
            motion_key = os.path.splitext(name)[0]
        score = _extract_mpjpe_g_mean(data)
        if score is None:
            # Push to the end by assigning +inf-like value
            score = float("inf")
        pairs.append((motion_key, float(score)))
    return pairs


def main():
    print("===== STEP 2: START sort_per_motion_by_mpjpe =====")
    parser = argparse.ArgumentParser(
        description="Sort per-motion JSONs by mpjpe_g_mean_mm and output motion names"
    )
    parser.add_argument(
        "--per_motion_dir",
        type=str,
        default=DEFAULT_PER_MOTION_DIR,
        help="Path to per_motion directory containing per-motion JSONs",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Optional output txt path. Default is parent_dir/per_motion/sorted_name.txt",
    )
    parser.add_argument(
        "--emit_top_mid_tail",
        action="store_true",
        help="Also emit first10.txt, mid10.txt, last10.txt alongside the main sorted file",
    )
    args = parser.parse_args()

    per_motion_dir = args.per_motion_dir
    if not os.path.isdir(per_motion_dir):
        raise FileNotFoundError(f"per_motion directory not found: {per_motion_dir}")

    # Compute output path
    parent_dir = os.path.dirname(per_motion_dir.rstrip("/"))
    output_txt = args.output_txt or os.path.join(parent_dir, "sorted_name.txt")

    # Gather and sort
    pairs = _gather_scores(per_motion_dir)
    pairs.sort(key=lambda x: (x[1], x[0]))  # sort by score asc, then name

    # Write motion names
    with open(output_txt, "w", encoding="utf-8") as f:
        for motion_key, _ in pairs:
            f.write(f"{motion_key}\n")

    print(f"Saved sorted motion names to: {output_txt}")

    # Optionally emit first/mid/last 10%
    if args.emit_top_mid_tail and pairs:
        names = [mk for mk, _ in pairs]
        n = len(names)
        k = max(1, int(n * 0.1))
        first10 = names[:k]
        last10 = names[-k:]
        # middle segment centered; if not divisible, take slice from n//2 - k//2
        mid_start = max(0, (n // 2) - (k // 2))
        mid_end = min(n, mid_start + k)
        mid10 = names[mid_start:mid_end]

        first_path = os.path.join(parent_dir, "first10.txt")
        mid_path = os.path.join(parent_dir, "mid10.txt")
        last_path = os.path.join(parent_dir, "last10.txt")
        for pth, lst in [(first_path, first10), (mid_path, mid10), (last_path, last10)]:
            with open(pth, "w", encoding="utf-8") as f:
                for mk in lst:
                    f.write(f"{mk}\n")
            print(f"Saved {pth}")

    print("===== STEP 2: DONE sort_per_motion_by_mpjpe =====")


if __name__ == "__main__":
    main() 