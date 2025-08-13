#!/usr/bin/env bash

source train.env
export CUDA_VISIBLE_DEVICES="0"
export NCCL_P2P_DISABLE="1" 
export NCCL_IB_DISABLE="1"

set -euo pipefail

# 配置 checkpoint（可在此处修改），也可通过第一个参数覆盖
CHECKPOINT_PATH="logs/HoloMotion/20250812_135959-train_unitree_g1_23dof_student_robodance100_dagger_cs/model_4000.pt"
LMDB_PATH="data/lmdb_datasets/lmdb_robodance100_combined_10"
GYMVD_NUM_WORKERS=1

if [[ ${1-} != "" ]]; then
  CHECKPOINT_PATH="$1"
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[Error] Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

# 解析 ckpt 号与目录
CKPT_FILE_NAME="$(basename "$CHECKPOINT_PATH")"
CKPT_DIR="$(dirname "$CHECKPOINT_PATH")"
CKPT_NUM="$(echo "$CKPT_FILE_NAME" | awk -F'[_\.]' '{print $(NF-1)}')"

# Step 1: 逐动作用例评估，导出 per_motion 结果
echo "==================== STEP 1 START: per-motion evaluation ===================="
# python -m holomotion.src.evaluation.eval_motion_tracking_per_motionv2_memoryfix \
#   robot.motion.max_frame_length=18000 \
#   checkpoint="$CHECKPOINT_PATH" \
#   motion_lmdb_path="$LMDB_PATH"
echo "==================== STEP 1 DONE: per-motion evaluation ====================="

# 推导 per_motion 目录
PER_MOTION_DIR="$CKPT_DIR/eval_logs/ckpt_${CKPT_NUM}/per_motion"
if [[ ! -d "$PER_MOTION_DIR" ]]; then
  echo "[Error] per_motion directory not found after Step 1: $PER_MOTION_DIR" >&2
  exit 1
fi

# Step 2: 按 MPJPE 排序，输出 sorted_name.txt（默认写到 eval_logs/ckpt_xxx/ 下）
echo "==================== STEP 2 START: sort per-motion by MPJPE ==================="
# python -m holomotion.src.evaluation.sort_per_motion_by_mpjpe \
#   --per_motion_dir "$PER_MOTION_DIR"
echo "==================== STEP 2 DONE: sort per-motion by MPJPE ===================="

SORTED_TXT="$(dirname "$PER_MOTION_DIR")/sorted_name.txt"
if [[ ! -f "$SORTED_TXT" ]]; then
  echo "[Error] sorted_name.txt not found after Step 2: $SORTED_TXT" >&2
  exit 1
fi

# Step 3: 基于排序结果做后续评估/可视化（通过环境变量传入路径，避免 Hydra struct 报错）
echo "==================== STEP 3 START: grouped render/eval (gymvd) ================"
export SORTED_TXT_PATH="$SORTED_TXT"
source train.env
${Train_CONDA_PREFIX}/bin/python -m holomotion.src.evaluation.eval_motion_tracking_per_motionv4_gymvd \
  robot.motion.max_frame_length=18000 \
  checkpoint="$CHECKPOINT_PATH" \
  robot.motion.motion_file="$LMDB_PATH" \
  motion_lmdb_path="$LMDB_PATH"
echo "==================== STEP 3 DONE: grouped render/eval (gymvd) ================="

echo "Pipeline finished successfully." 