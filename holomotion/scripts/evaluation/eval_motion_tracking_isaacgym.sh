#!/bin/bash
source train.env
export CUDA_VISIBLE_DEVICES="0"

# Configuration
# checkpoint_path="logs/HoloMotion/20250811_232409-train_unitree_g1_23dof_teacher_stage2_robodance100_ft/model_244000.pt"
checkpoint_path="logs/HoloMotion/20250812_135959-train_unitree_g1_23dof_student_robodance100_dagger_cs/model_5000.pt"
lmdb_path="data/lmdb_datasets/lmdb_robodance100_combined_10"
num_envs=1

${Train_CONDA_PREFIX}/bin/accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    holomotion/src/evaluation/eval_motion_tracking.py \
    use_accelerate=true \
    num_envs=${num_envs} \
    env.config.align_marker_to_root=true \
    headless=false \
    export_policy=true \
    env.config.termination.terminate_when_motion_far=false \
    motion_lmdb_path="${lmdb_path}" \
    checkpoint="${checkpoint_path}"
