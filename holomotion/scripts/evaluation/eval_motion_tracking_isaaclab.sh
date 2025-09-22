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

source train_isaaclab.env
export CUDA_VISIBLE_DEVICES="0"

config_name="train_g1_29dof_isaaclab"
motion_file="data/lmdb_datasets/lmdb_rtg_bydmmc_lafan_29dof"

num_envs=1

ckpt_path="logs/HoloMotionLabDebug/20250922_151615-train_g1_29dof_isaaclab/model_10500.pt"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/evaluation/eval_motion_tracking_isaaclab.py \
    --config-name=training/motion_tracking/${config_name} \
    project_name="HoloMotionLabDebug" \
    use_accelerate=false \
    num_envs=${num_envs} \
    headless=false \
    experiment_name=${config_name} \
    checkpoint=${ckpt_path} \
    motion_lmdb_path=${motion_file}
