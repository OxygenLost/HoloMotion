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


source train.env
export CUDA_VISIBLE_DEVICES="0"

# config_name="train_unitree_g1_23dof_student_robodance100_dagger_rl"
config_name="train_unitree_g1_23dof_student_robodance100_dagger_cs_mlp_ft"
teacher_ckpt_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/logs/HoloMotion/20250811_232409-train_unitree_g1_23dof_teacher_stage2_robodance100_ft/model_244000.pt"
# motion_file="data/lmdb_datasets/lmdb_g1_21dof_test"
motion_file="data/lmdb_datasets/lmdb_unitree_G1_23dof_robodance100"
num_envs=1024

${Train_CONDA_PREFIX}/bin/accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    holomotion/src/training/train_motion_tracking.py \
    --config-name=training/motion_tracking/${config_name} \
    use_accelerate=true \
    num_envs=${num_envs} \
    headless=true \
    experiment_name=${config_name} \
    motion_lmdb_path=${motion_file} \
    algo.algo.config.teacher_actor_ckpt_path=${teacher_ckpt_path}
