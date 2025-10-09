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

config_name="train_g1_29dof_isaaclab_robodance116"
# motion_file="data/lmdb_datasets/lmdb_rtg_bydmmc_lafan_29dof"
motion_file="data/lmdb_datasets/lmdb_29dof_RobodanceListV5_fps_btws_pad"

num_envs=1

# ckpt_path="logs/HoloMotionLabDebug/20250926_180547-train_g1_29dof_isaaclab/model_3500.pt"
ckpt_path="logs/HoloMotionLabDebug/20250928_175429-train_g1_29dof_isaaclab_robodance116/model_60000.pt"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/evaluation/eval_motion_tracking_isaaclab.py \
    --config-name=training/motion_tracking_isaaclab/${config_name} \
    project_name="HoloMotionLabDebug" \
    use_accelerate=false \
    num_envs=${num_envs} \
    headless=true \
    experiment_name=${config_name} \
    checkpoint=${ckpt_path} \
    +export_policy=true \
    +calculate_metrics=false \
    robot.motion.use_sub_motion_indexing=false \
    robot.motion.max_frame_length=2000 \
    robot.motion.handpicked_motion_names='["0-LAFAN1_肘汁摇_1+p02_btws_pad"]' \
    motion_lmdb_path=${motion_file}
    # robot.motion.handpicked_motion_names='["0-LAFAN1_La Song_2+p01_btws_pad","0-LAFAN1_新宝岛_1+p01_btws_pad","0-LAFAN1_科目三_2+p01_btws_pad","0-LAFAN1_阿萨舞_2+p01_btws_pad","0-LAFAN1_爱你+p01_btws_pad","0-LAFAN1_青海摇_3+p01_btws_pad","0-LAFAN1_胜利之舞_2+p01_btws_pad","0-LAFAN1_肘汁摇_1+p02_btws_pad"]' \
