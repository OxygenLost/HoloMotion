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
export CUDA_VISIBLE_DEVICES="0,1"

# config_name="train_unitree_g1_21dof_teacher"
# config_name="train_unitree_g1_23dof_teacher_stage2"
# config_name="train_unitree_g1_23dof_teacher_stage2_robodance100_ft_official_urdf"
# config_name="train_g1_23dof_beyondmimic_holostudent"
# config_name="train_g1_23dof_teacher_stage1_neo_v18"
# config_name="train_g1_23dof_teacher_stage2_robodance100_ft_pbhc_pd_rew_v18"
# config_name="train_g1_23dof_teacher_stage2_salsa_shines_ft"
# config_name="train_unitree_g1_23dof_teacher_stage2_lafan1_beyondmimc"
# config_name="train_g1_23dof_teacher_stage1_v2_bydmmc_pd"
# config_name="train_g1_23dof_teacher_stage1_v2_bydmmc_pd_rnd"
# config_name="train_g1_23dof_teacher_stage2_stand_squat_ft"
# config_name="train_g1_23dof_student_robodance100_dagger_mlp_beyondmimic"
# config_name="train_g1_23dof_teacher_stage1_v3"
# config_name="train_g1_23dof_teacher_stage1_v4_vae"
# config_name="train_g1_23dof_teacher_stage1_v2_bydmmc_pd_robodance100"
# config_name="train_g1_23dof_teacher_stage1_v3_smaller_moe_lafan_tdcu"
# config_name="train_g1_23dof_student_robodance100_dagger_mlp_ft_lafan_dance"
# config_name="train_g1_23dof_teacher_stage2_20250825_chengdu_demo_ft_s100"
# config_name="train_g1_23dof_teacher_stage1_v5_vae_full_data"
# config_name="train_g1_23dof_student_rd100_holopd_ft_rel_obs"
# config_name="train_g1_23dof_teacher_stage1_v6_vae_tdcu_fulldata"
# config_name="train_unitree_g1_23dof_teacher_stage2_robodance100_ft_bydmimic_pd"
config_name="train_g1_23dof_teacher_stage1_v6_vae_tdcu_fulldata"

# motion_file="data/lmdb_datasets/lmdb_g1_21dof_test"
# motion_file="data/lmdb_datasets/lmdb_unitree_G1_23dof_robodance100"
# motion_file="data/lmdb_datasets/lmdb_douyinhot10v0814_combined10"
motion_file="data/lmdb_datasets/lmdb_lafan1_23dof"
# motion_file="data/lmdb_datasets/full_amass_23dof_lockwrist_asap"
num_envs=32

# checkpoint="/home/maiyue01.chen/projects/humanoid_locomotion/logs/HoloMotionMoTrack/HoloMotionMoTrack/20250731_000211-exp_holomotion_g1_23dof_v27_phc_dr_ft_rew_v7+project-4090-robot-lab-bcloud-bj+20250730235755+exp_holomotion_g1_23dof_v27_phc_dr_ft_rew_v7+nenv_2048x1x8-motion_tracking-g1_23dof_lockwrist/model_240000.pt"

${Train_CONDA_PREFIX}/bin/accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    holomotion/src/training/train_motion_tracking.py \
    --config-name=training/motion_tracking/${config_name} \
    project_name="HoloMotionDebug" \
    use_accelerate=true \
    num_envs=${num_envs} \
    algo.algo.config.log_interval=10 \
    headless=true \
    experiment_name=${config_name} \
    motion_lmdb_path=${motion_file}
