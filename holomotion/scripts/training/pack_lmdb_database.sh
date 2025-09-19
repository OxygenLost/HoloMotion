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
export CUDA_VISIBLE_DEVICES=""

# retargeted_pkl_path="data/retargeted_datasets/waltz"
# dump_dir="data/lmdb_datasets/lmdb_g1_21dof_test"
# robot_config="unitree/G1/21dof/21dof_training"

# retargeted_pkl_path="data/retargeted_datasets/combined_clips_robodance100"
# retargeted_pkl_path="data/retargeted_datasets/robodance100"
# dump_dir="data/lmdb_datasets/lmdb_robodance100_combined_10"

# retargeted_pkl_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/robodance100_no_global_translation_combined10"
# dump_dir="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/lmdb_datasets/lmdb_robodance100_combined_10_no_global_trans"

# retargeted_pkl_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_salsa_shines_phc"
# dump_dir="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/lmdb_datasets/lmdb_23dof_salsa_shines_phc"

# retargeted_pkl_path="data/retargeted_datasets/23dof_recorded_stand_squat"
# dump_dir="data/lmdb_datasets/lmdb_23dof_recorded_stand_squat"

# retargeted_pkl_path="data/retargeted_datasets/douyinhot10v0814_combined10"
# dump_dir="data/lmdb_datasets/lmdb_douyinhot10v0814_combined10"

# retargeted_pkl_path="/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/lafan1_23dof"
# dump_dir="data/lmdb_datasets/lmdb_lafan1_23dof"

# retargeted_pkl_path="data/retargeted_datasets/23dof_0823retargeting_processed_stand_squat"
# dump_dir="data/lmdb_datasets/lmdb_23dof_0823retargeting_processed_stand_squat"

# retargeted_pkl_path="data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance"
# dump_dir="data/lmdb_datasets/lmdb_20250825_chengdu_demo_seg_lanfan_dance"

# retargeted_pkl_path="data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance"
# dump_dir="data/lmdb_datasets/lmdb_20250825_chengdu_demo_seg_lanfan_dance"

# retargeted_pkl_path="data/retargeted_datasets/23dof_0825retargeting_normed"
# dump_dir="data/lmdb_datasets/lmdb_23dof_0825retargeting_normed"

# retargeted_pkl_path="data/retargeted_datasets/23dof_salsa_shines_phc_0825"
# dump_dir="data/lmdb_datasets/lmdb_23dof_salsa_shines_phc_0825"

# retargeted_pkl_path="data/retargeted_datasets/20250825_chengdu_demo_train"
# dump_dir="data/lmdb_datasets/lmdb_20250825_chengdu_demo_train"

# retargeted_pkl_path="data/retargeted_datasets/23dof_stand_still"
# dump_dir="data/lmdb_datasets/lmdb_23dof_stand_still"

# retargeted_pkl_path="data/retargeted_datasets/20250826_chengdu_demo_train_v2"
# dump_dir="data/lmdb_datasets/lmdb_20250826_chengdu_demo_train_v2"

# retargeted_pkl_path="data/retargeted_datasets/20250826_chengdu_demo_train_v3"
# dump_dir="data/lmdb_datasets/lmdb_20250826_chengdu_demo_train_v3"

# retargeted_pkl_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_bydmimic_lafan_dance"
# retargeted_pkl_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_bydmimic_lafan_dance"
# dump_dir="data/lmdb_datasets/lmdb_23dof_bydmimic_lafan_dance"

retargeted_pkl_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/rtg_bydmmc_lafan_29dof"
dump_dir="data/lmdb_datasets/lmdb_rtg_bydmmc_lafan_29dof"


# robot_config="unitree/G1/23dof/23dof_training_v0_official_urdf_beyondmimic_pd"
robot_config="unitree/G1/29dof/29dof_training_v0_official_urdf_beyondmimic_pd"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/training/pack_lmdb.py \
    robot=$robot_config \
    retargeted_pkl_path=$retargeted_pkl_path \
    lmdb_save_dir=$dump_dir \
    debug_local_mode=true \
    debug_no_try_catch=true \
    num_jobs=1
