source train.env

export MUJOCO_GL="osmesa"
# export MUJOCO_GL="egl"

# retargeted path mosified by user
# motion_pkl_root="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted"
# motion_pkl_root="data/retargeted_datasets/robodance100_no_global_translation"
# motion_pkl_root="data/retargeted_datasets/full_amass_23dof_lockwrist_asap"
# motion_pkl_root="data/retargeted_datasets/robodance100_no_global_translation_combined10"
# motion_pkl_root="data/retargeted_datasets/23dof_salsa_shines_phc"
# motion_pkl_root="data/retargeted_datasets/23dof_recorded_stand_squat"
# motion_pkl_root="data/retargeted_datasets/lafan1_23dof"
# motion_pkl_root="data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance"
# motion_pkl_root="data/retargeted_datasets/23dof_stand_squat_norm_yaw"
# motion_pkl_root="data/retargeted_datasets/23dof_0825retargeting_normed"
# motion_pkl_root="data/retargeted_datasets/23dof_salsa_shines_phc_0825"
# motion_pkl_root="data/retargeted_datasets/20250825_chengdu_demo_pkls"
# motion_pkl_root="data/retargeted_datasets/20250825_chengdu_demo_train"
# motion_pkl_root="data/retargeted_datasets/20250826_chengdu_demo_train_v2"
# motion_pkl_root="data/retargeted_datasets/20250826_chengdu_demo_train_v3"
# motion_pkl_root="data/retargeted_datasets/23dof_bydmimic_lafan_dance"
# motion_pkl_root="data/retargeted_datasets/20250903_recorded_dance"
# motion_pkl_root="data/retargeted_datasets/rtg_bydmmc_lafan_29dof"
motion_pkl_root="data/retargeted_datasets/20250909_chengdu_combined_bydmmc"

# "all" for default 
# motion_name="0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded_normalized_sliced-0-15"
# motion_name="rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_full_normalized_smoothed"
# motion_name="0-20250811_douyin143_hps_douyin012_filter"
# motion_name="0-ACCAD_Male2MartialArtsStances_c3d"
# motion_name="0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded"
# motion_name="dance2_subject3_sliced-0-1980_padded"
# motion_name="rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_1_l1y_normalized"

# motion_name="0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-129-1680_padded_normalized"
# motion_name="all"
motion_name="chengdu_bydmmc_normalized_combined"

robot_config="unitree_G1_23dof_mujoco_viz_config"
# robot_config="unitree_G1_29dof_retargeting"

python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    --config-name=${robot_config} \
    +motion_pkl_root=${motion_pkl_root} \
    +motion_name=${motion_name}