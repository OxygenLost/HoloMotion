source train.env

export MUJOCO_GL="osmesa"
# export MUJOCO_GL="egl"

# retargeted path mosified by user
# motion_pkl_root="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted"
# motion_pkl_root="data/retargeted_datasets/robodance100_no_global_translation"
# motion_pkl_root="data/retargeted_datasets/full_amass_23dof_lockwrist_asap"
# motion_pkl_root="data/retargeted_datasets/robodance100_no_global_translation_combined10"
# motion_pkl_root="data/retargeted_datasets/23dof_salsa_shines_phc"
motion_pkl_root="data/retargeted_datasets/23dof_recorded_stand_squat"

# "all" for default 
# motion_name="all"
# motion_name="0-20250811_douyin143_hps_douyin012_filter"
# motion_name="0-ACCAD_Male2MartialArtsStances_c3d"
# motion_name="0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded"
motion_name="rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_1_l1y"

robot_config="unitree_G1_23dof_mujoco_viz_config"

python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    --config-name=${robot_config} \
    +motion_pkl_root=${motion_pkl_root} \
    +motion_name=${motion_name}