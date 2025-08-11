source train.env

export MUJOCO_GL="osmesa"
# export MUJOCO_GL="egl"

# retargeted path mosified by user
# motion_pkl_root="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted"
motion_pkl_root="data/retargeted_datasets/robodance100"

# "all" for default 
motion_name="0-20250811_douyin143_hps_douyin012_filter"

robot_config="unitree_G1_23dof_mujoco_viz_config"

python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    --config-name=${robot_config} \
    +motion_pkl_root=${motion_pkl_root} \
    +motion_name=${motion_name}