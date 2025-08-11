source train.env

robot_config="unitree_G1_23dof_shape_fitting_config"

python holomotion/src/motion_retargeting/robot_smpl_shape_fitting.py \
    --config-name=${robot_config}