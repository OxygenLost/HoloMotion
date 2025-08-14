source train.env

# amass_dir="assets/test_data/motion_retargeting/Male2Walking_c3d"
# dump_dir="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted" # path for testing
# dump_dir="data/retargeted_datasets/g1_21dof_test" # path for training

amass_dir="data/amass_robodance100_0814"
dump_dir="data/retargeted_datasets/rtg_douyinhot10v0814"

robot_config="unitree_G1_23dof_retargeting_v0"

python holomotion/src/motion_retargeting/phc_fitting.py \
       --config-name=${robot_config} \
       +dump_dir=${dump_dir} \
       +amass_root=${amass_dir} \
       +num_jobs=12
