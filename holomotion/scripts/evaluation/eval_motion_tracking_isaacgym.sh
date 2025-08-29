#!/bin/bash
source train.env
export CUDA_VISIBLE_DEVICES="0"

eval_config="eval_isaacgym"
# eval_config="eval_isaacgym_with_dr"

# Configuration
# checkpoint_path="logs/HoloMotion/20250811_232409-train_unitree_g1_23dof_teacher_stage2_robodance100_ft/model_244000.pt"
# checkpoint_path="logs/HoloMotion/20250812_135959-train_unitree_g1_23dof_student_robodance100_dagger_cs/model_5000.pt"
# checkpoint_path="logs/HoloMotion/20250813_164210-train_unitree_g1_23dof_teacher_stage2_robodance100_ft/model_247000.pt"
# checkpoint_path="logs/HoloMotion/20250815_192011-train_unitree_g1_23dof_student_robodance100_dagger_cs_mlp/model_15000.pt"
# checkpoint_path="logs/HoloMotion/20250815_030331-train_unitree_g1_23dof_teacher_stage2_robodance100_ft_pbhc_pd/model_233000.pt"
# checkpoint_path="logs/HoloMotion/20250817_154355-train_unitree_g1_23dof_teacher_stage1_lafan1_beyondmimc/model_2000.pt"
# checkpoint_path="logs/HoloMotion/20250817_223637-train_unitree_g1_23dof_teacher_stage2_lafan1_beyondmimc/model_8000.pt"
# checkpoint_path="logs/HoloMotion/20250817_201946-train_unitree_g1_23dof_teacher_stage2_lafan1_beyondmimc/model_8000.pt"
# checkpoint_path="logs/HoloMotion/20250817_112432-train_g1_23dof_student_robodance100_dagger_mlp_ft/model_35000.pt"
# checkpoint_path="logs/HoloMotion/20250817_120556-train_g1_23dof_student_robodance100_dagger_mlp_ft_urdf/model_35000.pt"
# checkpoint_path="logs/HoloMotion/20250817_120606-train_g1_23dof_student_robodance100_dagger_mlp_ft_lowerbody/model_35000.pt"
# checkpoint_path="logs/HoloMotion/20250817_120613-train_g1_23dof_student_robodance100_dagger_mlp_ft_actsmooth/model_35000.pt"
# checkpoint_path="logs/HoloMotion/20250817_120833-train_g1_23dof_student_robodance100_dagger_mlp_ft_power/model_35000.pt"
# checkpoint_path="logs/HoloMotion/20250819_102335-train_g1_23dof_student_robodance100_dagger_mlp_ft_fix_origin/model_26000.pt"
# checkpoint_path="logs/HoloMotion/20250819_103059-train_g1_23dof_student_robodance100_dagger_mlp_ft_noglobal_rew/model_26000.pt"
# checkpoint_path="logs/HoloMotion/20250819_103053-train_g1_23dof_student_robodance100_dagger_mlp_pbhc_pd/model_12000.pt"
# checkpoint_path="logs/HoloMotion/20250819_154244-train_g1_23dof_beyondmimic/model_50000.pt"
# checkpoint_path="logs/HoloMotion/20250819_161849-train_g1_23dof_beyondmimic_holostudent/model_18000.pt"
# checkpoint_path="logs/HoloMotion/20250820_103038-train_g1_23dof_beyondmimic_holostudent/model_35000.pt"
# checkpoint_path="logs/HoloMotion/20250820_214238-train_g1_23dof_teacher_stage2_salsa_shines_ft_pbhcpd/model_249000.pt"
# checkpoint_path="logs/HoloMotion/20250820_215303-train_g1_23dof_teacher_stage2_salsa_shines_ft/model_249000.pt"
# checkpoint_path="logs/HoloMotion/20250821_101920-train_g1_23dof_student_salsa_dagger_mlp_ft_pbhcpd/model_18500.pt"
# checkpoint_path="logs/HoloMotion/20250820_215303-train_g1_23dof_teacher_stage2_salsa_shines_ft/model_249000.pt"
# checkpoint_path="logs/HoloMotion/20250821_101904-train_g1_23dof_student_salsa_dagger_mlp_ft/model_29000.pt"
# checkpoint_path="logs/HoloMotion/20250821_133800-train_unitree_g1_23dof_teacher_stage2_lafan1_beyondmimc/model_11000.pt"
# checkpoint_path="logs/HoloMotion/20250821_122303-train_g1_23dof_beyondmimic/model_36000.pt"
# checkpoint_path="logs/HoloMotion/20250821_120045-train_g1_23dof_beyondmimic/model_132000.pt"
# checkpoint_path="logs/HoloMotion/20250821_204058-train_g1_23dof_teacher_stage2_lafan1_dance_ft/model_265000.pt"
# checkpoint_path="logs/HoloMotion/20250821_204917-train_unitree_g1_23dof_teacher_stage2_lafan1_beyondmimc/model_22000.pt"
# checkpoint_path="logs/HoloMotion/20250822_145826-train_g1_23dof_teacher_stage1_v2_bydmmc_pd/model_20000.pt"
# checkpoint_path="logs/HoloMotion/20250822_174210-train_g1_23dof_teacher_stage2_stand_squat_ft/model_252000.pt"
# checkpoint_path="logs/HoloMotion/20250822_154752-train_g1_23dof_teacher_stage1_v2_bydmmc_pd_rnd_coef01/model_11000.pt"
# checkpoint_path="logs/HoloMotion/20250822_162342-train_g1_23dof_teacher_stage2_amass_dance_ft/model_255000.pt"
# checkpoint_path="logs/HoloMotion/20250824_100345-train_g1_23dof_teacher_stage2_stand_squat_ft/model_248000.pt"
# checkpoint_path="logs/HoloMotion/20250824_100348-train_g1_23dof_student_robodance100_dagger_mlp_beyondmimic/model_1000.pt"
# checkpoint_path="logs/HoloMotion/20250824_232456-train_g1_23dof_teacher_stage1_v3_smaller_moe/model_4000.pt"
# checkpoint_path="logs/HoloMotion/20250825_100851-train_g1_23dof_student_robodance100_dagger_mlp_ft_stand_squat/model_26500.pt"
# checkpoint_path="logs/HoloMotion/20250825_102153-train_g1_23dof_student_robodance100_dagger_mlp_beyondmimic/model_48500.pt"
# checkpoint_path="logs/HoloMotion/20250824_223853-train_g1_23dof_teacher_stage1_v4_vae_critic_novae_kle-6/model_5000.pt"
# checkpoint_path="logs/HoloMotion/20250825_112248-train_g1_23dof_student_robodance100_dagger_mlp_ft_stand_squat/model_27000.pt"
# checkpoint_path="logs/HoloMotion/20250821_204058-train_g1_23dof_teacher_stage2_lafan1_dance_ft/model_265000.pt"
# checkpoint_path="logs/HoloMotion/20250825_135128-train_g1_23dof_student_robodance100_dagger_mlp_ft_lafan_dance/model_27500.pt"
# checkpoint_path="logs/HoloMotion/20250825_135406-train_g1_23dof_teacher_stage1_v4_vae_critic_novae_kle-8/model_9000.pt"
# checkpoint_path="logs/HoloMotion/20250825_220252-train_g1_23dof_teacher_stage2_20250825_chengdu_demo_ft/model_254500.pt"
# checkpoint_path="logs/HoloMotion/20250825_221208-train_g1_23dof_teacher_stage1_v4_vae_full_data/model_33000.pt"
# checkpoint_path="logs/HoloMotion/20250826_104250-train_g1_23dof_student_chengdu_demo_mlp_ft/model_34500.pt"
# checkpoint_path="logs/HoloMotion/20250826_180916-train_g1_23dof_student_chengdu_demo_mlp_ft_s100/model_40500.pt"
# checkpoint_path="logs/HoloMotion/20250827_081137-train_g1_23dof_student_chengdu_demo_mlp_ft_s100/model_62000.pt"
checkpoint_path="logs/HoloMotion/20250827_144925-train_g1_23dof_teacher_stage1_v4_vae_tdcu_fulldata/model_14000.pt"
# checkpoint_path="logs/HoloMotion/20250825_135406-train_g1_23dof_teacher_stage1_v4_vae_critic_novae_kle-8/model_41000.pt"
# checkpoint_path="logs/HoloMotion/20250827_232925-train_g1_23dof_teacher_stage1_v5_vae_tdcu_fulldata/model_6000.pt"
# checkpoint_path="logs/HoloMotion/20250828_174221-train_g1_23dof_teacher_stage1_v4_vae_tdcu_fulldata_lafandance/model_14500.pt"
# checkpoint_path="logs/HoloMotion/20250828_212135-train_g1_23dof_teacher_stage1_v5_vae_tdcu_rd100/model_11000.pt"

# lmdb_path="data/lmdb_datasets/lmdb_robodance100_combined_10"
# lmdb_path="data/lmdb_datasets/lmdb_unitree_G1_23dof_robodance100"
lmdb_path="data/lmdb_datasets/lmdb_lafan1_23dof"
# lmdb_path="data/lmdb_datasets/lmdb_23dof_salsa_shines_phc"
# lmdb_path="data/lmdb_datasets/lmdb_23dof_0823retargeting_processed_stand_squat"
# lmdb_path="data/lmdb_datasets/lmdb_20250825_chengdu_demo_seg_lanfan_dance"
# lmdb_path="data/lmdb_datasets/lmdb_20250825_chengdu_demo_seg_lanfan_dance"
# lmdb_path="data/lmdb_datasets/lmdb_23dof_salsa_shines_phc_0825"
# lmdb_path="data/lmdb_datasets/lmdb_20250825_chengdu_demo_train"
# lmdb_path="data/lmdb_datasets/lmdb_20250826_chengdu_demo_train_v2"

num_envs=4

${Train_CONDA_PREFIX}/bin/accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    holomotion/src/evaluation/eval_motion_tracking.py \
    --config-name=evaluation/${eval_config} \
    use_accelerate=true \
    num_envs=${num_envs} \
    env.config.align_marker_to_root=false \
    headless=false \
    export_policy=true \
    env.config.termination.terminate_by_gravity=true \
    env.config.termination.terminate_by_low_height=false \
    env.config.termination.terminate_when_motion_far=false \
    env.config.termination.terminate_when_ee_z_far=false \
    motion_lmdb_path="${lmdb_path}" \
    +robot.motion.handpicked_motion_names=["dance1_subject2_sliced-90-615_padded"] \
    checkpoint="${checkpoint_path}"
    # ++robot.motion.handpicked_motion_names=["dance2_subject3_sliced-0-1980_padded"] \
    # ++robot.motion.excluded_motion_names=[] \
    # ++robot.motion.handpicked_motion_names=[] \
    # +robot.motion.handpicked_motion_names=["rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_2_l1y"] \
    # +robot.motion.handpicked_motion_names=["dance1_subject2"] \
