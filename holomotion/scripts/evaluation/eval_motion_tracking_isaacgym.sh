#!/bin/bash
source train.env
export CUDA_VISIBLE_DEVICES="0"

eval_config="eval_isaacgym"

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
# checkpoint_path="logs/HoloMotion/20250819_103053-train_g1_23dof_student_robodance100_dagger_mlp_pbhc_pd/model_3000.pt"
# checkpoint_path="logs/HoloMotion/20250819_154244-train_g1_23dof_beyondmimic/model_20000.pt"
checkpoint_path="logs/HoloMotion/20250819_161849-train_g1_23dof_beyondmimic_holostudent/model_18000.pt"

# lmdb_path="data/lmdb_datasets/lmdb_robodance100_combined_10"
# lmdb_path="data/lmdb_datasets/lmdb_unitree_G1_23dof_robodance100"
lmdb_path="data/lmdb_datasets/lmdb_lafan1_23dof"
num_envs=1

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
    env.config.termination.terminate_by_low_height=true \
    env.config.termination.terminate_when_motion_far=false \
    env.config.termination.terminate_when_ee_z_far=false \
    motion_lmdb_path="${lmdb_path}" \
    checkpoint="${checkpoint_path}"
    # +robot.motion.handpicked_motion_names=["dance1_subject2_sliced-90-615_padded"] \
