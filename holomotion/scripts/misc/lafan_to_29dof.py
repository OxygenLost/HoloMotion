import argparse
import os
import sys

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


from holomotion.src.motion_retargeting.utils.torch_humanoid_batch import (
    HumanoidBatch,
)


def quaternion_to_axis_angle(quaternions):
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    # Extract the components
    x, y, z, w = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )

    # Compute the rotation angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Compute the axis
    # Handle the case where angle is close to zero
    sin_half_angle = torch.sqrt(1.0 - w * w)
    mask = sin_half_angle < 1e-6

    # Safe division
    sin_half_angle = torch.clamp(sin_half_angle, min=1e-6)
    axis_x = torch.where(mask, x, x / sin_half_angle)
    axis_y = torch.where(mask, y, y / sin_half_angle)
    axis_z = torch.where(mask, z, z / sin_half_angle)

    # Combine angle and axis
    axis_angles = torch.stack(
        [axis_x * angle, axis_y * angle, axis_z * angle], dim=-1
    )

    return axis_angles


def dof_to_pose_aa(
    dof_pos,
    humanoid_fk,
    num_augment_joint,
    root_rot=None,
):
    # Convert input to torch tensor if needed
    if not isinstance(dof_pos, torch.Tensor):
        dof_pos = torch.tensor(dof_pos, dtype=torch.float32)

    # Ensure correct shape
    if dof_pos.dim() == 3 and dof_pos.shape[2] == 1:
        dof_pos = dof_pos.squeeze(-1)  # Remove last dimension if it's 1

    N = dof_pos.shape[0]

    # Handle root rotation
    if root_rot is None:
        # If no root rotation provided, use identity quaternion
        root_rot_aa = torch.zeros((N, 3))
    else:
        # Convert input to torch tensor if needed
        if not isinstance(root_rot, torch.Tensor):
            root_rot = torch.tensor(root_rot, dtype=torch.float32)

        # Check if root_rot is quaternion (shape [N, 4])
        if root_rot.shape[-1] == 4:
            # Convert quaternion to axis-angle
            root_rot_aa = quaternion_to_axis_angle(root_rot)
        elif root_rot.shape[-1] == 3:
            # Already in axis-angle format
            root_rot_aa = root_rot
        else:
            raise ValueError(
                f"Root rotation should have shape [N, 4] for quaternion or [N, 3] for axis-angle, got {root_rot.shape}"
            )

    # Compute joint axis-angle representations
    joint_aa = humanoid_fk.dof_axis * dof_pos.unsqueeze(-1)  # [N, num_dof, 3]

    # Combine root rotation and joint rotations
    pose_aa = torch.cat(
        [
            root_rot_aa.unsqueeze(1),  # Root rotation [N, 1, 3]
            joint_aa,  # Joint rotations [N, num_dof, 3]
            torch.zeros(
                (N, num_augment_joint, 3)
            ),  # Augmented joints [N, num_augment_joint, 3]
        ],
        dim=1,
    )

    # Convert back to numpy for output
    return pose_aa.numpy()


def unitree_lafan_to_29dof(lafan_csv_dir, output_dir, robot_config_path):
    """
    Convert all CSV files in lafan_csv_dir to PKL format with 29 DOFs (including wrist joints)

    Args:
        lafan_csv_dir: Directory containing CSV motion files
        output_dir: Directory to save converted PKL files
        robot_config_path: Path to robot configuration YAML file
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(lafan_csv_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {lafan_csv_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to convert")

    # read the robot config
    robot_config = OmegaConf.load(robot_config_path)
    humanoid_fk = HumanoidBatch(robot_config.robot)
    num_augment_joint = len(robot_config.robot.extend_config)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for csv_filename in tqdm(csv_files, desc="Converting CSV files"):
        csv_path = os.path.join(lafan_csv_dir, csv_filename)

        try:
            # read the csv as numpy array
            lafan_data = np.loadtxt(csv_path, delimiter=",", skiprows=0)

            """
            each line of the csv file is a frame of the LAFAN motion, and the columns are:
            G1: (30 FPS)
                root_joint(XYZQXQYQZQW)
                left_hip_pitch_joint
                left_hip_roll_joint
                left_hip_yaw_joint
                left_knee_joint
                left_ankle_pitch_joint
                left_ankle_roll_joint
                right_hip_pitch_joint
                right_hip_roll_joint
                right_hip_yaw_joint
                right_knee_joint
                right_ankle_pitch_joint
                right_ankle_roll_joint
                waist_yaw_joint
                waist_roll_joint
                waist_pitch_joint
                left_shoulder_pitch_joint
                left_shoulder_roll_joint
                left_shoulder_yaw_joint
                left_elbow_joint
                left_wrist_roll_joint
                left_wrist_pitch_joint
                left_wrist_yaw_joint
                right_shoulder_pitch_joint
                right_shoulder_roll_joint
                right_shoulder_yaw_joint
                right_elbow_joint
                right_wrist_roll_joint
                right_wrist_pitch_joint
                right_wrist_yaw_joint
            """

            # Extract data components
            T = lafan_data.shape[0]  # Number of frames

            # Extract root position (XYZ) and rotation (quaternion WXYZ -> XYZW format)
            root_trans_offset = lafan_data[:, 0:3]  # [T, 3]
            root_rot_quat = lafan_data[:, 3:7]  # [T, 4] in QXQYQZQW format

            # Convert quaternion from QXQYQZQW to XYZW format for consistency
            root_rot = np.zeros_like(root_rot_quat)
            root_rot[:, :3] = root_rot_quat[:, :3]  # xyz components
            root_rot[:, 3] = root_rot_quat[:, 3]  # w component

            # Extract joint DOFs including wrist joints (29 DOFs total)
            # Indices for the 29 DOFs (including wrist joints)
            dof_indices = [
                # Left leg (6 DOFs): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                7,
                8,
                9,
                10,
                11,
                12,
                # Right leg (6 DOFs): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                13,
                14,
                15,
                16,
                17,
                18,
                # Waist (3 DOFs): yaw, roll, pitch
                19,
                20,
                21,
                # Left arm (4 DOFs): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
                22,
                23,
                24,
                25,
                # Left wrist (3 DOFs): wrist_roll, wrist_pitch, wrist_yaw
                26,
                27,
                28,
                # Right arm (4 DOFs): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
                29,
                30,
                31,
                32,
                # Right wrist (3 DOFs): wrist_roll, wrist_pitch, wrist_yaw
                33,
                34,
                35,
            ]

            dof = lafan_data[:, dof_indices]  # [T, 29]

            # Convert DOFs to pose axis-angle representation
            pose_aa = dof_to_pose_aa(
                dof_pos=dof,
                humanoid_fk=humanoid_fk,
                num_augment_joint=num_augment_joint,
            )  # [T, 30, 3] - 29 joints + 1 root

            # Get filename without extension for the key
            filename = os.path.splitext(csv_filename)[0]

            filename = f"0-LAFAN1_{filename}"

            # Create the output data structure
            motion_data = {
                filename: {
                    "dof": dof,  # [T, 29]
                    "pose_aa": pose_aa,  # [T, 30, 3]
                    "root_trans_offset": root_trans_offset,  # [T, 3]
                    "root_rot": root_rot,  # [T, 4]
                    "fps": 30,
                }
            }

            # Save as pickle file
            output_pkl_path = os.path.join(output_dir, f"{filename}.pkl")
            joblib.dump(motion_data, output_pkl_path)

        except Exception as e:
            print(f"Error processing {csv_filename}: {e}")
            continue

    print(f"Conversion completed! PKL files saved to: {output_dir}")


if __name__ == "__main__":
    lafan_csv_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/lafan_csv"
    # output_dir = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/lafan1_29dof"
    output_dir = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/rtg_bydmmc_lafan_29dof"
    robot_config_path = "holomotion/config/motion_retargeting/unitree_G1_29dof_retargeting.yaml"

    unitree_lafan_to_29dof(
        lafan_csv_dir=lafan_csv_dir,
        output_dir=output_dir,
        robot_config_path=robot_config_path,
    )
