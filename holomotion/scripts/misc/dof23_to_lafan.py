import argparse
import os
import sys
import glob

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def dof23_to_lafan_csv(pkl_dir, output_dir, default_wrist_values=None):
    """
    Convert 23 DOF PKL files back to LAFAN-style CSV format with full 29 DOFs

    Args:
        pkl_dir: Directory containing PKL motion files with 23 DOFs
        output_dir: Directory to save converted CSV files
        default_wrist_values: Default values for wrist joints (6 values). If None, uses zeros.
    """
    # Get all PKL files in the directory
    pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))

    if not pkl_files:
        print(f"No PKL files found in {pkl_dir}")
        return

    print(f"Found {len(pkl_files)} PKL files to convert")

    # Default wrist joint values (6 DOFs: left_wrist_roll, left_wrist_pitch, left_wrist_yaw, right_wrist_roll, right_wrist_pitch, right_wrist_yaw)
    if default_wrist_values is None:
        default_wrist_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each PKL file
    for pkl_path in tqdm(pkl_files, desc="Converting PKL files"):
        # Load the PKL file
        motion_data = joblib.load(pkl_path)

        # Extract the first (and typically only) motion sequence
        motion_key = list(motion_data.keys())[0]
        motion_sequence = motion_data[motion_key]

        # Extract data components
        dof_23 = motion_sequence["dof"]  # [T, 23]
        root_trans_offset = motion_sequence["root_trans_offset"]  # [T, 3]
        root_rot = motion_sequence["root_rot"]  # [T, 4] in XYZW format

        T = dof_23.shape[0]  # Number of frames

        # Root quaternion is already in XYZW format (same as LAFAN format)
        root_rot_lafan = root_rot  # [T, 4] - XYZW quaternion

        # Reconstruct full 29 DOF array
        # Original DOF indices in LAFAN format (from lafan_to_23dof.py):
        dof_indices_23 = [
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
            # Right arm (4 DOFs): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            29,
            30,
            31,
            32,
        ]

        # Create full 29 DOF array (indices 7-35 in LAFAN format)
        dof_29 = np.zeros((T, 29))

        # Fill in the 23 DOFs at their correct positions
        for i, lafan_idx in enumerate(dof_indices_23):
            lafan_array_idx = (
                lafan_idx - 7
            )  # Convert to 0-based index for 29 DOF array
            dof_29[:, lafan_array_idx] = dof_23[:, i]

        # Fill in default values for missing wrist joints
        # Left wrist: indices 26, 27, 28 -> array indices 19, 20, 21 (but those are waist, so it's 26-7=19, 27-7=20, 28-7=21)
        # Wait, let me recalculate this...
        # LAFAN indices: 7-35 (29 DOFs), so array indices are 0-28
        # Missing wrist joints are at LAFAN indices: 26, 27, 28, 33, 34, 35
        # Array indices: 19, 20, 21, 26, 27, 28
        # But wait, looking at the original comment, the left wrist joints are 26-28 and right wrist are 33-35
        # However, the dof_indices_23 shows that indices 26, 27, 28 are missing (left wrist) and 33, 34, 35 are missing (right wrist)

        # From the original script comments:
        # left_wrist_roll_joint (index 26-7=19)
        # left_wrist_pitch_joint (index 27-7=20)
        # left_wrist_yaw_joint (index 28-7=21)
        # right_wrist_roll_joint (index 33-7=26)
        # right_wrist_pitch_joint (index 34-7=27)
        # right_wrist_yaw_joint (index 35-7=28)

        wrist_indices = [
            19,
            20,
            21,
            26,
            27,
            28,
        ]  # Array indices in 29 DOF array
        for i, wrist_idx in enumerate(wrist_indices):
            dof_29[:, wrist_idx] = default_wrist_values[i]

        # Combine all data: root_pos (3) + root_quat (4) + dof_29 (29) = 36 columns total
        lafan_data = np.concatenate(
            [
                root_trans_offset,  # [T, 3] - XYZ position
                root_rot_lafan,  # [T, 4] - XYZW quaternion
                dof_29,  # [T, 29] - all joint DOFs
            ],
            axis=1,
        )

        # Generate output filename
        pkl_filename = os.path.basename(pkl_path)
        csv_filename = os.path.splitext(pkl_filename)[0] + ".csv"

        # Remove "0-LAFAN1_" prefix if present to get original filename
        if csv_filename.startswith("0-LAFAN1_"):
            csv_filename = csv_filename[9:]

        csv_path = os.path.join(output_dir, csv_filename)

        # Save as CSV file
        np.savetxt(csv_path, lafan_data, delimiter=",", fmt="%.6f")

        print(f"Converted {pkl_filename} -> {csv_filename}")

    print(f"Conversion completed! CSV files saved to: {output_dir}")


if __name__ == "__main__":
    # Example default usage (uncomment and modify paths as needed):
    # pkl_dir = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/rtg_holomotion_fulldata_g1_23dof"
    # output_dir = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/lafan1_reconstructed_29dof"

    pkl_dir = "data/retargeted_datasets/RobodanceListV5_fps_btws_pad"
    output_dir = "data/lafan_csv/RobodanceListV5_fps_btws_pad"

    dof23_to_lafan_csv(
        pkl_dir=pkl_dir,
        output_dir=output_dir,
        default_wrist_values=None,
    )
