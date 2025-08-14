# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil

import numpy as np
import SimpleITK as sitk
from monai.transforms import BorderPadd, Compose, LoadImaged, SaveImage, SqueezeDimd
from utility import convert_mesh_to_usd, convert_to_mesh

# Define the labels dictionary for organ segmentation
LABELS = {
    "Liver": 1,
    "Spleen": 3,
    "Pancreas": 4,
    "Heart": 115,
    "Body": 200,
    "Gallbladder": 10,
    "Stomach": 12,
    "Small_bowel": 19,
    "Colon": 62,
    "Kidney": {"right_kidney": 5, "left_kidney": 14},
    "Veins": {
        "aorta": 6,
        "inferior_vena_cava": 7,
        "portal_vein_and_splenic_vein": 17,
        "left_iliac_artery": 58,
        "right_iliac_artery": 59,
        "left_iliac_vena": 60,
        "right_iliac_vena": 61,
        "pulmonary_vein": 119,
        "left_subclavian_artery": 123,
        "right_subclavian_artery": 124,
        "superior_vena_cava": 125,
        "brachiocephalic_trunk": 109,
        "left_brachiocephalic_vein": 110,
        "right_brachiocephalic_vein": 111,
        "left_common_carotid_artery": 112,
        "right_common_carotid_artery": 113,
    },
    "Lungs": {
        "left_lung_upper_lobe": 28,
        "left_lung_lower_lobe": 29,
        "right_lung_upper_lobe": 30,
        "right_lung_middle_lobe": 31,
        "right_lung_lower_lobe": 32,
    },
    "Spine": {
        "vertebrae_L6": 131,
        "vertebrae_L5": 33,
        "vertebrae_L4": 34,
        "vertebrae_L3": 35,
        "vertebrae_L2": 36,
        "vertebrae_L1": 37,
        "vertebrae_T12": 38,
        "vertebrae_T11": 39,
        "vertebrae_T10": 40,
        "vertebrae_T9": 41,
        "vertebrae_T8": 42,
        "vertebrae_T7": 43,
        "vertebrae_T6": 44,
        "vertebrae_T5": 45,
        "vertebrae_T4": 46,
        "vertebrae_T3": 47,
        "vertebrae_T2": 48,
        "vertebrae_T1": 49,
        "vertebrae_C7": 50,
        "vertebrae_C6": 51,
        "vertebrae_C5": 52,
        "vertebrae_C4": 53,
        "vertebrae_C3": 54,
        "vertebrae_C2": 55,
        "vertebrae_C1": 56,
        "sacrum": 97,
        "vertebrae_S1": 127,
    },
    "Ribs": {
        "left_rib_1": 63,
        "left_rib_2": 64,
        "left_rib_3": 65,
        "left_rib_4": 66,
        "left_rib_5": 67,
        "left_rib_6": 68,
        "left_rib_7": 69,
        "left_rib_8": 70,
        "left_rib_9": 71,
        "left_rib_10": 72,
        "left_rib_11": 73,
        "left_rib_12": 74,
        "right_rib_1": 75,
        "right_rib_2": 76,
        "right_rib_3": 77,
        "right_rib_4": 78,
        "right_rib_5": 79,
        "right_rib_6": 80,
        "right_rib_7": 81,
        "right_rib_8": 82,
        "right_rib_9": 83,
        "right_rib_10": 84,
        "right_rib_11": 85,
        "right_rib_12": 86,
        "costal_cartilages": 114,
        "sternum": 122,
    },
    "Shoulders": {
        "left_scapula": 89,
        "right_scapula": 90,
        "left_clavicula": 91,
        "right_clavicula": 92,
    },
    "Hips": {"left_hip": 95, "right_hip": 96},
    "Back_muscles": {
        "left_gluteus_maximus": 98,
        "right_gluteus_maximus": 99,
        "left_gluteus_medius": 100,
        "right_gluteus_medius": 101,
        "left_gluteus_minimus": 102,
        "right_gluteus_minimus": 103,
        "left_autochthon": 104,
        "right_autochthon": 105,
        "left_iliopsoas": 106,
        "right_iliopsoas": 107,
    },
}


def convert_nrrd_to_nifti(input_nrrd_path, output_nifti_path):
    """
    Convert NRRD file to NIfTI format.

    Args:
        input_nrrd_path (str): Path to input NRRD file
        output_nifti_path (str): Path to output NIfTI file
    """
    if not os.path.exists(output_nifti_path):
        nrrd_image = sitk.ReadImage(input_nrrd_path)
        sitk.WriteImage(nrrd_image, output_nifti_path)


def nii_to_mesh(input_nii_path, output_nii_path, output_obj_path):
    """
    Convert NIfTI file to mesh format and generate GLTF file.

    Args:
        input_nii_path (str): Path to input NIfTI file
        output_nii_path (str): Path to save intermediate NIfTI files
        output_obj_path (str): Path to save OBJ and GLTF files
    """
    # Validate input file
    if not os.path.exists(input_nii_path):
        raise FileNotFoundError(f"Input file not found: {input_nii_path}")

    # copy the original file inplace and prepend with _original
    shutil.copy(input_nii_path, input_nii_path.replace(".nii.gz", "_original.nii.gz"))

    # Create output directories if they don't exist
    if not os.path.exists(output_nii_path):
        os.makedirs(output_nii_path)
    if not os.path.exists(output_obj_path):
        os.makedirs(output_obj_path)

    pre_trans = Compose(
        [
            LoadImaged(keys="label", ensure_channel_first=True),
            BorderPadd(keys="label", spatial_border=2),
            SqueezeDimd(keys="label", dim=0),
        ]
    )

    # Load the original segmentation - this creates a copy, doesn't modify the original file
    orig_seg = pre_trans({"label": input_nii_path})["label"]

    # Convert to numpy array and ensure proper data type
    # This creates a copy of the data in memory, preserving the original file
    if hasattr(orig_seg, "numpy"):
        orig_seg_array = (
            orig_seg.numpy().copy()
        )  # Explicit copy to ensure no reference to original
    else:
        orig_seg_array = np.array(
            orig_seg
        ).copy()  # Explicit copy to ensure no reference to original

    # Store original data for verification that it wasn't modified
    original_data_backup = orig_seg_array.copy()

    # Check if the segmentation has any non-zero values
    if np.sum(orig_seg_array > 0) == 0:
        print("WARNING: Input segmentation appears to be empty (no non-zero values)")
        return

    # Create output array with same data type as input
    all_organ = np.zeros_like(orig_seg_array, dtype=orig_seg_array.dtype)
    all_label_values = {}

    save_trans = SaveImage(output_ext="nii.gz", output_dtype=np.uint8)

    for j, (organ_name, label_val) in enumerate(LABELS.items(), start=1):
        single_organ = np.zeros_like(orig_seg_array, dtype=np.uint8)

        if isinstance(label_val, dict):
            for _, i in label_val.items():
                # Ensure data type compatibility for comparison
                mask = orig_seg_array == i
                all_organ[mask] = j
                single_organ[mask] = j
        else:
            # Ensure data type compatibility for comparison
            mask = orig_seg_array == label_val
            all_organ[mask] = j
            single_organ[mask] = j

        organ_filename = os.path.join(output_nii_path, organ_name)
        save_trans(single_organ[None], meta_data=orig_seg.meta, filename=organ_filename)

        convert_to_mesh(
            f"{organ_filename}.nii.gz",
            output_obj_path,
            f"{organ_name}.obj",
            label_value=j,
            smoothing_factor=0.5,
            reduction_ratio=0.0,
        )
        all_label_values[j] = organ_name

    # Verify that original data was not modified
    if not np.array_equal(orig_seg_array, original_data_backup):
        raise RuntimeError(
            "CRITICAL ERROR: Original segmentation data was modified during processing!"
        )

    # Check if any labels were found and processed
    if np.sum(all_organ > 0) == 0:
        print("ERROR: No labels were found in the segmentation!")
        return

    all_organ_filename = os.path.join(output_nii_path, "all_organs")
    save_trans(all_organ[None], meta_data=orig_seg.meta, filename=all_organ_filename)

    convert_to_mesh(
        f"{all_organ_filename}.nii.gz",
        output_obj_path,
        "all_organs.gltf",
        label_value=all_label_values,
        smoothing_factor=0.6,
        reduction_ratio=0.0,
    )
    print(f"Saved whole segmentation {all_organ_filename}")


def convert_to_usd(input_path, output_dir=None, pattern=".*label\\.(nii\\.gz|nrrd)$"):
    """
    Main function to convert medical imaging files to USD format. Can handle both single files and directories.

    Args:
        input_path (str): Path to input file or directory containing medical imaging files
        output_dir (str, optional): Directory to save all output files. If None, uses input file's directory
        pattern (str, optional): Regex pattern to match files when input_path is a directory.
                               Defaults to ".*label\\.(nii\\.gz|nrrd)$" which matches
                               files ending with label.nii.gz or label.nrrd
    """
    if os.path.isdir(input_path):
        # Handle directory of files
        matching_files = []

        # Compile the regex pattern once for better performance
        regex = re.compile(pattern)

        # Only look in the specified directory, not subdirectories
        for file in os.listdir(input_path):
            if regex.match(file):
                full_path = os.path.join(input_path, file)
                if os.path.isfile(full_path):  # Make sure it's a file, not a directory
                    matching_files.append(full_path)

        if not matching_files:
            print(f"No files matching pattern '{pattern}' found in {input_path}")
            return

        for input_file in matching_files:
            _process_single_file(input_file, output_dir)
    else:
        # Handle single file
        _process_single_file(input_path, output_dir)


def _process_single_file(input_filepath, output_dir=None):
    """
    Process a single medical imaging file and convert it to USD format.

    Args:
        input_filepath (str): Path to input file (NRRD or NIfTI format)
        output_dir (str, optional): Directory to save all output files
    """
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]

    # If no output directory specified, use input file's directory
    if output_dir is None:
        output_dir = os.path.dirname(input_filepath)

    sample_output_dir = os.path.join(output_dir, base_filename)

    # Skip if output directory already exists
    if os.path.exists(sample_output_dir):
        return

    # Create output directories
    nii_dir = os.path.join(sample_output_dir, "nii")
    obj_dir = os.path.join(sample_output_dir, "obj")
    os.makedirs(nii_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)

    # Determine file format and handle accordingly
    file_extension = os.path.splitext(input_filepath)[1].lower()
    if file_extension == ".nrrd":
        # Convert NRRD to NIfTI
        nifti_path = input_filepath.replace(".nrrd", ".nii.gz")
        convert_nrrd_to_nifti(input_filepath, nifti_path)
    elif file_extension == ".gz" and input_filepath.endswith(".nii.gz"):
        # Already in NIfTI format, use directly
        nifti_path = input_filepath
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Supported formats: .nrrd, .nii.gz"
        )

    # Convert NIfTI to mesh
    nii_to_mesh(nifti_path, nii_dir, obj_dir)

    # Convert mesh to USD
    obj_filename = os.path.join(obj_dir, "all_organs.gltf")
    usd_filename = os.path.join(sample_output_dir, "all_organs.usd")
    convert_mesh_to_usd(obj_filename, usd_filename)

    print(f"Conversion complete: {usd_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert medical imaging files (NRRD/NIfTI) to USD format"
    )
    parser.add_argument(
        "input_path",
        help="Path to input file or directory containing medical imaging files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Directory to save output files (optional, defaults to input file's directory)",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default=".*label\\.(nii\\.gz|nrrd)$",
        help="Regex pattern to match files when input_path is a directory (default: .*label\\.(nii\\.gz|nrrd)$)",
    )

    args = parser.parse_args()
    convert_to_usd(args.input_path, args.output_dir, args.pattern)
