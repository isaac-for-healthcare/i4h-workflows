import os
import argparse
import tempfile
import numpy as np

import vtk
import vtkmodules

from utility import convert_to_mesh, convert_mesh_to_usd

from monai.config import print_config
from monai.bundle.scripts import create_workflow, download
from monai.transforms import LoadImaged, SaveImage, Compose, BorderPadd, SqueezeDimd


def nii_to_mesh(input_nii_path, output_nii_path, output_obj_path):
    """
    This function converts each organ into a separate OBJ file and generates a GLTF file
    containing all organs with hierarchical structure.
    It processes the input NIfTI file and groups 140 labels into 17 categories.

    Args:
        input_nii_path: path to the nii file
        output_nii_path: path to save the obj files
        output_obj_path: path to save the gltf file
    """
    if not os.path.exists(output_nii_path):
        os.makedirs(output_nii_path)

    labels = {
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
        "Shoulders": {"left_scapula": 89, "right_scapula": 90, "left_clavicula": 91, "right_clavicula": 92},
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

    pre_trans = Compose(
        [
            LoadImaged(keys="label", ensure_channel_first=True),
            BorderPadd(keys="label", spatial_border=2),
            SqueezeDimd(keys="label", dim=0),
        ]
    )
    orig_seg = pre_trans({"label": input_nii_path})["label"]
    all_organ = np.zeros_like(orig_seg, dtype=np.uint8)
    all_label_values = {}

    save_trans = SaveImage(output_ext="nii.gz", output_dtype=np.uint8)
    for j, (organ_name, label_val) in enumerate(labels.items(), start=1):
        single_organ = np.zeros_like(orig_seg, dtype=np.uint8)
        print(f"Assigning index {j} to label {organ_name}")
        if isinstance(label_val, dict):
            for _, i in label_val.items():
                all_organ[orig_seg == i] = j
                single_organ[orig_seg == i] = j
        else:
            all_organ[orig_seg == label_val] = j
            single_organ[orig_seg == label_val] = j
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


def generate_mesh(seg_dir):
    ct_list = os.listdir(os.path.join(seg_dir))
    for ct in ct_list:
        print(f"Processing {ct}")
        ct_name = ct.split(".nii.gz")[0]
        input_nii_path = os.path.join(seg_dir, ct)
        output_nii_path = os.path.join(seg_dir, ct_name, "nii")
        output_obj_path = os.path.join(seg_dir, ct_name, "obj")
        out = nii_to_mesh(input_nii_path, output_nii_path, output_obj_path)

        obj_filename = f"{output_obj_path}/all_organs.gltf"
        usd_filename = f"{output_obj_path}/all_organs.usd"

        convert_mesh_to_usd(obj_filename, usd_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mesh from CT data.")
    parser.add_argument('--seg_dir', type=str, required=True)
    args = parser.parse_args()

    generate_mesh(args.seg_dir)
