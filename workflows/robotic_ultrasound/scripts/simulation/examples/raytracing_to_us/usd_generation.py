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

def generate_ct(root_dir, num_output_samples):
    bundle_root = os.path.join(root_dir, "maisi_ct_generative")
    override = {
        "output_size_xy": 256,
        "output_size_z": 256,
        "spacing_xy": 1.5,
        "spacing_z": 1.5,
        "num_output_samples": num_output_samples,
    }
    workflow = create_workflow(
        config_file=os.path.join(bundle_root, "configs/inference.json"),
        workflow_type="inference",
        bundle_root=bundle_root,
        **override,
    )

    workflow.run()

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
        "Gallbladder": 10,
        "Stomach": 12,
        "Small_bowel": 19,
        "Colon": 62,
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


def generate_mesh(root_dir, mesh_dir):
    ct_list = os.listdir(os.path.join(root_dir, "maisi_ct_generative", "output"))
    ct_list = [l for l in ct_list if l.endswith("_image.nii.gz")]
    for ct in ct_list:
        print(f"Processing {ct}")
        ct_name = ct.split(".nii.gz")[0]
        input_nii_path = os.path.join(root_dir, "maisi_ct_generative", "output", ct)
        output_nii_path = os.path.join(mesh_dir, ct_name, "nii")
        output_obj_path = os.path.join(mesh_dir, ct_name, "obj")
        out = nii_to_mesh(input_nii_path, output_nii_path, output_obj_path)

        obj_filename = f"{output_obj_path}/all_organs.gltf"
        usd_filename = f"{output_obj_path}/all_organs.usd"

        convert_mesh_to_usd(obj_filename, usd_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mesh from CT data.")
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--mesh_dir', type=str, default="mesh_data")
    parser.add_argument('--ct_samples', type=int, default=10)
    parser.add_argument('--generate_ct', type=bool, default=False)
    parser.add_argument('--generate_mesh', type=bool, default=False)
    args = parser.parse_args()

    if args.generate_ct:
        generate_ct(args.root_dir, args.ct_samples)

    if args.generate_mesh:
        generate_mesh(args.root_dir, args.mesh_dir)
