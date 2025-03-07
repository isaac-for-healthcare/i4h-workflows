"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
python convert_hdf5_to_lerobot.py /path/to/your/data [--repo_id REPO_ID]

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import argparse
import glob
import os
import re
import shutil

import h5py
import tqdm
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


def main(data_dir: str, repo_id: str, task_prompt: str):
    final_output_path = LEROBOT_HOME / repo_id
    if final_output_path.exists():
        shutil.rmtree(final_output_path)
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=final_output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Collect all the hdf5 files in the data directory
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    # Get the episode names from the file names
    episode_names = [re.search(r"data_(\d+)\.hdf5", os.path.basename(f)).group(1) for f in data_files]

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for episode_idx in tqdm.tqdm(episode_names):
        hdf5_path = os.path.join(data_dir, f"data_{episode_idx}.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            root_name = "data/demo_0"
            num_steps = len(f[root_name]["action"])
            for step in range(num_steps):
                rgb = f[root_name]["observations/rgb"][step]
                dataset.add_frame(
                    {
                        "image": rgb[0],
                        "wrist_image": rgb[1],
                        "state": f[root_name]["abs_joint_pos"][step],
                        "actions": f[root_name]["action"][step],
                    }
                )
        dataset.save_episode(task=task_prompt)

    print(f"Saving dataset to {final_output_path}")
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 files to LeRobot format")
    parser.add_argument("data_dir", type=str, help="Root directory of the HDF5 files to convert")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="i4h/robotic_ultrasound",
        help="Directory to save the dataset under (relative to LEROBOT_HOME)",
    )
    parser.add_argument(
        "--task_prompt", type=str, default="Perform a liver ultrasound.", help="Prompt description of the task"
    )
    args = parser.parse_args()
    main(args.data_dir, args.repo_id, args.task_prompt, args.task_prompt)
