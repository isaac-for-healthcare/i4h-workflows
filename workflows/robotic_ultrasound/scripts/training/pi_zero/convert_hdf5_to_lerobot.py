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

"""
Minimal example script for converting a dataset to LeRobot format.

Usage:
python convert_hdf5_to_lerobot.py /path/to/your/data \
    [--repo_id REPO_ID] [--task_prompt TASK_PROMPT] [--image_shape IMAGE_SHAPE]

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import argparse
import glob
import os
import re
import shutil
import warnings

import h5py
import tqdm
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from openpi_client import image_tools


def create_lerobot_dataset(
    output_path: str,
    robot_type: str = "panda",
    fps: int = 10,
    image_shape: tuple[int, int, int] = (224, 224, 3),
    state_shape: tuple[int, ...] = (7,),
    actions_shape: tuple[int, ...] = (6,),
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """
    Creates a LeRobot dataset with specified configurations.

    This function initializes a LeRobot dataset with the given parameters,
    defining the structure and features of the dataset.

    Parameters:
    - output_path: The path where the dataset will be saved.
    - robot_type: The type of robot.
    - fps: Frames per second for the dataset.
    - image_shape: Tuple defining the shape of the image data.
    - state_shape: Tuple defining the shape of the state data.
    - actions_shape: Tuple defining the shape of the action data.
    - image_writer_threads: Number of threads for image writing.
    - image_writer_processes: Number of processes for image writing.

    Returns:
    - An instance of LeRobotDataset configured with the specified parameters.
    """

    if os.path.isdir(output_path):
        raise Exception(f"Output path {output_path} already exists.")

    return LeRobotDataset.create(
        repo_id=output_path,
        robot_type=robot_type,
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": actions_shape,
                "names": ["actions"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def main(
    data_dir: str,
    repo_id: str,
    task_prompt: str,
    image_shape=(224, 224, 3),
    **dataset_config_kwargs,
):
    """
    Main function to convert HDF5 files to LeRobot format.

    This function processes HDF5 files in the specified directory, extracts
    relevant data, and saves it in the LeRobot format. It supports customization
    of dataset parameters such as image shape.

    Parameters:
    - data_dir: Directory containing the HDF5 files to convert.
    - repo_id: Identifier for the dataset repository.
    - task_prompt: Description of the task for which the dataset is used.
    - image_shape: Tuple defining the shape of the image data (default is (224, 224, 3)).
    - dataset_config_kwargs: Additional keyword arguments for dataset configuration.
    """
    final_output_path = LEROBOT_HOME / repo_id
    if final_output_path.exists():
        try:
            shutil.rmtree(final_output_path)
        except Exception as e:
            raise Exception(f"Error removing {final_output_path}: {e}. Please ensure that you have write permissions.")

    # Create LeRobot dataset, define features to store
    dataset = create_lerobot_dataset(final_output_path, image_shape=image_shape, **dataset_config_kwargs)

    # Collect all the hdf5 files in the data directory
    if not os.path.isdir(data_dir):
        raise Exception(f"Data directory {data_dir} does not exist.")
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not data_files:
        warnings.warn(f"No HDF5 files found in {data_dir}")
        return

    episode_names = []
    for f in data_files:
        match = re.search(r"data_(\d+)\.hdf5", os.path.basename(f))
        if match:
            episode_names.append(match.group(1))
        else:
            warnings.warn(f"File {f} does not match the expected pattern.")

    if not episode_names:
        warnings.warn(f"No episode names found in {data_dir}")
        return

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for episode_idx in tqdm.tqdm(episode_names):
        hdf5_path = os.path.join(data_dir, f"data_{episode_idx}.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            root_name = "data/demo_0"
            num_steps = len(f[root_name]["action"])
            for step in range(num_steps):
                rgb = f[root_name]["observations/rgb_images"][step]
                dataset.add_frame(
                    {
                        "image": image_tools.resize_with_pad(rgb[0], image_shape[0], image_shape[1]),
                        "wrist_image": image_tools.resize_with_pad(rgb[1], image_shape[0], image_shape[1]),
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
        "--task_prompt",
        type=str,
        default="Perform a liver ultrasound.",
        help="Prompt description of the task",
    )
    parser.add_argument(
        "--image_shape",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(224, 224, 3),
        help="Shape of the image data as a comma-separated string, e.g., '224,224,3'",
    )
    args = parser.parse_args()
    main(args.data_dir, args.repo_id, args.task_prompt, image_shape=args.image_shape)
