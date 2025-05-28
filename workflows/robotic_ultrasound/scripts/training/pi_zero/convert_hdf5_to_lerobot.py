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
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from openpi_client import image_tools
from PIL import Image


class BaseFeatureDict:
    action_key: str
    room_image_key: str
    wrist_image_key: str
    state_key: str
    seg_room_key: str
    seg_wrist_key: str
    depth_room_key: str
    depth_wrist_key: str

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3),
        state_shape: tuple[int, ...] = (7,),
        actions_shape: tuple[int, ...] = (6,),
        include_depth: bool = False,
        include_seg: bool = False,
        include_video: bool = False,
    ):
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.actions_shape = actions_shape
        self.include_depth = include_depth
        self.include_seg = include_seg
        self.include_video = include_video

    @property
    def features(self):
        features_dict = {
            self.room_image_key: {
                "dtype": "image",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.wrist_image_key: {
                "dtype": "image",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.state_key: {
                "dtype": "float32",
                "shape": self.state_shape,
                "names": ["state"],
            },
            self.action_key: {
                "dtype": "float32",
                "shape": self.actions_shape,
                "names": ["action"],
            },
        }

        if self.include_depth:
            depth_data_img = {
                self.depth_room_key: {
                    "dtype": "image",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channels"],
                },
                self.depth_wrist_key: {
                    "dtype": "image",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channels"],
                },
            }
            features_dict.update(depth_data_img)
            if self.include_video:
                depth_data_vid = {
                    self.depth_room_key: {
                        "dtype": "video",
                        "shape": self.image_shape,
                        "names": ["height", "width", "channels"],
                    },
                    self.depth_wrist_key: {
                        "dtype": "video",
                        "shape": self.image_shape,
                        "names": ["height", "width", "channels"],
                    },
                }
                features_dict.update(depth_data_vid)  # Overwrite with video version

        if self.include_seg:
            seg_data_img = {
                self.seg_room_key: {
                    "dtype": "image",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channels"],
                },
                self.seg_wrist_key: {
                    "dtype": "image",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channels"],
                },
            }
            features_dict.update(seg_data_img)
            if self.include_video:
                seg_data_vid = {
                    self.seg_room_key: {
                        "dtype": "video",
                        "shape": self.image_shape,
                        "names": ["height", "width", "channels"],
                    },
                    self.seg_wrist_key: {
                        "dtype": "video",
                        "shape": self.image_shape,
                        "names": ["height", "width", "channels"],
                    },
                }
                features_dict.update(seg_data_vid)  # Overwrite with video version

        if self.include_video:  # For main images
            main_img_vid = {
                self.room_image_key: {
                    "dtype": "video",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channels"],
                },
                self.wrist_image_key: {
                    "dtype": "video",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channels"],
                },
            }
            features_dict.update(main_img_vid)  # Overwrite with video version

        return features_dict

    def __call__(self, rgb, state, action, seg=None, depth_room=None, depth_wrist=None) -> dict:
        frame_data = {}
        img_h, img_w, _ = self.image_shape
        current_features = self.features  # Access property to ensure it's evaluated

        # Assign mandatory fields (assuming they are always in features_dict from the property)
        frame_data[self.room_image_key] = image_tools.resize_with_pad(rgb[0], img_h, img_w)
        frame_data[self.wrist_image_key] = image_tools.resize_with_pad(rgb[1], img_h, img_w)
        frame_data[self.state_key] = state
        frame_data[self.action_key] = action  # Use subclass-defined action_key

        if seg is not None and self.seg_room_key in current_features:
            frame_data[self.seg_room_key] = image_tools.resize_with_pad(seg[0], img_h, img_w, method=Image.NEAREST)
        if seg is not None and self.seg_wrist_key in current_features:
            frame_data[self.seg_wrist_key] = image_tools.resize_with_pad(seg[1], img_h, img_w, method=Image.NEAREST)

        if depth_room is not None and self.depth_room_key in current_features:
            frame_data[self.depth_room_key] = image_tools.resize_with_pad(depth_room, img_h, img_w).squeeze(2)
        if depth_wrist is not None and self.depth_wrist_key in current_features:
            frame_data[self.depth_wrist_key] = image_tools.resize_with_pad(depth_wrist, img_h, img_w).squeeze(2)

        return frame_data


class Pi0FeatureDict(BaseFeatureDict):
    action_key = "actions"
    room_image_key = "image"
    wrist_image_key = "wrist_image"
    state_key = "state"
    seg_room_key = "observation.seg.room"
    seg_wrist_key = "observation.seg.wrist"
    depth_room_key = "observation.depth.room"
    depth_wrist_key = "observation.depth.wrist"

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3),
        state_shape: tuple[int, ...] = (7,),
        actions_shape: tuple[int, ...] = (6,),
        include_depth: bool = False,
        include_seg: bool = False,
        include_video: bool = False,
    ):
        super().__init__(image_shape, state_shape, actions_shape, include_depth, include_seg, include_video)


class GR00TN1FeatureDict(BaseFeatureDict):
    action_key = "action"  # GR00T uses "action"
    room_image_key = "observation.images.room"
    wrist_image_key = "observation.images.wrist"
    state_key = "observation.state"
    seg_room_key = "observation.seg.room"
    seg_wrist_key = "observation.seg.wrist"
    depth_room_key = "observation.depth.room"
    depth_wrist_key = "observation.depth.wrist"

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3),
        state_shape: tuple[int, ...] = (7,),
        actions_shape: tuple[int, ...] = (6,),
        include_depth: bool = False,
        include_seg: bool = False,
        include_video: bool = True,
    ):
        super().__init__(image_shape, state_shape, actions_shape, include_depth, include_seg, include_video)


def normalize_depth_image(depth_image):
    """
    Normalizes a depth image to the range [0, 255] for visualization.

    Parameters:
    - depth_image: Input depth image

    Returns:
    - Normalized depth image as uint8
    """
    # Convert to inverse depth
    output = 1.0 / (depth_image + 1e-6)

    # Find min and max values
    depth_min = output.min()
    depth_max = output.max()
    max_val = (2**8) - 1  # Maximum value for uint8

    # Normalize to [0, 255]
    if depth_max - depth_min > np.finfo("float").eps:
        out_array = max_val * (output - depth_min) / (depth_max - depth_min)
    else:
        out_array = np.zeros_like(output)

    # Convert to uint8
    return out_array.astype("uint8")


def create_lerobot_dataset(
    output_path: str,
    features: dict,
    robot_type: str = "panda",
    fps: int = 30,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """
    Creates a LeRobot dataset with specified configurations.

    This function initializes a LeRobot dataset with the given parameters,
    defining the structure and features of the dataset.

    Parameters:
    - output_path: The path where the dataset will be saved.
    - features: A dictionary defining the features of the dataset.
    - robot_type: The type of robot.
    - fps: Frames per second for the dataset.
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
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def main(
    data_dir: str,
    repo_id: str,
    task_prompt: str,
    feature_builder,
    include_depth: bool = False,
    include_seg: bool = False,
    run_compute_stats: bool = False,
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
    - include_depth: Whether to include depth images in the dataset.
    - include_seg: Whether to include segmentation images in the dataset.
    - run_compute_stats: Whether to run compute stats.
    - dataset_config_kwargs: Additional keyword arguments for dataset configuration.
    - feature_builder: An instance of a feature dictionary builder class (e.g., Pi0FeatureDict).
    """
    final_output_path = LEROBOT_HOME / repo_id
    if final_output_path.exists():
        try:
            shutil.rmtree(final_output_path)
        except Exception as e:
            raise Exception(f"Error removing {final_output_path}: {e}. Please ensure that you have write permissions.")

    robot_type = dataset_config_kwargs.pop("robot_type", "panda")
    fps = dataset_config_kwargs.pop("fps", 30)
    image_writer_threads = dataset_config_kwargs.pop("image_writer_threads", 10)
    image_writer_processes = dataset_config_kwargs.pop("image_writer_processes", 5)

    dataset = create_lerobot_dataset(
        output_path=final_output_path,
        features=feature_builder.features,
        robot_type=robot_type,
        fps=fps,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

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
    # sort episode_names
    episode_names = sorted(episode_names, key=lambda x: int(x))
    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for episode_idx in tqdm.tqdm(episode_names):
        hdf5_path = os.path.join(data_dir, f"data_{episode_idx}.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            root_name = "data/demo_0"
            num_steps = len(f[root_name]["action"])

            for step in range(num_steps):
                rgb = f[root_name]["observations/rgb_images"][step]
                state = f[root_name]["abs_joint_pos"][step]
                action = f[root_name]["action"][step]

                seg = None
                if include_seg:
                    if "observations/seg_images" in f[root_name]:
                        seg = f[root_name]["observations/seg_images"][step]
                    else:
                        warnings.warn(f"'observations/seg_images' not found in {hdf5_path} but include_seg is True.")

                depth_room_processed, depth_wrist_processed = None, None
                if include_depth:
                    if "observations/depth_images" in f[root_name]:
                        depth_images_raw = f[root_name]["observations/depth_images"][step]
                        depth_room_processed = normalize_depth_image(depth_images_raw[0])
                        depth_wrist_processed = normalize_depth_image(depth_images_raw[1])
                    else:
                        warnings.warn(
                            f"'observations/depth_images' not found in {hdf5_path} but include_depth is True."
                        )

                frame_dict = feature_builder(
                    rgb=rgb,
                    state=state,
                    action=action,
                    seg=seg,
                    depth_room=depth_room_processed,
                    depth_wrist=depth_wrist_processed,
                )
                dataset.add_frame(frame_dict)

        dataset.save_episode(task=task_prompt)

    print(f"Saving dataset to {final_output_path}")
    if isinstance(feature_builder, GR00TN1FeatureDict):
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "..", "gr00t_n1", "modality.json"),
            final_output_path / "meta" / "modality.json",
        )
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=run_compute_stats)


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
    parser.add_argument(
        "--include_depth",
        action="store_true",
        help="Include depth images in the dataset",
    )
    parser.add_argument(
        "--include_seg",
        action="store_true",
        help="Include segmentation images in the dataset",
    )
    parser.add_argument(
        "--include_video",
        action="store_true",
        help="Include video images in the dataset",
    )
    parser.add_argument(
        "--run_compute_stats",
        type=bool,
        default=False,
        help="Run compute stats (true/false)",
    )

    args = parser.parse_args()

    feature_builder = Pi0FeatureDict(
        image_shape=args.image_shape,
        include_depth=args.include_depth,
        include_seg=args.include_seg,
        include_video=args.include_video,
    )

    main(
        args.data_dir,
        args.repo_id,
        args.task_prompt,
        feature_builder=feature_builder,
        include_depth=args.include_depth,
        include_seg=args.include_seg,
        run_compute_stats=args.run_compute_stats,
    )
