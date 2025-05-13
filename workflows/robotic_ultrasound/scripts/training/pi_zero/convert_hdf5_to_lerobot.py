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
import numpy as np
from PIL import Image
import h5py
import tqdm
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from openpi_client import image_tools


def colorize_segmentation_mask(seg_mask, color_map=None):
    """
    Converts a segmentation mask to a colored RGB image.
    Handles batch dimensions and multiple views.
    
    Parameters:
    -----------
    seg_mask : numpy.ndarray or h5py.Dataset
        Segmentation mask with integer labels, shape can be:
        (H, W) or (H, W, 1) for single image
        (B, H, W) or (B, H, W, 1) for batch of images
        (B, V, H, W) or (B, V, H, W, 1) for batch with multiple views
    color_map : dict, optional
        Dictionary mapping label values to RGB colors.
        If None, uses a default color map.
        
    Returns:
    --------
    numpy.ndarray
        Colored RGB image with same batch dimensions as input but with 3 channels:
        (H, W, 3) or (B, H, W, 3) or (B, V, H, W, 3)
    """
    # Convert h5py.Dataset to numpy array if needed
    if isinstance(seg_mask, h5py.Dataset):
        seg_mask = np.array(seg_mask)
    
    original_shape = seg_mask.shape
    
    # Default color map if none provided
    if color_map is None:
        color_map = {
            0: (0, 0, 0),        # Black
            1: (255, 0, 0),      # Red
            2: (0, 255, 0),      # Green
            3: (0, 0, 255),      # Blue
            4: (255, 255, 0)     # Yellow
        }
    
    # Handle different input shapes
    if seg_mask.ndim == 2:  # (H, W)
        # Convert to uint8 if needed
        seg_mask = seg_mask.astype(np.uint8)
        
        # Create RGB image
        H, W = seg_mask.shape
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Apply colors based on labels
        for label, color in color_map.items():
            colored_mask[seg_mask == label] = np.array(color, dtype=np.uint8)
            
        return colored_mask
    
    elif seg_mask.ndim == 3:
        if seg_mask.shape[-1] == 1:  # (H, W, 1)
            return colorize_segmentation_mask(seg_mask[:, :, 0], color_map)
        else:  # (B, H, W)
            B, H, W = seg_mask.shape
            colored_masks = np.zeros((B, H, W, 3), dtype=np.uint8)
            for b in range(B):
                colored_masks[b] = colorize_segmentation_mask(seg_mask[b], color_map)
            return colored_masks
    
    elif seg_mask.ndim == 4:
        if seg_mask.shape[-1] == 1:  # (B, H, W, 1)
            return colorize_segmentation_mask(seg_mask[:, :, :, 0], color_map)
        else:  # (B, V, H, W)
            B, V, H, W = seg_mask.shape
            colored_masks = np.zeros((B, V, H, W, 3), dtype=np.uint8)
            for b in range(B):
                for v in range(V):
                    colored_masks[b, v] = colorize_segmentation_mask(seg_mask[b, v], color_map)
            return colored_masks
    
    elif seg_mask.ndim == 5:  # (B, V, H, W, 1)
        return colorize_segmentation_mask(seg_mask[:, :, :, :, 0], color_map)
    
    else:
        raise ValueError(f"Unsupported shape: {original_shape}")


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
    robot_type: str = "panda",
    fps: int = 30,
    image_shape: tuple[int, int, int] = (224, 224, 3),
    state_shape: tuple[int, ...] = (7,),
    actions_shape: tuple[int, ...] = (6,),
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
    include_depth: bool = False,
    include_seg: bool = False,
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
    - include_depth: Whether to include depth images in the dataset.
    - include_seg: Whether to include segmentation images in the dataset.

    Returns:
    - An instance of LeRobotDataset configured with the specified parameters.
    """

    if os.path.isdir(output_path):
        raise Exception(f"Output path {output_path} already exists.")

    features = {
            "observation.images.room": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": actions_shape,
                "names": ["actions"],
            },
            "observation.images.room": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },

    }
    if include_depth:
        features.update({
            "observation.depth.room": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.depth.wrist": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.depth.room": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.depth.wrist": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            
        })
    if include_seg:
        features.update({
            "observation.seg.room": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.seg.wrist": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
                "observation.seg.room": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.seg.wrist": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
        })
        
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
    image_shape=(224, 224, 3),
    include_depth: bool = False,
    include_seg: bool = False,
    run_compute_stats: bool = False,
    include_camera_info: bool = True,
    save_seg_depth_npz: bool = True,
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
    dataset = create_lerobot_dataset(final_output_path, image_shape=image_shape, include_depth=include_depth, include_seg=include_seg, **dataset_config_kwargs)

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
                if include_seg:
                    seg = f[root_name]["observations/seg_images"][step]

                if include_depth:
                    depth = f[root_name]["observations/depth_images"][step]
                    depth_normalized = normalize_depth_image(depth)
                
                frame_dict = {
                        "observation.images.room": image_tools.resize_with_pad(rgb[0], image_shape[0], image_shape[1]),
                        "observation.images.wrist": image_tools.resize_with_pad(rgb[1], image_shape[0], image_shape[1]),
                        "observation.state": f[root_name]["abs_joint_pos"][step],
                        "action": f[root_name]["action"][step],
                    }
                if include_seg:
                    frame_dict["observation.seg.room"] = image_tools.resize_with_pad(
                        seg[0], image_shape[0], image_shape[1], method=Image.NEAREST)
                    frame_dict["observation.seg.wrist"] = image_tools.resize_with_pad(
                        seg[1], image_shape[0], image_shape[1], method=Image.NEAREST)
                if include_depth:
                    frame_dict["observation.depth.room"] = image_tools.resize_with_pad(depth_normalized[0], image_shape[0], image_shape[1]).squeeze(2)
                    frame_dict["observation.depth.wrist"] = image_tools.resize_with_pad(depth_normalized[1], image_shape[0], image_shape[1]).squeeze(2)
                dataset.add_frame(frame_dict)

            if include_camera_info:
                output_path = final_output_path / "camera_info"
                os.makedirs(output_path, exist_ok=True)
                room_camera_intrinsic_matrices = f['data/demo_0/observations/room_camera_intrinsic_matrices']  # (n_frames, 3, 3)
                room_camera_pos = f['data/demo_0/observations/room_camera_pos']  # (n_frames, 3)
                room_camera_quat = f['data/demo_0/observations/room_camera_quat_w_ros']  # (n_frames, 4)
                save_dict = {
                    "room_camera_intrinsic_matrices": room_camera_intrinsic_matrices,
                    "room_camera_pos": room_camera_pos,
                    "room_camera_quat": room_camera_quat
                }
                np.savez(f"{output_path}/room_camera_para_{episode_idx}.npz", **save_dict)

                wrist_camera_intrinsic_matrices = f['data/demo_0/observations/wrist_camera_intrinsic_matrices']  # (n_frames, 3, 3)
                wrist_camera_pos = f['data/demo_0/observations/wrist_camera_pos']  # (n_frames, 3)
                wrist_camera_quat = f['data/demo_0/observations/wrist_camera_quat_w_ros']  # (n_frames, 4)
                save_dict = {
                    "wrist_camera_intrinsic_matrices": wrist_camera_intrinsic_matrices,
                    "wrist_camera_pos": wrist_camera_pos,
                    "wrist_camera_quat": wrist_camera_quat
                }
                
                np.savez(f"{output_path}/wrist_camera_para_{episode_idx}.npz", **save_dict)
            if save_seg_depth_npz:
                save_dict = {
                    "depth_images": f['data/demo_0/observations/depth_images'],
                    "seg_images": f['data/demo_0/observations/seg_images'],
                }

                np.savez(f"{output_path}/seg_depth_images_{episode_idx}.npz", **save_dict)
        dataset.save_episode(task=task_prompt)

    print(f"Saving dataset to {final_output_path}")
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
        help="Shape of the image data as a comma-separated string, e.g., '480,640,3'",
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
        "--run_compute_stats",
        default=False,
        help="Run compute stats",
    )
    parser.add_argument(
        "--include_camera_info",
        default=True,
        help="Include camera info in the dataset",
    )
    parser.add_argument(
        "--save_seg_depth_npz",
        default=True,
        help="Save seg and depth images as npz files",
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.repo_id, args.task_prompt, image_shape=args.image_shape, include_depth=args.include_depth, include_seg=args.include_seg, run_compute_stats=args.run_compute_stats,
         include_camera_info=args.include_camera_info,
         save_seg_depth_npz=args.save_seg_depth_npz)
