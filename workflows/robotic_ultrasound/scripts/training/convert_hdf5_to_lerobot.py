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

For GR00T N1 datasets, the script automatically reorganizes videos into the expected folder structure:
├─meta
│ ├─episodes.jsonl
│ ├─modality.json # -> GR00T LeRobot specific
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   └─observation.images.room
│     └─episode_000000.mp4
│   └─observation.images.wrist
│     └─episode_000000.mp4
└─data
  └─chunk-000
    └─episode_000000.parquet
"""

import argparse
import glob
import json
import os
import re
import shutil
import warnings

import h5py
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from PIL import Image


def reorganize_videos_for_gr00t(dataset_path):
    """
    Reorganizes video files to match the expected GR00T folder structure.

    Expected structure:
    videos/chunk-000/observation.images.room/episode_XXXXXX.mp4
    videos/chunk-000/observation.images.wrist/episode_XXXXXX.mp4

    Parameters:
    - dataset_path: Path to the dataset directory
    """
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        print("No videos directory found, skipping reorganization.")
        return

    # Find all chunk directories
    chunk_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]

    for chunk_dir in chunk_dirs:
        print(f"Reorganizing videos in {chunk_dir}")

        # First, let's see what the current structure looks like
        print(f"Current structure in {chunk_dir}:")
        for item in chunk_dir.iterdir():
            print(f"  {item.name}")

        # Find all video files recursively in the chunk directory
        video_files = list(chunk_dir.rglob("*.mp4"))
        print(f"Found {len(video_files)} video files")

        # Group videos by episode number
        episode_videos = {}
        for video_file in video_files:
            # Extract episode number from filename
            episode_match = re.search(r"episode_(\d+)\.mp4", video_file.name)
            if not episode_match:
                print(f"Warning: Could not parse episode number from {video_file.name}")
                continue

            episode_num = episode_match.group(1)
            if episode_num not in episode_videos:
                episode_videos[episode_num] = []
            episode_videos[episode_num].append(video_file)

        # For each episode, organize the videos by feature type
        for episode_num, videos in episode_videos.items():
            print(f"Processing episode {episode_num} with {len(videos)} videos")

            # Sort videos to ensure consistent ordering (room first, wrist second)
            videos.sort()

            # Map videos to feature keys
            feature_mapping = {0: "observation.images.room", 1: "observation.images.wrist"}

            for i, video_file in enumerate(videos):
                if i in feature_mapping:
                    feature_key = feature_mapping[i]

                    # Create the feature directory
                    feature_dir = chunk_dir / feature_key
                    feature_dir.mkdir(exist_ok=True)

                    # Move the video file to the correct location
                    target_path = feature_dir / f"episode_{int(episode_num):06d}.mp4"
                    if video_file != target_path:
                        shutil.move(str(video_file), str(target_path))
                        print(f"Moved {video_file.name} to {target_path}")
                else:
                    print(f"Warning: No feature mapping for video {i} in episode {episode_num}")

        # Clean up any empty directories that might be left behind
        for item in chunk_dir.iterdir():
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                    print(f"Removed empty directory: {item}")
                except OSError:
                    pass  # Directory might not be empty or might have hidden files


def update_episode_metadata_for_gr00t(dataset_path):
    """
    Updates the episodes.jsonl file to reflect the correct video paths after reorganization.

    Parameters:
    - dataset_path: Path to the dataset directory
    """
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        print("No episodes.jsonl file found, skipping metadata update.")
        return

    # Read all episodes
    episodes = []
    with open(episodes_file, "r") as f:
        for line in f:
            episodes.append(json.loads(line.strip()))

    # Update video paths for each episode
    updated_episodes = []
    for i, episode in enumerate(episodes):
        # Try to get episode_id, but fall back to using the index if it's empty
        episode_id = episode.get("episode_id", "")
        if episode_id:
            # Try to extract episode number from episode_id
            episode_num = episode_id.replace("episode_", "")
            try:
                episode_num = int(episode_num)
            except ValueError:
                # If conversion fails, use the index
                episode_num = i
        else:
            # If episode_id is empty, use the index
            episode_num = i

        # Update video paths to match the new structure
        video_paths = {}
        for feature_key in ["observation.images.room", "observation.images.wrist"]:
            video_path = f"videos/chunk-000/{feature_key}/episode_{episode_num:06d}.mp4"
            # Extract the short name for the video_paths dict
            if "room" in feature_key:
                video_paths["room"] = video_path
            elif "wrist" in feature_key:
                video_paths["wrist"] = video_path

        # Update the episode data
        episode["video_paths"] = video_paths
        updated_episodes.append(episode)

    # Write the updated episodes back
    with open(episodes_file, "w") as f:
        for episode in updated_episodes:
            f.write(json.dumps(episode) + "\n")

    print(f"✅ Updated episode metadata in {episodes_file}")


def verify_gr00t_folder_structure(dataset_path):
    """
    Verifies that the dataset has the correct GR00T folder structure.

    Parameters:
    - dataset_path: Path to the dataset directory
    """
    print("\n🔍 Verifying GR00T folder structure...")

    # Check for required directories
    required_dirs = ["meta", "videos", "data"]
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"❌ Missing required directory: {dir_name}")
            return False
        else:
            print(f"✅ Found directory: {dir_name}")

    # Check for required meta files
    meta_files = ["episodes.jsonl", "modality.json", "info.json", "tasks.jsonl"]
    for file_name in meta_files:
        file_path = dataset_path / "meta" / file_name
        if not file_path.exists():
            print(f"❌ Missing required meta file: {file_name}")
            return False
        else:
            print(f"✅ Found meta file: {file_name}")

    # Check video structure
    videos_dir = dataset_path / "videos"
    chunk_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]

    if not chunk_dirs:
        print("❌ No chunk directories found in videos/")
        return False

    for chunk_dir in chunk_dirs:
        print(f"📁 Checking chunk directory: {chunk_dir.name}")

        # Check for feature directories
        feature_dirs = ["observation.images.room", "observation.images.wrist"]
        for feature_dir_name in feature_dirs:
            feature_dir = chunk_dir / feature_dir_name
            if not feature_dir.exists():
                print(f"❌ Missing feature directory: {feature_dir_name}")
                return False

            # Check for video files
            video_files = list(feature_dir.glob("*.mp4"))
            if not video_files:
                print(f"❌ No video files found in {feature_dir_name}")
                return False

            print(f"✅ Found {len(video_files)} videos in {feature_dir_name}")

    # Check data structure
    data_dir = dataset_path / "data"
    data_chunk_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]

    if not data_chunk_dirs:
        print("❌ No chunk directories found in data/")
        return False

    for chunk_dir in data_chunk_dirs:
        parquet_files = list(chunk_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"❌ No parquet files found in {chunk_dir.name}")
            return False
        print(f"✅ Found {len(parquet_files)} parquet files in {chunk_dir.name}")

    print("✅ GR00T folder structure verification completed successfully!")
    return True


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
        frame_data[self.room_image_key] = resize_with_pad(rgb[0], img_h, img_w)
        frame_data[self.wrist_image_key] = resize_with_pad(rgb[1], img_h, img_w)
        frame_data[self.state_key] = state
        frame_data[self.action_key] = action  # Use subclass-defined action_key

        if seg is not None and self.seg_room_key in current_features:
            frame_data[self.seg_room_key] = resize_with_pad(seg[0], img_h, img_w, method=Image.NEAREST)
        if seg is not None and self.seg_wrist_key in current_features:
            frame_data[self.seg_wrist_key] = resize_with_pad(seg[1], img_h, img_w, method=Image.NEAREST)

        if depth_room is not None and self.depth_room_key in current_features:
            frame_data[self.depth_room_key] = resize_with_pad(depth_room, img_h, img_w).squeeze(2)
        if depth_wrist is not None and self.depth_wrist_key in current_features:
            frame_data[self.depth_wrist_key] = resize_with_pad(depth_wrist, img_h, img_w).squeeze(2)

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


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


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
    skip_video_reorganization: bool = False,
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
            os.path.join(os.path.dirname(__file__), "gr00t_n1", "modality.json"),
            final_output_path / "meta" / "modality.json",
        )
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=run_compute_stats)

    # Post-process the info.json file to add the required "info" key with "video.fps" for GR00T compatibility
    if isinstance(feature_builder, GR00TN1FeatureDict):
        info_path = final_output_path / "meta" / "info.json"
        with open(info_path, "r") as f:
            info_data = json.load(f)

        # Get the fps from the root level
        fps = info_data.get("fps", 30)

        # Add "info" key with "video.fps" to each video feature
        for feature_key, feature_data in info_data["features"].items():
            if "images" in feature_key:  # This is a video feature
                feature_data["info"] = {"video.fps": fps}

        # Write the modified info.json back
        with open(info_path, "w") as f:
            json.dump(info_data, f, indent=4)

        print(f"✅ Added GR00T-compatible video metadata to {info_path}")

    # Reorganize videos for GR00T datasets
    if isinstance(feature_builder, GR00TN1FeatureDict) and not skip_video_reorganization:
        print("\n🔄 Reorganizing videos for GR00T compatibility...")
        reorganize_videos_for_gr00t(final_output_path)
        update_episode_metadata_for_gr00t(final_output_path)
        verify_gr00t_folder_structure(final_output_path)
    elif isinstance(feature_builder, GR00TN1FeatureDict) and skip_video_reorganization:
        print("\n⚠️ Skipping video reorganization (debug mode)")


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
        "--feature_builder_type",
        type=str,
        default="pi0",
        choices=["pi0", "gr00tn1"],
        help="Type of feature builder to use (pi0 or gr00tn1).",
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
    parser.add_argument(
        "--skip_video_reorganization",
        action="store_true",
        help="Skip video reorganization for GR00T datasets (for debugging)",
    )

    args = parser.parse_args()

    # Instantiate the feature builder based on args
    if args.feature_builder_type == "gr00tn1":
        feature_builder = GR00TN1FeatureDict(
            image_shape=args.image_shape,
            include_depth=args.include_depth,
            include_seg=args.include_seg,
            include_video=args.include_video,
        )
    else:
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
        skip_video_reorganization=args.skip_video_reorganization,
    )
