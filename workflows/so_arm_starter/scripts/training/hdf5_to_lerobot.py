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

import argparse
import json
import os

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

"""NOTE: Please use the environment of lerobot."""

# Feature definition for so101_follower
FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.images.room": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# preprocess actions and joint pos
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    joint_pos = joint_pos / np.pi * 180
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        # Normalize from isaaclab range to [0,1], then scale to lerobot range
        normalized = (joint_pos[:, i] - isaaclab_min) / (isaaclab_max - isaaclab_min)
        joint_pos[:, i] = normalized * (lerobot_max - lerobot_min) + lerobot_min
    return joint_pos


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Isaac Lab HDF5 data to LeRobot dataset format")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="datasets/so101_surgery_scissors",
        help="Repository ID for the dataset (default: datasets/so101_surgery_scissors)",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="./datasets/dataset.hdf5",
        help="Path to the HDF5 file (default: ./datasets/dataset.hdf5)",
    )
    parser.add_argument("--robot_type", type=str, default="so101_follower", help="Robot type (default: so101_follower)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument(
        "--task_description",
        type=str,
        default="Grip the scissors and put it into the tray",
        help="Task description (default: Grip the scissors and put it into the tray)",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push dataset to Hugging Face Hub (default: False)")
    return parser.parse_args()


def create_modality_json(dataset_path):
    """Create the modality.json file required for GR00T training."""
    meta_dir = os.path.join(dataset_path, "meta")
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir, exist_ok=True)

    # Create modality.json matching the so101_follower robot type
    modality = {
        "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
        "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
        "video": {
            "room": {"original_key": "observation.images.room"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }

    modality_path = os.path.join(meta_dir, "modality.json")
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=4)

    print(f"Created modality.json at: {modality_path}")
    return modality_path


def convert_isaaclab_to_lerobot():
    """Convert Isaac Lab HDF5 data to LeRobot dataset format"""
    args = parse_args()

    repo_id = args.repo_id
    robot_type = args.robot_type
    fps = args.fps
    hdf5_files = [args.hdf5_path]
    task_description = args.task_description
    push_to_hub = args.push_to_hub

    # Validate HDF5 file exists
    if not os.path.exists(args.hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {args.hdf5_path}")

    print("Converting dataset:")
    print(f"  Repository ID: {repo_id}")
    print(f"  HDF5 file: {args.hdf5_path}")
    print(f"  Robot type: {robot_type}")
    print(f"  Task: {task_description}")
    print(f"  FPS: {fps}")
    print(f"  Push to hub: {push_to_hub}")

    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=FEATURES,
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} is not successful, skip it")
                    continue

                try:
                    actions = np.array(demo_group["obs/actions"])
                    joint_pos = np.array(demo_group["obs/joint_pos"])
                    room_images = np.array(demo_group["obs/room"])
                    wrist_images = np.array(demo_group["obs/wrist"])
                except KeyError:
                    print(f"Demo {demo_name} is not valid, skip it")
                    continue

                # preprocess actions and joint pos
                actions = preprocess_joint_pos(actions)
                joint_pos = preprocess_joint_pos(joint_pos)

                assert actions.shape[0] == joint_pos.shape[0] == room_images.shape[0] == wrist_images.shape[0]
                total_state_frames = actions.shape[0]

                for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
                    frame = {
                        "action": actions[frame_index],
                        "observation.state": joint_pos[frame_index],
                        "observation.images.room": room_images[frame_index],
                        "observation.images.wrist": wrist_images[frame_index],
                    }
                    dataset.add_frame(frame=frame, task=task_description)
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saving episode {now_episode_index} successfully")

    if push_to_hub:
        dataset.push_to_hub()

    # Create modality.json file for GR00T compatibility
    print("\nCreating modality.json for GR00T compatibility...")
    create_modality_json(dataset.root)


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
