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

import torch


def capture_camera_images(env, cam_names, include_seg=False, device="cuda"):
    """
    Captures RGB and depth images from specified cameras

    Args:
        env: The environment containing the cameras
        cam_names (list): List of camera names to capture from
        include_seg (bool): Whether to include semantic segmentation images
        device (str): Device to use for tensor operations

    Returns:
        tuple: If include_seg is False:
            (stacked_rgbs, stacked_depths, None) - Tensors of shape (1, num_cams, H, W, 3),
            (1, num_cams, H, W), and None
        If include_seg is True:
            (stacked_rgbs, stacked_depths, stacked_segs) - Tensors of shape (1, num_cams, H, W, 3),
            (1, num_cams, H, W), and (1, num_cams, H, W)
    """
    _depths, rgbs, segs = [], [], []
    for cam_name in cam_names:
        camera_data = env.unwrapped.scene[cam_name].data

        # Extract RGB and depth images
        rgb = camera_data.output["rgb"][..., :3].squeeze(0)

        if include_seg:
            seg = camera_data.output["semantic_segmentation"][..., :3].squeeze(0)
            segs.append(seg)

        # Append to lists
        rgbs.append(rgb)

    # Stack results
    stacked_rgbs = torch.stack(rgbs).unsqueeze(0)
    if include_seg:
        stacked_segs = torch.stack(segs).unsqueeze(0)
        return stacked_rgbs, stacked_segs
    return stacked_rgbs, None


def get_joint_states(env):
    """Get the robot joint states from the environment."""
    robot_data = env.unwrapped.scene["robot"].data
    robot_joint_pos = robot_data.joint_pos
    return robot_joint_pos.cpu().numpy()
