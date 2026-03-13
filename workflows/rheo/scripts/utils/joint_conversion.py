# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition


def remap_policy_joints_to_sim_joints(
    policy_joints: dict[str, np.array],
    policy_joints_config: dict[str, list[str]],
    sim_joints_config: dict[str, int],
    device: torch.device,
) -> JointsAbsPosition:
    """
    Remap the actions joints from policy joint orders to simulation joint orders
    """
    # Validate all values in policy_joint keys have the same shape and save the shape to init data
    policy_joint_shape = None
    for key, joint_pos in policy_joints.items():
        if policy_joint_shape is None:
            policy_joint_shape = joint_pos.shape
        else:
            if joint_pos.ndim != 3:
                raise ValueError(f"Expected 3D tensor for joint '{key}', got {joint_pos.ndim}D")
            if joint_pos.shape[:2] != policy_joint_shape[:2]:
                raise ValueError(
                    f"Shape mismatch for joint '{key}': expected {policy_joint_shape[:2]}, got {joint_pos.shape[:2]}"
                )

    if policy_joint_shape is None:
        raise ValueError("policy_joints dict is empty, cannot determine joint shape")
    data = torch.zeros([policy_joint_shape[0], policy_joint_shape[1], len(sim_joints_config)], device=device)
    for joint_name, joint_index in sim_joints_config.items():
        match joint_name.split("_")[0]:
            case "left":
                # For G1
                if "hand" in joint_name:
                    joint_group = "left_hand"
                else:
                    joint_group = "left_arm"
            case "right":
                # For G1
                if "hand" in joint_name:
                    joint_group = "right_hand"
                else:
                    joint_group = "right_arm"
            case "waist":
                # For G1
                joint_group = "waist"
            case _:
                continue
        if joint_name in policy_joints_config[joint_group]:
            # Try both with and without "action." prefix for compatibility with GR00T N1.6
            policy_key = None
            if f"action.{joint_group}" in policy_joints:
                policy_key = f"action.{joint_group}"
            elif joint_group in policy_joints:
                policy_key = joint_group

            if policy_key is not None:
                gr00t_index = policy_joints_config[joint_group].index(joint_name)
                data[..., joint_index] = torch.from_numpy(policy_joints[policy_key][..., gr00t_index]).to(device)

    sim_joints = JointsAbsPosition(joints_pos=data, joints_order_config=sim_joints_config)
    return sim_joints
