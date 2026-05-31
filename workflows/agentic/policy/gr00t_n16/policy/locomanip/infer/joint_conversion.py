# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition


def remap_policy_joints_to_sim_joints(
    policy_joints: dict[str, np.array],
    policy_joints_config: dict[str, list[str]],
    sim_joints_config: dict[str, int],
    device: torch.device,
) -> JointsAbsPosition:
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
                joint_group = "left_hand" if "hand" in joint_name else "left_arm"
            case "right":
                joint_group = "right_hand" if "hand" in joint_name else "right_arm"
            case "waist":
                joint_group = "waist"
            case _:
                continue
        if joint_name in policy_joints_config[joint_group]:
            policy_key = None
            if f"action.{joint_group}" in policy_joints:
                policy_key = f"action.{joint_group}"
            elif joint_group in policy_joints:
                policy_key = joint_group
            if policy_key is not None:
                gr00t_index = policy_joints_config[joint_group].index(joint_name)
                data[..., joint_index] = torch.from_numpy(policy_joints[policy_key][..., gr00t_index]).to(device)

    return JointsAbsPosition(joints_pos=data, joints_order_config=sim_joints_config)
