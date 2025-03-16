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

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""functions to reset the robot joints"""

from __future__ import annotations

import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg


def reset_panda_joints_by_fraction_of_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fraction: float = 0.1,
):
    """
    Reset the robot joints with offsets sampled from a fraction of the joint limits.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # Get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()  # shape: [num_envs, num_dofs]
    joint_vel = asset.data.default_joint_vel[env_ids].clone()  # shape: [num_envs, num_dofs]

    # get the ranges for each joint from the asset
    joint_limits = asset.data.default_joint_limits[env_ids].clone()  # shape: [num_envs, num_dofs, 2]
    # make the joint_sample_ranges a central fraction of the joint limits
    joint_sample_ranges = joint_limits * fraction

    # Safety checks
    num_dofs = joint_pos.shape[1]
    assert (
        joint_sample_ranges.shape[1] == num_dofs
    ), f"joint_sample_ranges must have {num_dofs} elements, got {len(joint_sample_ranges)}"

    # Generate random joint positions within the limits
    joint_pos_delta = torch.zeros_like(joint_pos, device=joint_pos.device)
    lower = joint_sample_ranges[:, :, 0]
    upper = joint_sample_ranges[:, :, 1]

    # Sample random joint positions for the n x num_dofs environments
    joint_pos_delta = torch.rand(joint_pos.shape, device=joint_pos.device) * (upper - lower) + lower
    # add the random offsets to the default joint positions
    joint_pos += joint_pos_delta

    # Clamp the joint positions to the joint limits
    joint_pos = torch.max(joint_pos, joint_limits[:, :, 0])
    joint_pos = torch.min(joint_pos, joint_limits[:, :, 1])

    # Write the new joint positions and velocities to the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
