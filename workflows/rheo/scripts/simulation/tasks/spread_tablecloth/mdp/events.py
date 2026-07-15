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

"""Custom events for the spread_tablecloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["reset_robot_to_default_joint_positions"]


def reset_robot_to_default_joint_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints/root directly to defaults (bypasses PD to avoid arm swing)."""
    if len(env_ids) == 0:
        return

    robot = env.scene[robot_cfg.name]

    default_joint_pos = wp.to_torch(robot.data.default_joint_pos)[env_ids].clone()
    default_joint_vel = wp.to_torch(robot.data.default_joint_vel)[env_ids].clone()

    robot.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)

    default_root_pose = wp.to_torch(robot.data.default_root_pose)[env_ids].clone()
    default_root_velocity = wp.to_torch(robot.data.default_root_vel)[env_ids].clone()
    robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=default_root_velocity, env_ids=env_ids)
