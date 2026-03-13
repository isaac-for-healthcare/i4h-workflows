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

"""Termination helpers for multi-stage tasks."""

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def object_at_destination(
    env: ManagerBasedRLEnv,
    cart_cfg: SceneEntityCfg,
    target_position_x: float,
    target_position_y: float,
    target_position_z: float,
    max_x_separation: float,
    max_y_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """
    Single-stage success check: Object at fixed target position.

    This is a simplified version that only checks if the object has reached
    the target position, without any multi-stage logic or state tracking.

    Args:
        env: Environment
        object_cfg: Config for the object object
        target_position_x/y/z: Fixed target coordinates
        max_x/y/z_separation: Maximum allowed separation in each axis

    Returns:
        Boolean tensor: True when cart is at target position
    """
    # Get cart object
    cart: RigidObject = env.scene[cart_cfg.name]

    # Get cart position relative to environment origin
    cart_pos = cart.data.root_pos_w - env.scene.env_origins

    # Target position (fixed coordinates)
    target_pos = torch.tensor([target_position_x, target_position_y, target_position_z], device=env.device).unsqueeze(0)

    # Calculate separation in each axis
    x_separation = torch.abs(cart_pos[:, 0] - target_pos[0, 0])
    y_separation = torch.abs(cart_pos[:, 1] - target_pos[0, 1])
    z_separation = torch.abs(cart_pos[:, 2] - target_pos[0, 2])

    # Check if cart is within threshold in all axes
    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done
