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

"""Joint state observations for the spread_tablecloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_robot_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return [pos | vel | torque] concatenated along the joint axis."""
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    joint_vel = wp.to_torch(env.scene["robot"].data.joint_vel)
    joint_torque = wp.to_torch(env.scene["robot"].data.applied_torque)

    return torch.cat([joint_pos, joint_vel, joint_torque], dim=-1)
