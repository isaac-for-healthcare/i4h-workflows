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

"""Workflow-specific background assets."""

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from simulation.assets.assets import MAIN_BACKGROUND_USD


@register_asset
class PreOpBackground(LibraryBackground):
    """Background scene for the Pre-Operative environment."""

    name = "pre_op"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = MAIN_BACKGROUND_USD
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.5  # Minimum Z for drop detection.

    def __init__(self):
        super().__init__()
