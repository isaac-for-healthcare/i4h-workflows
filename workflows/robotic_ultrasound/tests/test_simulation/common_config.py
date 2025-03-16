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

from simulation.configs.config import (
    Config,
    FrankaConfig,
    RoomCameraConfig,
    TargetConfig,
    UltraSoundConfig,
    WristCameraConfig,
)
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets

config = Config(
    main_usd_path=robot_us_assets.basic,
    room_camera=RoomCameraConfig(prim_path="/RoomCamera", enabled=True),
    wrist_camera=WristCameraConfig(prim_path="/Franka/panda_hand/geometry/realsense/realsense_camera", enabled=True),
    franka=FrankaConfig(prim_path="/Franka", ik=False, auto_pos=False, enabled=True),
    target=TargetConfig(prim_path="/Target", auto_pos=False, enabled=True),
    ultrasound=UltraSoundConfig(prim_path="/Target", enabled=True),
)
