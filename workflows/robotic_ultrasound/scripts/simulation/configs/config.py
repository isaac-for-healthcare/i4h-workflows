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

import os
from typing import Tuple

from pydantic import BaseModel

domain_id = int(os.environ.get("OVH_DDS_DOMAIN_ID", 1))
physics_dt = float(os.environ.get("OVH_PHYSICS_DT", 1 / 30.0))
period = float(os.environ.get("OVH_DEFAULT_HZ", 1 / 30.0))
h = int(os.environ.get("OVH_HEIGHT", 224))
w = int(os.environ.get("OVH_WIDTH", 224))
random_seed = int(os.environ.get("OVH_RANDOM_SEED", 1234))


class Topic(BaseModel):
    name: str
    domain_id: int = domain_id
    period: float = period


class CameraConfig(BaseModel):
    prim_path: str
    height: int = h
    width: int = w
    range: Tuple[int, int] | None = (20, 200)
    topic_ctrl: Topic | None = Topic(name="topic_camera_ctrl")
    topic_data_rgb: Topic | None = Topic(name="topic_camera_data_rgb")
    topic_data_depth: Topic | None = Topic(name="topic_camera_data_depth")
    enabled: bool = False


class RoomCameraConfig(CameraConfig):
    topic_ctrl: Topic | None = Topic(name="topic_room_camera_ctrl")
    topic_data_rgb: Topic | None = Topic(name="topic_room_camera_data_rgb")
    topic_data_depth: Topic | None = Topic(name="topic_room_camera_data_depth")


class WristCameraConfig(CameraConfig):
    topic_ctrl: Topic | None = Topic(name="topic_wrist_camera_ctrl")
    topic_data_rgb: Topic | None = Topic(name="topic_wrist_camera_data_rgb")
    topic_data_depth: Topic | None = Topic(name="topic_wrist_camera_data_depth")


class FrankaConfig(BaseModel):
    prim_path: str
    ik: bool = False
    topic_ctrl: Topic | None = Topic(name="topic_franka_ctrl")
    topic_info: Topic | None = Topic(name="topic_franka_info")
    auto_pos: bool = False
    enabled: bool = False


class TargetConfig(BaseModel):
    prim_path: str
    topic_ctrl: Topic | None = Topic(name="topic_target_ctrl")
    topic_info: Topic | None = Topic(name="topic_target_info")
    auto_pos: bool = False
    enabled: bool = False


class UltraSoundConfig(BaseModel):
    prim_path: str
    height: int = h
    width: int = w
    topic_ctrl: Topic | None = Topic(name="topic_ultrasound_ctrl")
    topic_info: Topic | None = Topic(name="topic_ultrasound_info")
    topic_data: Topic | None = Topic(name="topic_ultrasound_data")
    enabled: bool = False


class Config(BaseModel):
    main_usd_path: str
    physics_dt: float = physics_dt
    random_seed: int = random_seed
    room_camera: RoomCameraConfig | None = None
    wrist_camera: WristCameraConfig | None = None
    franka: FrankaConfig | None = None
    target: TargetConfig | None = None
    ultrasound: UltraSoundConfig | None = None
