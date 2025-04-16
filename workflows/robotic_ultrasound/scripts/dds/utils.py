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

def get_default_qos_provider_path():
    """Get the default QoS provider path."""
    return os.path.join(os.path.dirname(__file__), "qos_profiles.xml")


def get_default_qos_library():
    """Get the default QoS library."""
    return "RobotUsLib"


def get_default_transport_profile():
    """Get the default transport profile."""
    return f"{get_default_qos_library()}::RobotUsProfile"


def get_default_entity_profile(topic: str):
    """Get the default entity profile for a given topic."""
    mapping = {
        "topic_room_camera_data_rgb": "CameraInfo",
        "topic_room_camera_data_depth": "CameraInfo",
        "topic_wrist_camera_data_rgb": "CameraInfo",
        "topic_wrist_camera_data_depth": "CameraInfo",
        "topic_room_camera_ctrl": "CameraCtrl",
        "topic_wrist_camera_ctrl": "CameraCtrl",
        "topic_franka_info": "FrankaInfo",
        "topic_franka_ctrl": "FrankaCtrl",
        "topic_ultrasound_info": "UspInfo",
        "topic_ultrasound_data": "UspData",
    }
    
    if topic not in mapping:
        raise ValueError(f"No default entity profile found for topic: {topic}")
    return f"{get_default_qos_library()}::{mapping[topic]}"
