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
import unittest
from importlib.util import find_spec
from unittest import skipUnless

from holoscan_apps.realsense.camera import RealsenseApp
from holoscan_apps.realsense.enumerate import count_devices

from ..helpers import requires_rti

DEVICES_AVAILABLE = count_devices() > 0


@requires_rti
@skipUnless(DEVICES_AVAILABLE, "No Intel RealSense devices found")
class TestRealSenseCamera(unittest.TestCase):
    def test_realsense_camera(self):
        domain_id = 4321
        height, width = 480, 640
        topic_rgb = "topic_test_camera_rgb"
        topic_depth = "topic_test_camera_depth"
        device_idx = 0
        framerate = 30
        test = False
        count = 10

        app = RealsenseApp(
            domain_id,
            height,
            width,
            topic_rgb,
            topic_depth,
            device_idx,
            framerate,
            test,
            count,
        )
        app.run()
