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

import unittest

from holoscan_apps.realsense.camera import RealsenseApp


class TestRealSenseCameraUnit(unittest.TestCase):
    def test_realsense_app(self):
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

        self.assertIsNotNone(app)
        self.assertEqual(app.width, width)
        self.assertEqual(app.height, height)
        self.assertEqual(app.framerate, framerate)
        self.assertEqual(app.topic_rgb, topic_rgb)
        self.assertEqual(app.topic_depth, topic_depth)
