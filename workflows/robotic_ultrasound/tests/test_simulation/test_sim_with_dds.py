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

import time
import unittest

import numpy as np
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import SubscriberWithCallback
from helpers import requires_rti

"""
Must immediately execute the `sim_with_dds.py` in another process after executing this test.

"""


@requires_rti
class TestSimWithDDS(unittest.TestCase):
    def setUp(self):
        topic1 = "topic_room_camera_data_rgb"
        topic2 = "topic_wrist_camera_data_rgb"
        topic3 = "topic_franka_info"

        def dds_callback1(topic, data):
            self.assertEqual(topic, topic1)
            o: CameraInfo = data
            room_cam = np.frombuffer(o.data, dtype=np.uint8)
            self.assertTupleEqual(room_cam.shape, (150528,))
            self.test_1_pass = True

        def dds_callback2(topic, data):
            self.assertEqual(topic, topic2)
            o: CameraInfo = data
            room_cam = np.frombuffer(o.data, dtype=np.uint8)
            self.assertTupleEqual(room_cam.shape, (150528,))
            self.test_2_pass = True

        def dds_callback3(topic, data):
            self.assertEqual(topic, topic3)
            o: FrankaInfo = data
            self.assertEqual(len(o.joints_state_positions), 7)
            self.test_3_pass = True

        self.domain_id = 101
        hz = 1 / 30
        self.test_1_pass = self.test_2_pass = self.test_3_pass = False
        self.reader1 = SubscriberWithCallback(dds_callback1, self.domain_id, topic1, CameraInfo, hz)
        self.reader2 = SubscriberWithCallback(dds_callback2, self.domain_id, topic2, CameraInfo, hz)
        self.reader3 = SubscriberWithCallback(dds_callback3, self.domain_id, topic3, FrankaInfo, hz)

        self.reader1.start()
        self.reader2.start()
        self.reader3.start()
        time.sleep(1.0)

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "reader1"):
            self.reader1.stop()
            del self.reader1
        if hasattr(self, "reader2"):
            self.reader2.stop()
            del self.reader2
        if hasattr(self, "reader3"):
            self.reader3.stop()
            del self.reader3
        time.sleep(0.1)  # Allow time for cleanup

    def test_init(self):
        """Test Subscriber initialization"""
        self.assertEqual(self.reader1.domain_id, self.domain_id)

    def test_read(self):
        """Test read method with actual DDS communication"""
        for _ in range(60):
            if self.test_1_pass and self.test_2_pass and self.test_3_pass:
                break
            time.sleep(1.0)
        self.assertTrue(self.test_1_pass)
        self.assertTrue(self.test_2_pass)
        self.assertTrue(self.test_3_pass)


if __name__ == "__main__":
    unittest.main()
