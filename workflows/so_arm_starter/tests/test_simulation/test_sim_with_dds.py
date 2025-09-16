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

import threading
import time
import unittest

import numpy as np
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.soarm_ctrl import SOARM101CtrlInput
from dds.schemas.soarm_info import SOARM101Info
from dds.subscriber import SubscriberWithCallback
from helpers import requires_rti

"""
Self-contained DDS communication test with mock publishers.
No need to run sim_with_dds.py separately - this test creates its own mock data publishers.

"""


class MockControlPublisher(Publisher):
    """Mock publisher to send dummy control commands to keep sim_with_dds.py running."""

    def __init__(self, domain_id: int, topic: str = "topic_soarm_ctrl"):
        super().__init__(topic, SOARM101CtrlInput, 1 / 60, domain_id)

    def produce(self, dt: float, sim_time: float):
        """Produce dummy control commands."""
        output = SOARM101CtrlInput()
        # Send zero joint positions (6 joints for SO-ARM101)
        output.joint_positions = [0.0] * 6
        output.joint_velocities = [0.0] * 6
        output.joint_efforts = [0.0] * 6
        return output


class MockCameraPublisher(Publisher):
    """Mock publisher for camera data."""

    def __init__(self, domain_id: int, topic: str, camera_name: str):
        super().__init__(topic, CameraInfo, 1 / 60, domain_id)
        self.camera_name = camera_name

    def produce(self, dt: float, sim_time: float):
        """Produce dummy camera data."""
        output = CameraInfo()
        output.height = 480
        output.width = 640
        output.focal_len = 12.0

        dummy_image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        output.data = dummy_image.tobytes()

        return output


class MockJointStatePublisher(Publisher):
    """Mock publisher for joint state data."""

    def __init__(self, domain_id: int, topic: str = "topic_soarm_info"):
        super().__init__(topic, SOARM101Info, 1 / 60, domain_id)

    def produce(self, dt: float, sim_time: float):
        """Produce dummy joint state data."""
        output = SOARM101Info()

        output.joints_state_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        output.joints_state_velocities = [0.0] * 6
        return output


@requires_rti
class TestSimWithDDS(unittest.TestCase):
    def setUp(self):
        topic1 = "topic_room_camera_data_rgb"
        topic2 = "topic_wrist_camera_data_rgb"
        topic3 = "topic_soarm_info"

        def dds_callback1(topic, data):
            self.assertEqual(topic, topic1)
            o: CameraInfo = data
            room_cam = np.frombuffer(o.data, dtype=np.uint8)
            self.assertTupleEqual(room_cam.shape, (921600,))  # 480*640*3 for SO-ARM101
            self.test_1_pass = True

        def dds_callback2(topic, data):
            self.assertEqual(topic, topic2)
            o: CameraInfo = data
            wrist_cam = np.frombuffer(o.data, dtype=np.uint8)
            self.assertTupleEqual(wrist_cam.shape, (921600,))  # 480*640*3 for SO-ARM101
            self.test_2_pass = True

        def dds_callback3(topic, data):
            self.assertEqual(topic, topic3)
            o: SOARM101Info = data
            self.assertEqual(len(o.joints_state_positions), 6)  # SO-ARM101 has 6 joints
            self.assertEqual(len(o.joints_state_velocities), 6)  # SO-ARM101 has 6 joints
            self.test_3_pass = True

        self.domain_id = 101  # Use 101 for testing
        hz = 1 / 60

        # Create mock publishers for testing
        self.mock_room_cam_pub = MockCameraPublisher(self.domain_id, topic1, "room")
        self.mock_wrist_cam_pub = MockCameraPublisher(self.domain_id, topic2, "wrist")
        self.mock_joint_pub = MockJointStatePublisher(self.domain_id, topic3)

        self.test_1_pass = self.test_2_pass = self.test_3_pass = False
        self.reader1 = SubscriberWithCallback(dds_callback1, self.domain_id, topic1, CameraInfo, hz)
        self.reader2 = SubscriberWithCallback(dds_callback2, self.domain_id, topic2, CameraInfo, hz)
        self.reader3 = SubscriberWithCallback(dds_callback3, self.domain_id, topic3, SOARM101Info, hz)

        self.reader1.start()
        self.reader2.start()
        self.reader3.start()

        time.sleep(2.0)

        # Start background thread to continuously publish mock data
        self.publishing = True
        self.publish_thread = threading.Thread(target=self._publish_mock_data, daemon=True)
        self.publish_thread.start()

    def _publish_mock_data(self):
        """Background thread function to continuously publish mock data."""
        while self.publishing:
            self.mock_room_cam_pub.write()
            self.mock_wrist_cam_pub.write()
            self.mock_joint_pub.write()
            time.sleep(1.0)  # Publish every second

    def tearDown(self):
        """Clean up after each test method"""
        # Stop background publishing
        if hasattr(self, "publishing"):
            self.publishing = False
        if hasattr(self, "publish_thread"):
            self.publish_thread.join(timeout=1.0)

        if hasattr(self, "reader1"):
            self.reader1.stop()
            del self.reader1
        if hasattr(self, "reader2"):
            self.reader2.stop()
            del self.reader2
        if hasattr(self, "reader3"):
            self.reader3.stop()
            del self.reader3
        if hasattr(self, "mock_room_cam_pub"):
            del self.mock_room_cam_pub
        if hasattr(self, "mock_wrist_cam_pub"):
            del self.mock_wrist_cam_pub
        if hasattr(self, "mock_joint_pub"):
            del self.mock_joint_pub
        time.sleep(0.1)

    def test_init(self):
        """Test Subscriber initialization"""
        self.assertEqual(self.reader1.domain_id, self.domain_id)
        self.assertEqual(self.reader2.domain_id, self.domain_id)
        self.assertEqual(self.reader3.domain_id, self.domain_id)

    def test_read(self):
        """Test read method with actual DDS communication"""
        for i in range(60):
            if self.test_1_pass and self.test_2_pass and self.test_3_pass:
                break
            time.sleep(1.0)

        self.assertTrue(self.test_1_pass, "Room camera data not received")
        self.assertTrue(self.test_2_pass, "Wrist camera data not received")
        self.assertTrue(self.test_3_pass, "Joint state data not received")

    def test_control_publishing(self):
        """Test that control commands can be published for sim_with_dds.py to receive"""
        # Create a mock control publisher
        control_topic = "topic_soarm_ctrl"
        mock_control_pub = MockControlPublisher(self.domain_id, control_topic)

        control_received = False

        def control_callback(topic, data):
            nonlocal control_received
            self.assertEqual(topic, control_topic)
            o: SOARM101CtrlInput = data
            self.assertEqual(len(o.joint_positions), 6)
            self.assertEqual(len(o.joint_velocities), 6)
            self.assertEqual(len(o.joint_efforts), 6)
            control_received = True

        control_reader = SubscriberWithCallback(
            control_callback, self.domain_id, control_topic, SOARM101CtrlInput, 1 / 60
        )
        control_reader.start()

        # Publish control commands
        for i in range(5):
            mock_control_pub.write()
            time.sleep(0.2)

        # Wait for data to be received
        for i in range(10):
            if control_received:
                break
            time.sleep(0.5)

        self.assertTrue(control_received, "Control command data not received")

        control_reader.stop()
        del mock_control_pub


if __name__ == "__main__":
    unittest.main()
