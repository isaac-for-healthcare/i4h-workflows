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

import rti.connextdds as dds
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.soarm_ctrl import SOARM101CtrlInput
from dds.schemas.soarm_info import SOARM101Info
from helpers import requires_rti


@requires_rti
class TestPublisher(unittest.TestCase):
    class TestSOARM101InfoPublisher(Publisher):
        """Concrete implementation of Publisher for SOARM101 info testing"""

        def produce(self, dt: float, sim_time: float) -> SOARM101Info:
            return SOARM101Info(
                joints_state_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                joints_state_velocities=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            )

    class TestSOARM101CtrlPublisher(Publisher):
        """Concrete implementation of Publisher for SOARM101 control testing"""

        def produce(self, dt: float, sim_time: float) -> SOARM101CtrlInput:
            return SOARM101CtrlInput(
                joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                joint_velocities=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                joint_efforts=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )

    class TestWristCameraPublisher(Publisher):
        """Concrete implementation of Publisher for wrist camera info testing"""

        def produce(self, dt: float, sim_time: float) -> CameraInfo:
            return CameraInfo(
                focal_len=12.0,
                stream_id=1,  # Wrist camera stream ID
                frame_num=100,
                width=640,
                height=480,
                data=[255] * 100,  # Mock wrist camera image data
            )

    class TestRoomCameraPublisher(Publisher):
        """Concrete implementation of Publisher for room camera info testing"""

        def produce(self, dt: float, sim_time: float) -> CameraInfo:
            return CameraInfo(
                focal_len=12.0,
                stream_id=2,  # Room camera stream ID
                frame_num=101,
                width=640,
                height=480,
                data=[128] * 100,  # Mock room camera image data (different values)
            )

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.domain_id = 100  # Use a unique domain ID for testing
        self.soarm_info_publisher = self.TestSOARM101InfoPublisher(
            topic="SOARM101Info", cls=SOARM101Info, period=0.1, domain_id=self.domain_id
        )
        self.soarm_ctrl_publisher = self.TestSOARM101CtrlPublisher(
            topic="SOARM101Ctrl", cls=SOARM101CtrlInput, period=0.1, domain_id=self.domain_id
        )
        self.wrist_camera_publisher = self.TestWristCameraPublisher(
            topic="WristCameraInfo", cls=CameraInfo, period=0.1, domain_id=self.domain_id
        )
        self.room_camera_publisher = self.TestRoomCameraPublisher(
            topic="RoomCameraInfo", cls=CameraInfo, period=0.1, domain_id=self.domain_id
        )

        # Create subscribers to verify published messages
        self.participant = dds.DomainParticipant(self.domain_id)
        self.soarm_info_topic = dds.Topic(self.participant, "SOARM101Info", SOARM101Info)
        self.soarm_ctrl_topic = dds.Topic(self.participant, "SOARM101Ctrl", SOARM101CtrlInput)
        self.wrist_camera_topic = dds.Topic(self.participant, "WristCameraInfo", CameraInfo)
        self.room_camera_topic = dds.Topic(self.participant, "RoomCameraInfo", CameraInfo)
        self.soarm_info_reader = dds.DataReader(self.participant.implicit_subscriber, self.soarm_info_topic)
        self.soarm_ctrl_reader = dds.DataReader(self.participant.implicit_subscriber, self.soarm_ctrl_topic)
        self.wrist_camera_reader = dds.DataReader(self.participant.implicit_subscriber, self.wrist_camera_topic)
        self.room_camera_reader = dds.DataReader(self.participant.implicit_subscriber, self.room_camera_topic)
        time.sleep(1.0)

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "soarm_info_publisher"):
            del self.soarm_info_publisher
        if hasattr(self, "soarm_ctrl_publisher"):
            del self.soarm_ctrl_publisher
        if hasattr(self, "wrist_camera_publisher"):
            del self.wrist_camera_publisher
        if hasattr(self, "room_camera_publisher"):
            del self.room_camera_publisher
        if hasattr(self, "soarm_info_reader"):
            del self.soarm_info_reader
        if hasattr(self, "soarm_ctrl_reader"):
            del self.soarm_ctrl_reader
        if hasattr(self, "wrist_camera_reader"):
            del self.wrist_camera_reader
        if hasattr(self, "room_camera_reader"):
            del self.room_camera_reader
        if hasattr(self, "soarm_info_topic"):
            del self.soarm_info_topic
        if hasattr(self, "soarm_ctrl_topic"):
            del self.soarm_ctrl_topic
        if hasattr(self, "wrist_camera_topic"):
            del self.wrist_camera_topic
        if hasattr(self, "room_camera_topic"):
            del self.room_camera_topic
        if hasattr(self, "participant"):
            del self.participant
        time.sleep(0.1)  # Allow time for cleanup

    def test_soarm_info_publisher_init(self):
        """Test SOARM101 info publisher initialization"""
        self.assertEqual(self.soarm_info_publisher.topic, "SOARM101Info")
        self.assertEqual(self.soarm_info_publisher.cls, SOARM101Info)
        self.assertEqual(self.soarm_info_publisher.period, 0.1)
        self.assertEqual(self.soarm_info_publisher.domain_id, self.domain_id)
        self.assertIsNotNone(self.soarm_info_publisher.logger)
        self.assertIsNotNone(self.soarm_info_publisher.dds_writer)

    def test_soarm_ctrl_publisher_init(self):
        """Test SOARM101 control publisher initialization"""
        self.assertEqual(self.soarm_ctrl_publisher.topic, "SOARM101Ctrl")
        self.assertEqual(self.soarm_ctrl_publisher.cls, SOARM101CtrlInput)
        self.assertEqual(self.soarm_ctrl_publisher.period, 0.1)
        self.assertEqual(self.soarm_ctrl_publisher.domain_id, self.domain_id)
        self.assertIsNotNone(self.soarm_ctrl_publisher.logger)
        self.assertIsNotNone(self.soarm_ctrl_publisher.dds_writer)

    def test_wrist_camera_publisher_init(self):
        """Test wrist camera info publisher initialization"""
        self.assertEqual(self.wrist_camera_publisher.topic, "WristCameraInfo")
        self.assertEqual(self.wrist_camera_publisher.cls, CameraInfo)
        self.assertEqual(self.wrist_camera_publisher.period, 0.1)
        self.assertEqual(self.wrist_camera_publisher.domain_id, self.domain_id)
        self.assertIsNotNone(self.wrist_camera_publisher.logger)
        self.assertIsNotNone(self.wrist_camera_publisher.dds_writer)

    def test_room_camera_publisher_init(self):
        """Test room camera info publisher initialization"""
        self.assertEqual(self.room_camera_publisher.topic, "RoomCameraInfo")
        self.assertEqual(self.room_camera_publisher.cls, CameraInfo)
        self.assertEqual(self.room_camera_publisher.period, 0.1)
        self.assertEqual(self.room_camera_publisher.domain_id, self.domain_id)
        self.assertIsNotNone(self.room_camera_publisher.logger)
        self.assertIsNotNone(self.room_camera_publisher.dds_writer)

    def test_soarm_info_write(self):
        """Test SOARM101 info write method with actual DDS communication"""
        # Write data
        exec_time = self.soarm_info_publisher.write(0.1, 1.0)
        self.assertGreaterEqual(exec_time, 0)

        max_retries = 5
        for _ in range(max_retries):
            samples = self.soarm_info_reader.take()
            if samples:
                break
            time.sleep(0.2)

        self.assertEqual(len(samples), 1)
        sample_data = samples[0].data

        # Verify 6-DOF joint data
        self.assertEqual(len(sample_data.joints_state_positions), 6)
        self.assertEqual(len(sample_data.joints_state_velocities), 6)

        # Verify joint position values
        expected_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for i, expected in enumerate(expected_positions):
            self.assertAlmostEqual(sample_data.joints_state_positions[i], expected, places=6)

    def test_soarm_ctrl_write(self):
        """Test SOARM101 control write method with actual DDS communication"""
        # Write data
        exec_time = self.soarm_ctrl_publisher.write(0.1, 2.0)
        self.assertGreaterEqual(exec_time, 0)

        max_retries = 5
        for _ in range(max_retries):
            samples = self.soarm_ctrl_reader.take()
            if samples:
                break
            time.sleep(0.2)

        self.assertEqual(len(samples), 1)
        sample_data = samples[0].data

        # Verify 6-DOF joint data
        self.assertEqual(len(sample_data.joint_positions), 6)
        self.assertEqual(len(sample_data.joint_velocities), 6)
        self.assertEqual(len(sample_data.joint_efforts), 6)

        # Verify joint position values
        expected_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for i, expected in enumerate(expected_positions):
            self.assertAlmostEqual(sample_data.joint_positions[i], expected, places=6)

    def test_wrist_camera_write(self):
        """Test wrist camera info write method with actual DDS communication"""
        # Write data
        exec_time = self.wrist_camera_publisher.write(0.1, 3.0)
        self.assertGreaterEqual(exec_time, 0)

        max_retries = 5
        for _ in range(max_retries):
            samples = self.wrist_camera_reader.take()
            if samples:
                break
            time.sleep(0.2)

        self.assertEqual(len(samples), 1)
        sample_data = samples[0].data

        # Verify wrist camera data
        self.assertEqual(sample_data.focal_len, 12.0)
        self.assertEqual(sample_data.stream_id, 1)  # Wrist camera stream ID
        self.assertEqual(sample_data.frame_num, 100)
        self.assertEqual(sample_data.width, 640)
        self.assertEqual(sample_data.height, 480)
        self.assertEqual(len(sample_data.data), 100)  # Mock image data length
        self.assertEqual(sample_data.data[0], 255)  # Wrist camera specific data

    def test_room_camera_write(self):
        """Test room camera info write method with actual DDS communication"""
        # Write data
        exec_time = self.room_camera_publisher.write(0.1, 4.0)
        self.assertGreaterEqual(exec_time, 0)

        max_retries = 5
        for _ in range(max_retries):
            samples = self.room_camera_reader.take()
            if samples:
                break
            time.sleep(0.2)

        self.assertEqual(len(samples), 1)
        sample_data = samples[0].data

        # Verify room camera data
        self.assertEqual(sample_data.focal_len, 12.0)
        self.assertEqual(sample_data.stream_id, 2)  # Room camera stream ID
        self.assertEqual(sample_data.frame_num, 101)
        self.assertEqual(sample_data.width, 640)
        self.assertEqual(sample_data.height, 480)
        self.assertEqual(len(sample_data.data), 100)  # Mock image data length
        self.assertEqual(sample_data.data[0], 128)  # Room camera specific data

    def test_produce_abstract(self):
        """Test that produce() is abstract and must be implemented"""
        with self.assertRaises(TypeError):
            Publisher("test", str, 0.1, 0)


if __name__ == "__main__":
    unittest.main()
