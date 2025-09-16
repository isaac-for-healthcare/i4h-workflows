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

import queue
import time
import unittest

import rti.connextdds as dds
from dds.schemas.camera_info import CameraInfo
from dds.schemas.soarm_ctrl import SOARM101CtrlInput
from dds.schemas.soarm_info import SOARM101Info
from dds.subscriber import Subscriber, SubscriberWithCallback, SubscriberWithQueue
from helpers import requires_rti


class _TestSOARM101InfoSubscriber(Subscriber):
    """Concrete implementation of Subscriber for SOARM101 info testing."""

    def consume(self, data) -> None:
        return data


@requires_rti
class TestDDSSubscriber(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.domain_id = 0
        self.soarm_info_topic_name = "soarm_info_topic"
        self.soarm_ctrl_topic_name = "soarm_ctrl_topic"
        self.wrist_camera_topic_name = "wrist_camera_info_topic"
        self.room_camera_topic_name = "room_camera_info_topic"

        self.participant = dds.DomainParticipant(domain_id=self.domain_id)
        self.soarm_info_topic = dds.Topic(self.participant, self.soarm_info_topic_name, SOARM101Info)
        self.soarm_ctrl_topic = dds.Topic(self.participant, self.soarm_ctrl_topic_name, SOARM101CtrlInput)
        self.wrist_camera_topic = dds.Topic(self.participant, self.wrist_camera_topic_name, CameraInfo)
        self.room_camera_topic = dds.Topic(self.participant, self.room_camera_topic_name, CameraInfo)
        self.soarm_info_writer = dds.DataWriter(self.participant.implicit_publisher, self.soarm_info_topic)
        self.soarm_ctrl_writer = dds.DataWriter(self.participant.implicit_publisher, self.soarm_ctrl_topic)
        self.wrist_camera_writer = dds.DataWriter(self.participant.implicit_publisher, self.wrist_camera_topic)
        self.room_camera_writer = dds.DataWriter(self.participant.implicit_publisher, self.room_camera_topic)

        self.soarm_info_subscriber = _TestSOARM101InfoSubscriber(
            topic=self.soarm_info_topic_name, cls=SOARM101Info, period=0.1, domain_id=self.domain_id
        )

        time.sleep(1.0)

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "soarm_info_subscriber"):
            self.soarm_info_subscriber.stop()
        if hasattr(self, "soarm_info_writer"):
            self.soarm_info_writer.close()
        if hasattr(self, "soarm_ctrl_writer"):
            self.soarm_ctrl_writer.close()
        if hasattr(self, "wrist_camera_writer"):
            self.wrist_camera_writer.close()
        if hasattr(self, "room_camera_writer"):
            self.room_camera_writer.close()
        if hasattr(self, "soarm_info_topic"):
            self.soarm_info_topic.close()
        if hasattr(self, "soarm_ctrl_topic"):
            self.soarm_ctrl_topic.close()
        if hasattr(self, "wrist_camera_topic"):
            self.wrist_camera_topic.close()
        if hasattr(self, "room_camera_topic"):
            self.room_camera_topic.close()
        if hasattr(self, "participant"):
            self.participant.close()
        time.sleep(0.1)

    def test_soarm_info_subscriber_init(self):
        """Test SOARM101 info subscriber initialization"""
        self.assertEqual(self.soarm_info_subscriber.topic, self.soarm_info_topic_name)
        self.assertEqual(self.soarm_info_subscriber.cls, SOARM101Info)
        self.assertEqual(self.soarm_info_subscriber.period, 0.1)
        self.assertEqual(self.soarm_info_subscriber.domain_id, self.domain_id)
        self.assertTrue(self.soarm_info_subscriber.add_to_queue)
        self.assertIsInstance(self.soarm_info_subscriber.data_q, queue.Queue)

    def test_soarm_info_start_stop(self):
        """Test start and stop functionality with real DDS"""
        self.soarm_info_subscriber.start()
        time.sleep(0.5)

        # Create test SOARM101 info data with 6-DOF
        test_soarm_info = SOARM101Info(
            joints_state_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            joints_state_velocities=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        )

        self.soarm_info_writer.write(test_soarm_info)

        for data in self.soarm_info_subscriber.dds_reader.take_data():
            self.assertEqual(len(data.joints_state_positions), 6)
            self.assertEqual(len(data.joints_state_velocities), 6)
            break

        self.soarm_info_subscriber.stop()
        self.assertIsNone(self.soarm_info_subscriber.stop_event)


@requires_rti
class TestSubscriberWithQueue(unittest.TestCase):
    def setUp(self):
        self.domain_id = 100
        self.soarm_ctrl_topic_name = "soarm_ctrl_queue_topic"
        self.period = 0.1

        self.participant = dds.DomainParticipant(domain_id=self.domain_id)
        self.soarm_ctrl_topic_dds = dds.Topic(self.participant, self.soarm_ctrl_topic_name, SOARM101CtrlInput)
        self.soarm_ctrl_writer = dds.DataWriter(self.participant.implicit_publisher, self.soarm_ctrl_topic_dds)

        self.soarm_ctrl_subscriber = SubscriberWithQueue(
            topic=self.soarm_ctrl_topic_name, cls=SOARM101CtrlInput, period=self.period, domain_id=self.domain_id
        )

    def tearDown(self):
        if hasattr(self, "soarm_ctrl_subscriber"):
            self.soarm_ctrl_subscriber.stop()
        if hasattr(self, "soarm_ctrl_writer"):
            self.soarm_ctrl_writer.close()
        if hasattr(self, "soarm_ctrl_topic_dds"):
            self.soarm_ctrl_topic_dds.close()
        if hasattr(self, "participant"):
            self.participant.close()

    def test_soarm_ctrl_read_data(self):
        self.soarm_ctrl_subscriber.start()
        time.sleep(1.0)

        self.assertIsNotNone(self.soarm_ctrl_subscriber.dds_reader, "DDS reader not created")

        # Create test SOARM101 control data with 6-DOF
        test_soarm_ctrl = SOARM101CtrlInput(
            joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            joint_velocities=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            joint_efforts=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

        self.soarm_ctrl_writer.write(test_soarm_ctrl)

        max_retries = 5
        for _ in range(max_retries):
            data = self.soarm_ctrl_subscriber.read_data()
            if data is not None:
                self.assertEqual(len(data.joint_positions), 6)
                self.assertEqual(len(data.joint_velocities), 6)
                self.assertEqual(len(data.joint_efforts), 6)
                break
            time.sleep(0.2)
        else:
            self.fail("No data received after multiple retries")


@requires_rti
class TestSubscriberWithCallback(unittest.TestCase):
    def setUp(self):
        self.domain_id = 200  # Use a unique domain ID
        self.wrist_camera_topic_name = "wrist_camera_info_callback_topic"
        self.room_camera_topic_name = "room_camera_info_callback_topic"
        self.wrist_callback_called = False
        self.room_callback_called = False
        self.wrist_received_data = None
        self.room_received_data = None

        # Define callback functions
        def wrist_test_callback(topic: str, data: CameraInfo) -> None:
            self.wrist_callback_called = True
            self.wrist_received_data = data

        def room_test_callback(topic: str, data: CameraInfo) -> None:
            self.room_callback_called = True
            self.room_received_data = data

        self.participant = dds.DomainParticipant(domain_id=self.domain_id)
        self.wrist_camera_topic = dds.Topic(self.participant, self.wrist_camera_topic_name, CameraInfo)
        self.room_camera_topic = dds.Topic(self.participant, self.room_camera_topic_name, CameraInfo)
        self.wrist_camera_writer = dds.DataWriter(self.participant.implicit_publisher, self.wrist_camera_topic)
        self.room_camera_writer = dds.DataWriter(self.participant.implicit_publisher, self.room_camera_topic)

        self.wrist_camera_subscriber = SubscriberWithCallback(
            cb=wrist_test_callback,
            domain_id=self.domain_id,
            topic=self.wrist_camera_topic_name,
            cls=CameraInfo,
            period=0.1,
        )
        self.room_camera_subscriber = SubscriberWithCallback(
            cb=room_test_callback,
            domain_id=self.domain_id,
            topic=self.room_camera_topic_name,
            cls=CameraInfo,
            period=0.1,
        )

    def tearDown(self):
        if hasattr(self, "wrist_camera_subscriber"):
            self.wrist_camera_subscriber.stop()
        if hasattr(self, "room_camera_subscriber"):
            self.room_camera_subscriber.stop()
        if hasattr(self, "wrist_camera_writer"):
            self.wrist_camera_writer.close()
        if hasattr(self, "room_camera_writer"):
            self.room_camera_writer.close()
        if hasattr(self, "wrist_camera_topic"):
            self.wrist_camera_topic.close()
        if hasattr(self, "room_camera_topic"):
            self.room_camera_topic.close()
        if hasattr(self, "participant"):
            self.participant.close()
        time.sleep(0.1)

    def test_wrist_camera_callback_with_dds(self):
        # Start the wrist camera subscriber
        self.wrist_camera_subscriber.start()
        time.sleep(1.0)  # Allow time for discovery

        # Verify DDS entities are properly created
        self.assertIsNotNone(self.wrist_camera_subscriber.dds_reader, "DDS reader not created")

        # Write test wrist camera info data
        test_wrist_camera_info = CameraInfo(
            focal_len=12.0,
            stream_id=1,  # Wrist camera stream ID
            frame_num=100,
            width=640,
            height=480,
            data=[255] * 100,  # Mock wrist camera image data
        )

        self.wrist_camera_writer.write(test_wrist_camera_info)

        # Wait for callback to be processed
        max_retries = 5
        for _ in range(max_retries):
            if self.wrist_callback_called:
                break
            time.sleep(0.2)
        else:
            self.fail("Wrist camera callback was not called after multiple retries")

        # Verify callback data
        self.assertIsNotNone(self.wrist_received_data, "No wrist camera data received in callback")
        self.assertEqual(self.wrist_received_data.focal_len, 12.0)
        self.assertEqual(self.wrist_received_data.stream_id, 1)
        self.assertEqual(self.wrist_received_data.frame_num, 100)
        self.assertEqual(self.wrist_received_data.width, 640)
        self.assertEqual(self.wrist_received_data.height, 480)
        self.assertEqual(len(self.wrist_received_data.data), 100)
        self.assertEqual(self.wrist_received_data.data[0], 255)  # Wrist camera specific data

    def test_room_camera_callback_with_dds(self):
        # Start the room camera subscriber
        self.room_camera_subscriber.start()
        time.sleep(1.0)  # Allow time for discovery

        # Verify DDS entities are properly created
        self.assertIsNotNone(self.room_camera_subscriber.dds_reader, "DDS reader not created")

        # Write test room camera info data
        test_room_camera_info = CameraInfo(
            focal_len=12.0,
            stream_id=2,  # Room camera stream ID
            frame_num=101,
            width=640,
            height=480,
            data=[128] * 100,  # Mock room camera image data
        )

        self.room_camera_writer.write(test_room_camera_info)

        # Wait for callback to be processed
        max_retries = 5
        for _ in range(max_retries):
            if self.room_callback_called:
                break
            time.sleep(0.2)
        else:
            self.fail("Room camera callback was not called after multiple retries")

        # Verify callback data
        self.assertIsNotNone(self.room_received_data, "No room camera data received in callback")
        self.assertEqual(self.room_received_data.focal_len, 12.0)
        self.assertEqual(self.room_received_data.stream_id, 2)
        self.assertEqual(self.room_received_data.frame_num, 101)
        self.assertEqual(self.room_received_data.width, 640)
        self.assertEqual(self.room_received_data.height, 480)
        self.assertEqual(len(self.room_received_data.data), 100)
        self.assertEqual(self.room_received_data.data[0], 128)  # Room camera specific data


if __name__ == "__main__":
    unittest.main()
