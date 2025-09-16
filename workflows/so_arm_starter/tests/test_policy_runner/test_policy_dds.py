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
DDS-based policy runner test for SO-ARM Starter workflow.
Must execute the policy runner in another process before executing this test.
"""

# Expected chunk size for GR00T policy runner (96 for GR00T)
expected_chunk_size = 96


class TestRoomCamPublisher(Publisher):
    """Publisher for room camera data with inline dummy data generation."""

    def __init__(self, domain_id: int):
        super().__init__("topic_room_camera_data_rgb", CameraInfo, 1 / 30, domain_id)

    def produce(self, dt: float, sim_time: float):
        """Generate dummy room camera data."""
        output = CameraInfo()
        output.focal_len = 12.0  # focal length
        output.height = 480
        output.width = 640

        # Generate dummy RGB image data (480x640x3)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output.data = dummy_image.tobytes()

        return output


class TestWristCamPublisher(Publisher):
    """Publisher for wrist camera data with inline dummy data generation."""

    def __init__(self, domain_id: int):
        super().__init__("topic_wrist_camera_data_rgb", CameraInfo, 1 / 30, domain_id)

    def produce(self, dt: float, sim_time: float):
        """Generate dummy wrist camera data."""
        output = CameraInfo()
        output.focal_len = 12.0  # focal length
        output.height = 480
        output.width = 640

        # Generate dummy RGB image data (480x640x3)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output.data = dummy_image.tobytes()

        return output


class TestPosPublisher(Publisher):
    """Publisher for robot joint position data with inline dummy data generation."""

    def __init__(self, domain_id: int):
        super().__init__("topic_soarm_info", SOARM101Info, 1 / 30, domain_id)

    def produce(self, dt: float, sim_time: float):
        """Generate dummy joint position data."""
        dummy_positions = np.random.uniform(-np.pi / 2, np.pi / 2, 6)
        dummy_velocities = np.random.uniform(-1.0, 1.0, 6)

        output = SOARM101Info()
        output.joints_state_positions = dummy_positions.tolist()
        output.joints_state_velocities = dummy_velocities.tolist()
        return output


class InlineMockPolicyRunner:
    """Inline mock policy runner that runs in the same process."""

    def __init__(self, domain_id: int):
        self.domain_id = domain_id
        self.chunk_length = 16  # 16 * 6 = 96 actions
        self.is_running = False

        self.current_state = {
            "room_cam": None,
            "wrist_cam": None,
            "joint_pos": None,
        }

        self.writer = None
        self.room_cam_subscriber = None
        self.wrist_cam_subscriber = None
        self.pos_subscriber = None

    def _setup_dds(self):
        """Set up DDS publishers and subscribers."""
        hz = 60

        class MockPolicyPublisher(Publisher):
            def __init__(self, domain_id: int, chunk_length: int):
                super().__init__("topic_soarm_ctrl", SOARM101CtrlInput, 1 / hz, domain_id)
                self.chunk_length = chunk_length

            def produce(self, dt: float, sim_time: float):
                # Generate mock actions (chunk_length * 6 joints)
                mock_actions = np.random.uniform(-0.1, 0.1, (self.chunk_length, 6))

                output = SOARM101CtrlInput()
                # Flatten to (chunk_length * 6,) as expected by schema
                output.joint_positions = mock_actions.flatten().astype(np.float32).tolist()

                print(f"[MOCK] Generated action chunk of size {len(output.joint_positions)}")
                return output

        self.writer = MockPolicyPublisher(self.domain_id, self.chunk_length)

        def dds_callback(topic, data):
            if topic == "topic_room_camera_data_rgb":
                o: CameraInfo = data
                self.current_state["room_cam"] = o.data
                print(f"[MOCK] Room camera: {len(o.data)} bytes, focal_len={o.focal_len}")

            elif topic == "topic_wrist_camera_data_rgb":
                o: CameraInfo = data
                self.current_state["wrist_cam"] = o.data
                print(f"[MOCK] Wrist camera: {len(o.data)} bytes, focal_len={o.focal_len}")

            elif topic == "topic_soarm_info":
                o: SOARM101Info = data
                self.current_state["joint_pos"] = o.joints_state_positions
                print(f"[MOCK] Joint positions: {len(o.joints_state_positions)} joints")

            # Generate action when all data is available
            if (
                self.current_state["room_cam"] is not None
                and self.current_state["wrist_cam"] is not None
                and self.current_state["joint_pos"] is not None
            ):
                self.writer.write(0.1, 1.0)
                print("[MOCK] Published actions to topic_soarm_ctrl")

                # Reset state for next iteration
                self.current_state["room_cam"] = None
                self.current_state["wrist_cam"] = None
                self.current_state["joint_pos"] = None

        # Start DDS subscribers
        hz = 60
        self.room_cam_subscriber = SubscriberWithCallback(
            dds_callback, self.domain_id, "topic_room_camera_data_rgb", CameraInfo, 1 / hz
        )
        self.wrist_cam_subscriber = SubscriberWithCallback(
            dds_callback, self.domain_id, "topic_wrist_camera_data_rgb", CameraInfo, 1 / hz
        )
        self.pos_subscriber = SubscriberWithCallback(
            dds_callback, self.domain_id, "topic_soarm_info", SOARM101Info, 1 / hz
        )

        self.room_cam_subscriber.start()
        self.wrist_cam_subscriber.start()
        self.pos_subscriber.start()

    def run(self):
        """Main run loop for the mock policy runner."""
        print(f"[MOCK] Starting inline mock policy runner on domain {self.domain_id}")
        print(f"[MOCK] Expected action chunk size: {self.chunk_length * 6}")

        self.is_running = True
        self._setup_dds()

        try:
            while self.is_running:
                time.sleep(0.1)  # Small sleep to prevent busy waiting
        except Exception as e:
            print(f"[MOCK] Error in mock policy runner: {e}")
        finally:
            print("[MOCK] Mock policy runner stopped")

    def stop(self):
        """Stop the mock policy runner."""
        print("[MOCK] Stopping mock policy runner...")
        self.is_running = False

        # Clean up DDS subscribers gracefully
        if self.room_cam_subscriber:
            try:
                self.room_cam_subscriber.stop()
            except Exception as e:
                print(f"[MOCK] Warning: Error stopping room_cam_subscriber: {e}")

        if self.wrist_cam_subscriber:
            try:
                self.wrist_cam_subscriber.stop()
            except Exception as e:
                print(f"[MOCK] Warning: Error stopping wrist_cam_subscriber: {e}")

        if self.pos_subscriber:
            try:
                self.pos_subscriber.stop()
            except Exception as e:
                print(f"[MOCK] Warning: Error stopping pos_subscriber: {e}")

        # Clean up publisher
        if self.writer:
            try:
                del self.writer
            except Exception as e:
                print(f"[MOCK] Warning: Error cleaning writer: {e}")

        print("[MOCK] Mock policy runner cleanup completed")


@requires_rti
class TestRunSOARMStarterPolicy(unittest.TestCase):
    """Test DDS-based policy runner for SO-ARM Starter workflow."""

    def setUp(self):
        """Set up test fixtures before each test method."""

        def cb(topic, data):
            """Callback function to verify policy output."""
            self.assertEqual(topic, "topic_soarm_ctrl")
            self.assertIsInstance(data, SOARM101CtrlInput)

            o: SOARM101CtrlInput = data
            action_chunk = np.array(o.joint_positions, dtype=np.float32)

            # Verify inference result length (chunk_length * 6 = 16 * 6 = 96)
            self.assertEqual(action_chunk.shape, (expected_chunk_size,))
            self.assertGreater(len(o.joint_positions), 0)
            self.assertEqual(len(o.joint_positions), 96)  # Explicit length check

            # Verify action values are reasonable
            self.assertTrue(np.all(np.abs(action_chunk) <= 1.0))  # Actions should be bounded

            print(f"[TEST] Received valid action chunk of size {len(o.joint_positions)}")
            self.test_pass = True

        # Use a unique domain ID for testing
        self.domain_id = 102

        # Initialize DDS publishers with dummy data
        self.room_cam_writer = TestRoomCamPublisher(self.domain_id)
        self.wrist_cam_writer = TestWristCamPublisher(self.domain_id)
        self.pos_writer = TestPosPublisher(self.domain_id)

        # Initialize DDS subscriber for policy output
        self.reader = SubscriberWithCallback(cb, self.domain_id, "topic_soarm_ctrl", SOARM101CtrlInput, 1 / 30)

        # Start inline mock policy runner
        self.mock_policy_runner = InlineMockPolicyRunner(self.domain_id)
        self.policy_thread = threading.Thread(target=self.mock_policy_runner.run, daemon=True)
        self.policy_thread.start()

        self.test_pass = False
        self.reader.start()
        time.sleep(1.0)  # Allow time for DDS setup

    def tearDown(self):
        """Clean up after each test method."""
        # Stop inline mock policy runner first
        if hasattr(self, "mock_policy_runner"):
            self.mock_policy_runner.stop()

        # Wait for policy thread to finish
        if hasattr(self, "policy_thread") and self.policy_thread.is_alive():
            self.policy_thread.join(timeout=2.0)

        # Stop DDS subscriber
        if hasattr(self, "reader"):
            try:
                self.reader.stop()
            except Exception as e:
                print(f"[TEST] Warning: Error stopping reader: {e}")
            del self.reader

        # Clean up DDS publishers
        if hasattr(self, "room_cam_writer"):
            try:
                del self.room_cam_writer
            except Exception as e:
                print(f"[TEST] Warning: Error cleaning room_cam_writer: {e}")

        if hasattr(self, "wrist_cam_writer"):
            try:
                del self.wrist_cam_writer
            except Exception as e:
                print(f"[TEST] Warning: Error cleaning wrist_cam_writer: {e}")

        if hasattr(self, "pos_writer"):
            try:
                del self.pos_writer
            except Exception as e:
                print(f"[TEST] Warning: Error cleaning pos_writer: {e}")

        # Give extra time for DDS cleanup
        time.sleep(0.5)

    def test_publisher_initialization(self):
        """Test DDS publisher initialization."""
        self.assertEqual(self.room_cam_writer.domain_id, self.domain_id)
        self.assertEqual(self.wrist_cam_writer.domain_id, self.domain_id)
        self.assertEqual(self.pos_writer.domain_id, self.domain_id)
        self.assertEqual(self.reader.domain_id, self.domain_id)

    def test_policy_dds_communication(self):
        """Test DDS communication with policy runner and inference result length validation."""
        print("[TEST] Starting DDS communication test with inline mock policy runner")

        # Verify mock policy runner is running
        self.assertTrue(self.mock_policy_runner.is_running)

        # Publish sensor data multiple times to trigger policy inference
        print("[TEST] Publishing sensor data to trigger policy inference...")
        for i in range(3):
            print(f"[TEST] Publishing data batch {i+1}/3")

            # Publish all sensor data simultaneously
            self.room_cam_writer.write(0.1, 1.0 + i * 0.1)
            self.wrist_cam_writer.write(0.1, 1.0 + i * 0.1)
            self.pos_writer.write(0.1, 1.0 + i * 0.1)

            time.sleep(0.2)

        # Wait for policy runner to process and respond with action chunks
        print("[TEST] Waiting for policy runner inference results...")
        max_wait_time = 10  # seconds
        for i in range(max_wait_time):
            if self.test_pass:
                print(f"[TEST] ✅ Received valid inference results after {i+1} seconds")
                break
            time.sleep(1.0)
            print(f"[TEST] Waiting... ({i+1}/{max_wait_time})")

        # Verify policy runner responded with valid action chunk
        self.assertTrue(
            self.test_pass,
            f"Policy runner did not produce expected action chunk of size {expected_chunk_size} "
            f"within {max_wait_time} seconds",
        )

        print("[TEST] ✅ DDS communication and inference result validation completed successfully")

    def test_data_consistency(self):
        """Test consistency of generated dummy data."""
        # Generate multiple data samples and verify consistency
        room_samples = [self.room_cam_writer.produce(0.1, 1.0) for _ in range(3)]
        wrist_samples = [self.wrist_cam_writer.produce(0.1, 1.0) for _ in range(3)]
        pos_samples = [self.pos_writer.produce(0.1, 1.0) for _ in range(3)]

        # Verify all samples have consistent structure
        for sample in room_samples:
            self.assertEqual(sample.focal_len, 12.0)
            self.assertEqual(sample.height, 480)
            self.assertEqual(sample.width, 640)

        for sample in wrist_samples:
            self.assertEqual(sample.focal_len, 12.0)
            self.assertEqual(sample.height, 480)
            self.assertEqual(sample.width, 640)

        for sample in pos_samples:
            self.assertEqual(len(sample.joints_state_positions), 6)


if __name__ == "__main__":
    unittest.main()
