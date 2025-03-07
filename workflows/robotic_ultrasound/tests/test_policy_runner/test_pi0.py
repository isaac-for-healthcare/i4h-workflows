import os
import time
import unittest
from unittest import skipUnless

import numpy as np
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.subscriber import SubscriberWithCallback
from PIL import Image

try:
    import rti.connextdds as dds  # noqa: F401

    license_path = os.getenv("RTI_LICENSE_FILE")
    RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False

"""
Must execute the pi0 policy runner in another process before execute this test.
Ensure to set the `height=480` and `width=640` in the`run_policy.py`.

"""

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))


class TestRoomCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__("topic_room_camera_data_rgb", CameraInfo, 1 / 30, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 20
        output.height = 480
        output.width = 640
        output.data = Image.open(os.path.join(test_dir, "room_cam.png")).tobytes()
        return output


class TestWristCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__("topic_wrist_camera_data_rgb", CameraInfo, 1 / 30, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 20
        output.height = 480
        output.width = 640
        output.data = Image.open(os.path.join(test_dir, "wrist_cam.png")).tobytes()
        return output


class TestPosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__("topic_franka_info", FrankaInfo, 1 / 30, domain_id)

    def produce(self, dt: float, sim_time: float):
        data = np.loadtxt(os.path.join(test_dir, "pos.txt"), delimiter=",")
        output = FrankaInfo()
        output.joints_state_positions = data.tolist()
        return output


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestRunPI0Policy(unittest.TestCase):
    def setUp(self):
        def cb(topic, data):
            self.assertEqual(topic, "topic_franka_ctrl")
            self.assertIsInstance(data, FrankaCtrlInput)
            o: FrankaCtrlInput = data
            action_chunk = np.array(o.joint_positions, dtype=np.float32)
            self.assertEqual(action_chunk.shape, (300,))
            self.test_pass = True

        """Set up test fixtures before each test method"""
        self.domain_id = 101  # Use a unique domain ID for testing
        self.r_cam_writer = TestRoomCamPublisher(self.domain_id)
        self.w_cam_writer = TestWristCamPublisher(self.domain_id)
        self.pos_writer = TestPosPublisher(self.domain_id)
        self.reader = SubscriberWithCallback(cb, self.domain_id, "topic_franka_ctrl", FrankaCtrlInput, 1 / 30)
        self.test_pass = False
        self.reader.start()
        time.sleep(1.0)

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "r_cam_writer"):
            del self.r_cam_writer
        if hasattr(self, "w_cam_writer"):
            del self.w_cam_writer
        if hasattr(self, "pos_writer"):
            del self.pos_writer
        if hasattr(self, "reader"):
            self.reader.stop()
            del self.reader
        time.sleep(0.1)  # Allow time for cleanup

    def test_init(self):
        """Test publisher initialization"""
        self.assertEqual(self.r_cam_writer.domain_id, self.domain_id)
        self.assertEqual(self.reader.domain_id, self.domain_id)

    def test_write(self):
        """Test write method with actual DDS communication"""
        # Write data
        self.r_cam_writer.write(0.1, 1.0)
        self.w_cam_writer.write(0.1, 1.0)
        self.pos_writer.write(0.1, 1.0)
        for _ in range(10):
            if self.test_pass:
                break
            time.sleep(1.0)
        self.assertTrue(self.test_pass)


if __name__ == "__main__":
    unittest.main()
