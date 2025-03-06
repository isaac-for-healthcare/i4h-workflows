import os
import time
import unittest
from unittest import skipUnless

import numpy as np
from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.schemas.franka_info import FrankaInfo
from rti_dds.subscriber import SubscriberWithCallback

try:
    import rti.connextdds as dds  # noqa: F401

    license_path = os.getenv("RTI_LICENSE_FILE")
    RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False

"""
Must immediately execute the `sim_with_dds.py` in another process after executing this test.

"""


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
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
