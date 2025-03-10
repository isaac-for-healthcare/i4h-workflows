import os
import unittest
from importlib.util import find_spec
from unittest import skipUnless

import numpy as np
from dds.schemas.camera_ctrl import CameraCtrlInput
from dds.schemas.camera_info import CameraInfo
from holoscan_apps.realsense.enumerate import count_devices
from holoscan_apps.realsense.camera import RealsenseApp

try:
    RTI_AVAILABLE = bool(find_spec("rti.connextdds"))
    if RTI_AVAILABLE:
        license_path = os.getenv("RTI_LICENSE_FILE")
        RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False

DEVICES_AVAILABLE = count_devices() > 0

@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
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
