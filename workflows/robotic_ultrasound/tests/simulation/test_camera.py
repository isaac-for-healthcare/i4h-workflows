import unittest
import os
import numpy as np
from unittest import skipUnless
from parameterized import parameterized
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from robotic_ultrasound.scripts.simulation.annotators.camera import (
    CameraPublisher,
    CameraSubscriber
)
from robotic_ultrasound.scripts.rti_dds.schemas.camera_ctrl import CameraCtrlInput
from robotic_ultrasound.scripts.rti_dds.schemas.camera_info import CameraInfo

import os
os.environ['RTI_LICENSE_FILE'] = "/home/yunliu/Workspace/Code/i4h-workflows/workflows/robotic_ultrasound/scripts/rti_dds/rti_license.dat"

try:
    import rti.connextdds as dds
    import rti.idl as idl
    license_path = os.getenv('RTI_LICENSE_FILE')
    RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False
    
@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestCameraBase(unittest.TestCase):
    """Base test class for camera tests with USD stage setup."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the USD stage with basic.usda file."""
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = os.path.join(cls.test_dir, "basic.usda")
        cls.camera_prim_path = "/RoomCamera"
        assert os.path.exists(cls.usda_path), f"basic.usda not found at {cls.usda_path}"

    def setUp(self):
        """Set up test fixtures before each test method."""
        
        omni.usd.get_context().open_stage(self.usda_path)
        simulation_app.update()
        self.stage = omni.usd.get_context().get_stage()

    def tearDown(self):
        """Clean up after each test method."""
        omni.usd.get_context().close_stage()
        simulation_app.update()


class TestCameraPublisher(TestCameraBase):
    """Test cases for CameraPublisher class."""

    def test_init_rgb_camera(self):
        """Test initialization of RGB camera publisher with actual USD stage."""
        publisher = CameraPublisher(
            annotator="rgb",
            prim_path=self.camera_prim_path,
            height=480,
            width=640,
            topic="camera_rgb",
            period=1/30.0,
            domain_id=60
        )
        
        self.assertEqual(publisher.annotator, "rgb")
        self.assertEqual(publisher.prim_path, self.camera_prim_path)
        self.assertEqual(publisher.resolution, (640, 480))
        self.assertIsNotNone(publisher.rp)
        
        camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)
        self.assertTrue(camera_prim.IsValid())
        

    @parameterized.expand([
        ("rgb", "camera_rgb", np.uint8),
        ("distance_to_camera", "camera_depth", np.float32)
    ])
    def test_camera_data(self, annotator, topic, dtype):
        """Test camera data production for both RGB and depth cameras."""
        simulation_app.update()
        
        publisher = CameraPublisher(
            annotator=annotator,
            prim_path=self.camera_prim_path,
            height=480,
            width=640,
            topic=topic,
            period=1/30.0,
            domain_id=60
        )
        
        camera_info = publisher.produce(0.033, 1.0)
        self.assertIsInstance(camera_info, CameraInfo)
        self.assertIsNotNone(camera_info.data)
        
        # Verify focal length from actual camera
        camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)
        expected_focal_length = camera_prim.GetAttribute("focalLength").Get()
        self.assertEqual(camera_info.focal_len, expected_focal_length)

        # Verify data can be converted to numpy array
        data = np.frombuffer(camera_info.data, dtype=dtype)
        self.assertTrue(np.all(data >= 0))


class TestCameraSubscriber(TestCameraBase):
    """Test cases for CameraSubscriber class."""

    def test_consume_focal_length(self):
        """Test consuming focal length control input with actual USD stage."""
        subscriber = CameraSubscriber(
            prim_path=self.camera_prim_path,
            topic="camera_ctrl",
            period=1/30.0,
            domain_id=60
        )
        
        # Get initial focal length
        camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)
        initial_focal_length = camera_prim.GetAttribute("focalLength").Get()
        
        # Change focal length
        new_focal_length = initial_focal_length + 15.0
        ctrl_input = CameraCtrlInput()
        ctrl_input.focal_len = new_focal_length
        subscriber.consume(ctrl_input)
        
        # Step simulation to ensure changes are applied
        simulation_app.update()
        
        # Verify focal length was updated
        updated_focal_length = camera_prim.GetAttribute("focalLength").Get()
        self.assertEqual(updated_focal_length, new_focal_length)


if __name__ == '__main__':
    unittest.main()
