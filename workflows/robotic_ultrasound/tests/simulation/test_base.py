import unittest
import os
from unittest import skipUnless
from parameterized import parameterized
from unittest.mock import MagicMock
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd
import omni.usd

from robotic_ultrasound.scripts.simulation.annotators.base import BaseAnnotator
from robotic_ultrasound.scripts.rti_dds.publisher import Publisher
from robotic_ultrasound.scripts.rti_dds.subscriber import Subscriber

TEST_CASES = [
        ("none_publishers_subscribers",
         None, None,
         0, 0, None),
        ("mixed_none_publishers", 
         [None, MagicMock(spec=Publisher), None], [MagicMock(spec=Subscriber)],
         1, 1, 0)
]

try:
    import rti.connextdds as dds
    import rti.idl as idl
    license_path = os.getenv('RTI_LICENSE_FILE')
    RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False
    
@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestBaseAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the USD stage with basic.usda file."""
        # Get path to basic.usda
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.basic_usd_path = os.path.join(cls.test_dir, "basic.usda")
        assert os.path.exists(cls.basic_usd_path), f"basic.usda not found at {cls.basic_usd_path}"

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Load the basic USD file
        self.stage = Usd.Stage.Open(self.basic_usd_path)
        self.context = omni.usd.get_context()
        self.context.open_stage(self.basic_usd_path)
        
        # Mock publishers and subscribers
        self.mock_publisher = MagicMock(spec=Publisher)
        self.mock_publisher.topic = "test_pub_topic"
        self.mock_publisher.hz = 30
        self.mock_publisher.write = MagicMock()

        self.mock_subscriber = MagicMock(spec=Subscriber)
        self.mock_subscriber.topic = "test_sub_topic"
        self.mock_subscriber.hz = 10
        self.mock_subscriber.read = MagicMock()
        self.mock_subscriber.start = MagicMock()
        self.mock_subscriber.stop = MagicMock()

        # Mock simulation world
        self.mock_world = MagicMock()

    def tearDown(self):
        """Clean up after each test method."""
        self.context.close_stage()

    @parameterized.expand([
        ("/Target", "Target path"),
        ("/Franka", "Franka path")
    ])
    def test_init_with_valid_paths(self, prim_path, test_desc):
        """Test initialization with valid paths from basic.usda."""
        annotator = BaseAnnotator(
            name="test_annotator", 
            prim_path=prim_path,
            publishers=[self.mock_publisher],
            subscribers=[self.mock_subscriber]
        )

        self.assertEqual(annotator.name, "test_annotator")
        self.assertEqual(annotator.prim_path, prim_path)
        self.assertEqual(len(annotator.publishers), 1)
        self.assertEqual(len(annotator.subscribers), 1)
        self.assertIsNotNone(annotator.sensor_prim)
        self.assertTrue(annotator.sensor_prim.IsValid())

    @parameterized.expand(TEST_CASES)
    def test_init_with_none_cases(self, name, publishers, subscribers, 
                                expected_pub_len, expected_sub_len, expected_pub_idx):
        """Test initialization with None/mixed None publishers and subscribers."""
        if publishers and isinstance(publishers[1], MagicMock):
            publishers[1] = self.mock_publisher
            
        annotator = BaseAnnotator(
            name="test_annotator",
            prim_path="/Target",
            publishers=publishers,
            subscribers=subscribers
        )
        
        self.assertEqual(len(annotator.publishers), expected_pub_len)
        self.assertEqual(len(annotator.subscribers), expected_sub_len)
        if expected_pub_idx is not None:
            self.assertEqual(annotator.publishers[expected_pub_idx], self.mock_publisher)
        self.assertIsNotNone(annotator.sensor_prim)
        self.assertTrue(annotator.sensor_prim.IsValid())

    def test_start_and_stop(self):
        """Test start and stop methods with real USD stage."""
        annotator = BaseAnnotator(
            name="test_annotator",
            prim_path="/Target",
            publishers=[self.mock_publisher],
            subscribers=[self.mock_subscriber]
        )
        
        # Test start
        annotator.start(self.mock_world)
        self.mock_subscriber.start.assert_called_once()
        
        # Test stop
        annotator.stop(self.mock_world)
        self.mock_subscriber.stop.assert_called_once()

    def test_error_handling_invalid_prim_path(self):
        """Test handling of invalid prim path with real USD stage."""
        annotator = BaseAnnotator(
            name="test_annotator",
            prim_path="/invalid/path",
            publishers=[self.mock_publisher],
            subscribers=[self.mock_subscriber]
        )
        self.assertIsNotNone(annotator.sensor_prim)
        self.assertFalse(annotator.sensor_prim.IsValid())


if __name__ == '__main__':
    unittest.main()
