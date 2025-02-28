import os
import unittest
from importlib.util import find_spec
from unittest import skipUnless
from unittest.mock import MagicMock

import numpy as np
from isaacsim import SimulationApp
from parameterized import parameterized
from rti_dds.publisher import Publisher
from rti_dds.schemas.camera_ctrl import CameraCtrlInput
from rti_dds.schemas.camera_info import CameraInfo
from rti_dds.schemas.franka_ctrl import FrankaCtrlInput
from rti_dds.schemas.franka_info import FrankaInfo
from rti_dds.schemas.target_ctrl import TargetCtrlInput
from rti_dds.schemas.target_info import TargetInfo
from rti_dds.schemas.usp_info import UltraSoundProbeInfo
from rti_dds.subscriber import Subscriber

simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.robots import Robot  # noqa: E402
from simulation.annotators.base import Annotator  # noqa: E402
from simulation.annotators.camera import CameraPublisher, CameraSubscriber  # noqa: E402
from simulation.annotators.franka import FrankaPublisher, FrankaSubscriber  # noqa: E402
from simulation.annotators.target import TargetPublisher, TargetSubscriber  # noqa: E402
from simulation.annotators.ultrasound import UltraSoundPublisher  # noqa: E402

TEST_CASES = [
    ("none_publishers_subscribers", None, None, 0, 0, None),
    ("mixed_none_publishers", [None, MagicMock(spec=Publisher), None], [MagicMock(spec=Subscriber)], 1, 1, 0),
]

try:
    RTI_AVAILABLE = bool(find_spec("rti.connextdds"))
    if RTI_AVAILABLE:
        license_path = os.getenv("RTI_LICENSE_FILE")
        RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestFrankaBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = os.getenv("BASIC_USD_PATH", os.path.join(cls.test_dir, "basic.usda"))
        cls.franka_prim_path = "/Franka"
        assert os.path.exists(cls.usda_path), f"basic.usda not found at {cls.usda_path}"

    def setUp(self):
        omni.usd.get_context().open_stage(self.usda_path)
        self.stage = omni.usd.get_context().get_stage()

        self.franka = Robot(prim_path=self.franka_prim_path, name="franka", position=np.array([0.0, 0.0, 0.0]))
        self.simulation_world = World()
        self.simulation_world.scene.add(self.franka)
        self.simulation_world.reset()

    def tearDown(self):
        omni.usd.get_context().close_stage()
        self.simulation_world.stop()
        simulation_app.update()


class TestFrankaPublisher(TestFrankaBase):
    def test_init_publisher(self):
        publisher = FrankaPublisher(
            franka=self.franka,
            ik=True,
            prim_path=self.franka_prim_path,
            topic="franka_info",
            period=1 / 30.0,
            domain_id=60,
        )

        self.assertIsNotNone(publisher)
        self.assertEqual(publisher.prim_path, self.franka_prim_path)
        self.assertTrue(publisher.ik)

    def test_produce_robot_state(self):
        publisher = FrankaPublisher(
            franka=self.franka,
            ik=True,
            prim_path=self.franka_prim_path,
            topic="franka_info",
            period=1 / 30.0,
            domain_id=60,
        )
        self.simulation_world.step(render=True)
        robot_info = publisher.produce(0.033, 1.0)

        self.assertIsNotNone(robot_info)
        self.assertIsInstance(robot_info, FrankaInfo)

        self.assertEqual(len(robot_info.joints_state_positions), 9)
        self.assertEqual(len(robot_info.joints_state_velocities), 9)


class TestFrankaSubscriber(TestFrankaBase):
    def test_init_subscriber(self):
        subscriber = FrankaSubscriber(
            franka=self.franka,
            ik=True,
            prim_path=self.franka_prim_path,
            topic="franka_ctrl",
            period=1 / 30.0,
            domain_id=60,
        )

        self.assertIsNotNone(subscriber)
        self.assertEqual(subscriber.prim_path, self.franka_prim_path)
        self.assertTrue(subscriber.ik)
        self.assertIsNotNone(subscriber.franka_controller)

    def test_joint_control(self):
        subscriber = FrankaSubscriber(
            franka=self.franka,
            ik=False,
            prim_path=self.franka_prim_path,
            topic="franka_ctrl",
            period=1 / 30.0,
            domain_id=60,
        )

        init_joint_positions = self.franka.get_joints_state().positions
        # Create control input with joint positions
        ctrl_input = FrankaCtrlInput()
        ctrl_input.joint_positions = [1.0] * 9

        subscriber.consume(ctrl_input)
        self.simulation_world.step(render=True)

        joints_state = self.franka.get_joints_state()
        self.assertFalse(all(a == b for a, b in zip(joints_state.positions, init_joint_positions)))

    def test_cartesian_control_ik(self):
        """Test Cartesian control using IK."""
        subscriber = FrankaSubscriber(
            franka=self.franka,
            ik=True,
            prim_path=self.franka_prim_path,
            topic="franka_ctrl",
            period=1 / 30.0,
            domain_id=60,
        )

        init_position = self.franka.get_joints_state().positions
        # Create control input with target position/orientation
        ctrl_input = FrankaCtrlInput()
        ctrl_input.target_position = [0.5, 0.0, 0.5]
        ctrl_input.target_orientation = [0.0, 0.0, 0.0, 1.0]

        subscriber.consume(ctrl_input)
        self.simulation_world.step(render=True)

        end_effector_position = self.franka.get_joints_state().positions
        self.assertFalse(np.allclose(end_effector_position, init_position))


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestCameraBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = os.getenv("BASIC_USD_PATH", os.path.join(cls.test_dir, "basic.usda"))
        cls.camera_prim_path = "/RoomCamera"
        assert os.path.exists(cls.usda_path), f"basic.usda not found at {cls.usda_path}"

    def setUp(self):
        omni.usd.get_context().open_stage(self.usda_path)
        simulation_app.update()
        self.stage = omni.usd.get_context().get_stage()

    def tearDown(self):
        omni.usd.get_context().close_stage()
        simulation_app.update()


class TestCameraPublisher(TestCameraBase):
    def test_init_rgb_camera(self):
        publisher = CameraPublisher(
            annotator="rgb",
            prim_path=self.camera_prim_path,
            height=480,
            width=640,
            topic="camera_rgb",
            period=1 / 30.0,
            domain_id=60,
        )

        self.assertEqual(publisher.annotator, "rgb")
        self.assertEqual(publisher.prim_path, self.camera_prim_path)
        self.assertEqual(publisher.resolution, (640, 480))
        self.assertIsNotNone(publisher.rp)

        camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)
        self.assertTrue(camera_prim.IsValid())

    @parameterized.expand([("rgb", "camera_rgb", np.uint8), ("distance_to_camera", "camera_depth", np.float32)])
    def test_camera_data(self, annotator, topic, dtype):
        simulation_app.update()

        publisher = CameraPublisher(
            annotator=annotator,
            prim_path=self.camera_prim_path,
            height=480,
            width=640,
            topic=topic,
            period=1 / 30.0,
            domain_id=60,
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
    def test_consume_focal_length(self):
        subscriber = CameraSubscriber(
            prim_path=self.camera_prim_path, topic="camera_ctrl", period=1 / 30.0, domain_id=60
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


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the USD stage with basic.usda file."""
        # Get path to basic.usda
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.basic_usd_path = os.getenv("BASIC_USD_PATH", os.path.join(cls.test_dir, "basic.usda"))
        assert os.path.exists(cls.basic_usd_path), f"basic.usda not found at {cls.basic_usd_path}"

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Load the basic USD file
        from pxr import Usd

        self.stage = Usd.Stage.Open(self.basic_usd_path)
        self.context = omni.usd.get_context()
        self.context.open_stage(self.basic_usd_path)

        # Mock publishers and subscribers
        self.mock_publisher = MagicMock(spec=Publisher)
        self.mock_publisher.topic = "test_pub_topic"
        self.mock_publisher.period = 1 / 30.0
        self.mock_publisher.write = MagicMock()

        self.mock_subscriber = MagicMock(spec=Subscriber)
        self.mock_subscriber.topic = "test_sub_topic"
        self.mock_subscriber.period = 1 / 10.0
        self.mock_subscriber.read = MagicMock()
        self.mock_subscriber.start = MagicMock()
        self.mock_subscriber.stop = MagicMock()

        # Mock simulation world
        self.mock_world = MagicMock()

    def tearDown(self):
        """Clean up after each test method."""
        self.context.close_stage()

    @parameterized.expand([("/Target", "Target path"), ("/Franka", "Franka path")])
    def test_init_with_valid_paths(self, prim_path, test_desc):
        """Test initialization with valid paths from basic.usda."""
        annotator = Annotator(
            name="test_annotator",
            prim_path=prim_path,
            publishers=[self.mock_publisher],
            subscribers=[self.mock_subscriber],
        )

        self.assertEqual(annotator.name, "test_annotator")
        self.assertEqual(annotator.prim_path, prim_path)
        self.assertEqual(len(annotator.publishers), 1)
        self.assertEqual(len(annotator.subscribers), 1)
        self.assertIsNotNone(annotator.sensor_prim)
        self.assertTrue(annotator.sensor_prim.IsValid())

    @parameterized.expand(TEST_CASES)
    def test_init_with_none_cases(
        self, name, publishers, subscribers, expected_pub_len, expected_sub_len, expected_pub_idx
    ):
        """Test initialization with None/mixed None publishers and subscribers."""
        if publishers and isinstance(publishers[1], MagicMock):
            publishers[1] = self.mock_publisher

        annotator = Annotator(
            name="test_annotator", prim_path="/Target", publishers=publishers, subscribers=subscribers
        )

        self.assertEqual(len(annotator.publishers), expected_pub_len)
        self.assertEqual(len(annotator.subscribers), expected_sub_len)
        if expected_pub_idx is not None:
            self.assertEqual(annotator.publishers[expected_pub_idx], self.mock_publisher)
        self.assertIsNotNone(annotator.sensor_prim)
        self.assertTrue(annotator.sensor_prim.IsValid())

    def test_start_and_stop(self):
        """Test start and stop methods with real USD stage."""
        annotator = Annotator(
            name="test_annotator",
            prim_path="/Target",
            publishers=[self.mock_publisher],
            subscribers=[self.mock_subscriber],
        )

        # Test start
        annotator.start(self.mock_world)
        self.mock_subscriber.start.assert_called_once()

        # Test stop
        annotator.stop(self.mock_world)
        self.mock_subscriber.stop.assert_called_once()

    def test_error_handling_invalid_prim_path(self):
        """Test handling of invalid prim path with real USD stage."""
        annotator = Annotator(
            name="test_annotator",
            prim_path="/invalid/path",
            publishers=[self.mock_publisher],
            subscribers=[self.mock_subscriber],
        )
        self.assertIsNotNone(annotator.sensor_prim)
        self.assertFalse(annotator.sensor_prim.IsValid())


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestTargetBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = os.getenv("BASIC_USD_PATH", os.path.join(cls.test_dir, "basic.usda"))
        cls.target_prim_path = "/Target"
        assert os.path.exists(cls.usda_path), f"basic.usda not found at {cls.usda_path}"

    def setUp(self):
        omni.usd.get_context().open_stage(self.usda_path)
        self.stage = omni.usd.get_context().get_stage()

        simulation_app.update()

    def tearDown(self):
        omni.usd.get_context().close_stage()
        simulation_app.update()


class TestTargetPublisher(TestTargetBase):
    def test_init_target(self):
        target_prim = self.stage.GetPrimAtPath(self.target_prim_path)
        self.assertTrue(target_prim.IsValid())
        self.assertEqual(target_prim.GetTypeName(), "Mesh")

    def test_produce_target_data(self):
        publisher = TargetPublisher(prim_path=self.target_prim_path, topic="target_info", period=1 / 30.0, domain_id=60)

        simulation_app.update()

        target_info = publisher.produce(0.033, 1.0)
        self.assertIsNotNone(target_info)
        self.assertIsInstance(target_info, TargetInfo)

        self.assertEqual(len(target_info.position), 3)
        self.assertEqual(len(target_info.orientation), 4)  # Quaternion

        self.assertTrue(all(isinstance(x, float) for x in target_info.position))
        self.assertTrue(all(isinstance(x, float) for x in target_info.orientation))


class TestTargetSubscriber(TestTargetBase):
    def test_target_control(self):
        subscriber = TargetSubscriber(
            prim_path=self.target_prim_path, topic="target_ctrl", period=1 / 30.0, domain_id=60
        )

        ctrl_input = TargetCtrlInput()
        new_position = [1.0, 2.0, 3.0]
        new_orientation = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
        ctrl_input.position = new_position
        ctrl_input.orientation = new_orientation

        subscriber.consume(ctrl_input)

        simulation_app.update()

        target_prim = self.stage.GetPrimAtPath(self.target_prim_path)
        current_position = target_prim.GetAttribute("xformOp:translate").Get()

        self.assertAlmostEqual(current_position[0], new_position[0], places=5)
        self.assertAlmostEqual(current_position[1], new_position[1], places=5)
        self.assertAlmostEqual(current_position[2], new_position[2], places=5)

    def test_target_movement_tracking(self):
        publisher = TargetPublisher(prim_path=self.target_prim_path, topic="target_info", period=1 / 30.0, domain_id=60)

        subscriber = TargetSubscriber(
            prim_path=self.target_prim_path, topic="target_ctrl", period=1 / 30.0, domain_id=60
        )

        initial_info = publisher.produce(0.033, 1.0)
        initial_position = initial_info.position

        ctrl_input = TargetCtrlInput()
        new_position = [2.0, 3.0, 4.0]
        ctrl_input.position = new_position
        ctrl_input.orientation = [0.0, 0.0, 0.0, 1.0]

        subscriber.consume(ctrl_input)

        simulation_app.update()

        updated_info = publisher.produce(0.033, 1.0)
        updated_position = updated_info.position

        self.assertNotEqual(initial_position, updated_position)
        self.assertAlmostEqual(updated_position[0], new_position[0], places=5)
        self.assertAlmostEqual(updated_position[1], new_position[1], places=5)
        self.assertAlmostEqual(updated_position[2], new_position[2], places=5)


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestUltraSoundBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = os.getenv("BASIC_USD_PATH", os.path.join(cls.test_dir, "basic.usda"))
        cls.us_prim_path = "/Target"
        assert os.path.exists(cls.usda_path), f"basic.usda not found at {cls.usda_path}"

    def setUp(self):
        omni.usd.get_context().open_stage(self.usda_path)
        self.stage = omni.usd.get_context().get_stage()
        simulation_app.update()

    def tearDown(self):
        omni.usd.get_context().close_stage()
        simulation_app.update()


class TestUltraSoundPublisher(TestUltraSoundBase):
    def test_init_probe(self):
        probe_prim = self.stage.GetPrimAtPath(self.us_prim_path)
        self.assertTrue(probe_prim.IsValid())
        self.assertEqual(probe_prim.GetTypeName(), "Mesh")

    def test_produce_probe_data(self):
        publisher = UltraSoundPublisher(
            prim_path="/UltrasoundProbe", topic="ultrasound_probe", period=1 / 30.0, domain_id=60
        )

        simulation_app.update()

        probe_info = publisher.produce(0.033, 1.0)
        self.assertIsNotNone(probe_info)
        self.assertIsInstance(probe_info, UltraSoundProbeInfo)

        self.assertEqual(len(probe_info.position), 3)
        self.assertEqual(len(probe_info.orientation), 4)

        self.assertTrue(all(isinstance(x, float) for x in probe_info.position))
        self.assertTrue(all(isinstance(x, float) for x in probe_info.orientation))

    def test_probe_movement(self):
        publisher = UltraSoundPublisher(
            prim_path=self.us_prim_path, topic="ultrasound_probe", period=1 / 30.0, domain_id=60
        )

        initial_info = publisher.produce(0.033, 1.0)
        initial_position = initial_info.position

        probe_prim = self.stage.GetPrimAtPath(self.us_prim_path)
        new_position = (1.0, 2.0, 3.0)
        probe_prim.GetAttribute("xformOp:translate").Set(new_position)

        simulation_app.update()

        updated_info = publisher.produce(0.033, 1.0)
        updated_position = updated_info.position

        self.assertNotEqual(initial_position, updated_position)
        self.assertAlmostEqual(updated_position[0], new_position[0], places=5)
        self.assertAlmostEqual(updated_position[1], new_position[1], places=5)
        self.assertAlmostEqual(updated_position[2], new_position[2], places=5)


if __name__ == "__main__":
    unittest.main()
