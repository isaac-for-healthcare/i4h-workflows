import os
import unittest
from importlib.util import find_spec
from unittest import skipUnless

from dds.schemas.target_ctrl import TargetCtrlInput
from dds.schemas.target_info import TargetInfo
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from simulation.annotators.target import TargetPublisher, TargetSubscriber  # noqa: E402
from workflows.robotic_ultrasound.scripts.utils.assets import robotic_ultrasound_assets as rus_assets

try:
    RTI_AVAILABLE = bool(find_spec("rti.connextdds"))
    if RTI_AVAILABLE:
        license_path = os.getenv("RTI_LICENSE_FILE")
        RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestTargetBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = rus_assets.basic
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


if __name__ == "__main__":
    unittest.main()
