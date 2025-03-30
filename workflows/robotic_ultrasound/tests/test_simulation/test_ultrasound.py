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

import os
import unittest
from importlib.util import find_spec
from unittest import skipUnless

from dds.schemas.usp_info import UltraSoundProbeInfo
from isaacsim import SimulationApp
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets
from ..helpers import requires_rti


simulation_app = SimulationApp({"headless": True})
import omni.usd
from simulation.annotators.ultrasound import UltraSoundPublisher


@requires_rti
class TestUltraSoundBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = robot_us_assets.basic
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
