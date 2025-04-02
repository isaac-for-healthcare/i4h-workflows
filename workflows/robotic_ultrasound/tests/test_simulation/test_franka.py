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

import numpy as np
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from helpers import requires_rti
from isaacsim import SimulationApp
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets

simulation_app = SimulationApp({"headless": True})

import omni.usd
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.world import World
from simulation.annotators.franka import FrankaPublisher, FrankaSubscriber


@requires_rti
class TestFrankaBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.usda_path = robot_us_assets.basic
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


if __name__ == "__main__":
    unittest.main()
