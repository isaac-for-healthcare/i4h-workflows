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

import unittest
from unittest.mock import MagicMock

import numpy as np
import omni.isaac.lab.utils.math as math_utils
import torch
from simulation.environments.state_machine.utils import compute_transform_sequence, get_probe_pos_ori


class TestComputeTransformSequence(unittest.TestCase):
    def setUp(self):
        # Create a mock environment
        self.env = MagicMock()
        self.env.unwrapped.scene = {}

        # Create mock transform objects
        self.transform_a_to_b = MagicMock()
        self.transform_b_to_c = MagicMock()

        # Set up quaternions and positions for transforms
        self.quat_a_to_b = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.pos_a_to_b = torch.tensor([1.0, 0.0, 0.0])  # 1 unit in x direction

        self.quat_b_to_c = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # 90 degree rotation around y
        self.pos_b_to_c = torch.tensor([0.0, 1.0, 0.0])  # 1 unit in y direction

        # Configure mock transform objects
        self.transform_a_to_b.data.target_quat_source = self.quat_a_to_b.unsqueeze(0)
        self.transform_a_to_b.data.target_pos_source = self.pos_a_to_b.unsqueeze(0)

        self.transform_b_to_c.data.target_quat_source = self.quat_b_to_c.unsqueeze(0)
        self.transform_b_to_c.data.target_pos_source = self.pos_b_to_c.unsqueeze(0)

        # Add transforms to the scene
        self.env.unwrapped.scene["a_to_b_transform"] = self.transform_a_to_b
        self.env.unwrapped.scene["b_to_c_transform"] = self.transform_b_to_c

    def test_compute_transform_sequence(self):
        # Test the transformation from A to C
        quat, pos = compute_transform_sequence(self.env, ["a", "b", "c"])

        # Expected results:
        # - Position should be [1.0, 1.0, 0.0] (first move 1 in x, then 1 in y)
        # - Quaternion should be the multiplication of the two quaternions
        expected_quat = math_utils.quat_mul(self.quat_a_to_b, self.quat_b_to_c)
        expected_pos = self.pos_a_to_b + math_utils.quat_apply(self.quat_a_to_b, self.pos_b_to_c)

        self.assertTrue(torch.allclose(quat, expected_quat, atol=1e-4))
        self.assertTrue(torch.allclose(pos, expected_pos, atol=1e-4))

    def test_single_frame_error(self):
        # Test that an error is raised when only one frame is provided
        with self.assertRaises(ValueError):
            compute_transform_sequence(self.env, ["a"])

    def test_empty_sequence_error(self):
        # Test that an error is raised when an empty sequence is provided
        with self.assertRaises(ValueError):
            compute_transform_sequence(self.env, [])

    def test_missing_transform_error(self):
        # Test that an error is raised when a transform is missing
        with self.assertRaises(Exception):
            compute_transform_sequence(self.env, ["a", "d", "c"])


class TestGetProbePositionOrientation(unittest.TestCase):
    def setUp(self):
        # Create test quaternions and positions
        self.quat_mesh_to_us = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
        self.pos_mesh_to_us = torch.tensor([[0.1, 0.2, 0.3]])  # Position in meters

        # Create numpy versions for testing
        self.quat_mesh_to_us_np = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos_mesh_to_us_np = np.array([0.1, 0.2, 0.3])

    def test_get_probe_pos_ori_torch_input(self):
        # Test with torch tensor inputs
        pos, euler = get_probe_pos_ori(self.quat_mesh_to_us, self.pos_mesh_to_us)

        # Check position scaling (meters to millimeters)
        expected_pos = np.array([100.0, 200.0, 300.0])
        expected_euler = np.array([0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(pos, expected_pos))
        self.assertTrue(np.allclose(euler, expected_euler))


if __name__ == "__main__":
    unittest.main()
