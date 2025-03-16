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
from unittest.mock import MagicMock, patch

import torch
from simulation.environments.state_machine.modules.orientation_module import OrientationControlModule
from simulation.environments.state_machine.utils import RobotQuaternions, SMState, UltrasoundState


class TestOrientationControlModule(unittest.TestCase):
    """Test cases for the OrientationControlModule class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Use CPU for testing
        self.device = "cpu"

        # Create the orientation module
        self.orientation_module = OrientationControlModule(device=self.device, use_quaternion=False)

        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.unwrapped.num_envs = 1

        # Create a mock SMState
        self.sm_state = MagicMock(spec=SMState)
        self.sm_state.robot_obs = torch.zeros((1, 7), device=self.device)  # [pos, quat]
        self.sm_state.contact_normal_force = torch.zeros((1), device=self.device)

        # Set up the down quaternion for comparison
        self.down_quaternion = torch.tensor(RobotQuaternions.DOWN, device=self.device)

    def test_initialization(self):
        """Test initialization of the orientation module."""
        # Test default initialization
        module = OrientationControlModule(device=self.device)
        self.assertEqual(module.device, self.device)
        # Check state_dim instead of use_quaternion
        self.assertEqual(module.state_dim, 6)  # Default is 6 (not using quaternion)
        self.assertTrue(torch.allclose(module.down_quaternion, self.down_quaternion))

        # Test with quaternion flag
        module_quat = OrientationControlModule(device=self.device, use_quaternion=True)
        self.assertEqual(module_quat.state_dim, 7)  # Should be 7 when using quaternion

    def test_reset(self):
        """Test reset functionality."""
        # Reset should not change any state in this module
        initial_down_quat = self.orientation_module.down_quaternion.clone()
        self.orientation_module.reset()
        self.assertTrue(torch.allclose(self.orientation_module.down_quaternion, initial_down_quat))

    def test_quaternion_distance(self):
        """Test the quaternion distance calculation."""
        # Same quaternion should have zero distance
        q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.assertAlmostEqual(self.orientation_module.quaternion_distance(q1, q1).item(), 0.0, places=5)

        # No rotation vs. a 180 degree rotation should have pi distance
        q2 = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        self.assertAlmostEqual(self.orientation_module.quaternion_distance(q1, q2).item(), torch.pi, places=5)

        # Test with non-normalized quaternions
        q3 = torch.tensor([2.0, 0.0, 0.0, 0.0], device=self.device)
        self.assertAlmostEqual(self.orientation_module.quaternion_distance(q1, q3).item(), 0.0, places=5)

    def test_compute_action_setup_state(self):
        """Test compute_action in SETUP state."""
        # Set state to SETUP
        self.sm_state.state = UltrasoundState.SETUP

        # Set current quaternion far from down_quaternion
        self.sm_state.robot_obs[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        # Mock get_base_action to return a zero tensor
        self.orientation_module.get_base_action = MagicMock(return_value=torch.zeros((1, 7), device=self.device))

        # Compute action
        action, updated_state = self.orientation_module.compute_action(self.mock_env, self.sm_state)

        # Check that action sets orientation to down_quaternion
        self.assertTrue(torch.allclose(action[0, 3:], self.down_quaternion))

        # State should not change because quaternion distance is large
        self.assertEqual(updated_state.state, UltrasoundState.SETUP)

        # Now test with quaternion close to down_quaternion
        self.sm_state.robot_obs[0, 3:] = self.down_quaternion * 0.99  # Very close
        action, updated_state = self.orientation_module.compute_action(self.mock_env, self.sm_state)

        # State should change to APPROACH
        self.assertEqual(updated_state.state, UltrasoundState.APPROACH)

    def test_compute_action_scanning_state(self):
        """Test compute_action in SCANNING state."""
        # Set state to SCANNING
        self.sm_state.state = UltrasoundState.SCANNING

        # Set contact force to non-zero
        self.sm_state.contact_normal_force = torch.ones((1), device=self.device)

        # Mock torso data
        torso_data = MagicMock()
        torso_data.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.mock_env.unwrapped.scene = {"organs": MagicMock(data=torso_data)}

        # Mock get_base_action
        self.orientation_module.get_base_action = MagicMock(return_value=torch.zeros((1, 7), device=self.device))

        # Mock euler_xyz_from_quat to return controlled values - make sure it returns a tuple of tensors
        with patch(
            "simulation.environments.state_machine.modules.orientation_module.euler_xyz_from_quat"
        ) as mock_euler:
            # First call for down_quaternion, second for torso_quat
            mock_euler.side_effect = [
                (
                    torch.tensor(0.1, device=self.device),
                    torch.tensor(0.2, device=self.device),
                    torch.tensor(0.0, device=self.device),
                ),  # For down_quaternion
                (
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.3, device=self.device),
                ),  # For torso_quat
            ]

            # Mock quat_from_euler_xyz
            with patch(
                "simulation.environments.state_machine.modules.orientation_module.quat_from_euler_xyz"
            ) as mock_quat:
                mock_quat.return_value = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device)

                # Compute action
                action, updated_state = self.orientation_module.compute_action(self.mock_env, self.sm_state)

                # Check that action sets orientation to the new quaternion
                self.assertTrue(torch.allclose(action[0, 3:], torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device)))

    def test_get_torso_yaw(self):
        """Test get_torso_yaw method."""
        # Mock torso data
        torso_data = MagicMock()
        torso_data.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.mock_env.unwrapped.scene = {"organs": MagicMock(data=torso_data)}

        # Mock euler_xyz_from_quat to return a specific yaw - make sure it returns a tuple of tensors
        with patch(
            "simulation.environments.state_machine.modules.orientation_module.euler_xyz_from_quat"
        ) as mock_euler:
            mock_euler.return_value = (
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.5, device=self.device),
            )

            # Get torso yaw
            yaw = self.orientation_module.get_torso_yaw(self.mock_env)

            # Check that yaw is rotated by pi - use torch.pi directly
            expected_value = 0.5 - torch.pi
            self.assertAlmostEqual(yaw.item(), expected_value, places=5)


if __name__ == "__main__":
    unittest.main()
