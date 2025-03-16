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

import torch
from simulation.environments.state_machine.modules.force_module import ForceControlModule
from simulation.environments.state_machine.utils import SMState, UltrasoundState


class TestForceControlModule(unittest.TestCase):
    """Test cases for the ForceControlModule class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Use CPU for testing
        self.device = "cpu"

        # Create the force control module with default parameters
        self.force_module = ForceControlModule(
            device=self.device, use_quaternion=False, desired_force_sum=0.30, Kp=0.5, Ki=0.0, Kd=0.0
        )

        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.unwrapped.num_envs = 1

        # Create a mock SMState
        self.sm_state = MagicMock(spec=SMState)
        self.sm_state.contact_normal_force = torch.zeros((1), device=self.device)
        self.sm_state.state = UltrasoundState.SETUP

    def test_initialization(self):
        """Test initialization of the force control module."""
        # Test default initialization
        module = ForceControlModule(device=self.device)
        self.assertEqual(module.device, self.device)
        self.assertEqual(module.state_dim, 6)  # Default is 6 (not using quaternion)
        self.assertEqual(module.desired_force_sum, 0.30)
        self.assertEqual(module.Kp, 0.5)
        self.assertEqual(module.Ki, 0.0)
        self.assertEqual(module.Kd, 0.0)
        self.assertEqual(module.integral, 0.0)
        self.assertEqual(module.prev_error, 0.0)
        self.assertEqual(module.dt, 1.0)
        self.assertEqual(module.active_states, [str(UltrasoundState.SCANNING), str(UltrasoundState.CONTACT)])

        # Test with custom parameters
        custom_module = ForceControlModule(
            device=self.device, use_quaternion=True, desired_force_sum=0.5, Kp=1.0, Ki=0.1, Kd=0.2
        )
        self.assertEqual(custom_module.state_dim, 7)  # Should be 7 when using quaternion
        self.assertEqual(custom_module.desired_force_sum, 0.5)
        self.assertEqual(custom_module.Kp, 1.0)
        self.assertEqual(custom_module.Ki, 0.1)
        self.assertEqual(custom_module.Kd, 0.2)

    def test_reset(self):
        """Test reset functionality."""
        # Change PID state
        self.force_module.integral = 1.0
        self.force_module.prev_error = 0.5

        # Reset
        self.force_module.reset()

        # Check that PID state is reset
        self.assertEqual(self.force_module.integral, 0.0)
        self.assertEqual(self.force_module.prev_error, 0.0)

    def test_compute_action_inactive_state(self):
        """Test compute_action in an inactive state (SETUP)."""
        # Set state to SETUP (not in active_states)
        self.sm_state.state = UltrasoundState.SETUP

        # Compute action
        action, updated_state = self.force_module.compute_action(self.mock_env, self.sm_state)

        # Action should be zeros
        self.assertTrue(torch.allclose(action, torch.zeros((1, self.force_module.state_dim), device=self.device)))

        # State should not change
        self.assertEqual(updated_state.state, UltrasoundState.SETUP)

        # PID state should not change
        self.assertEqual(self.force_module.integral, 0.0)
        self.assertEqual(self.force_module.prev_error, 0.0)

    def test_compute_action_contact_state_no_force(self):
        """Test compute_action in CONTACT state with no force."""
        # Set state to CONTACT
        self.sm_state.state = UltrasoundState.CONTACT

        # Set contact force to zero
        self.sm_state.contact_normal_force = torch.zeros((1), device=self.device)

        # Compute action
        action, updated_state = self.force_module.compute_action(self.mock_env, self.sm_state)

        # Action should have a negative value in z-direction (pushing down)
        self.assertLess(action[0, 2].item(), 0)

        # State should not change because there's no contact force
        self.assertEqual(updated_state.state, UltrasoundState.CONTACT)

        # PID state should update
        self.assertNotEqual(self.force_module.integral, 0.0)
        self.assertNotEqual(self.force_module.prev_error, 0.0)

    def test_compute_action_contact_state_with_force(self):
        """Test compute_action in CONTACT state with force."""
        # Set state to CONTACT
        self.sm_state.state = UltrasoundState.CONTACT

        # Set contact force to non-zero
        self.sm_state.contact_normal_force = torch.ones((1), device=self.device)

        # Compute action
        action, updated_state = self.force_module.compute_action(self.mock_env, self.sm_state)

        # Action should have a small positive value in z-direction (maintaining contact)
        self.assertGreater(action[0, 2].item(), 0)

        # State should change to SCANNING because there's contact force
        self.assertEqual(updated_state.state, UltrasoundState.SCANNING)

        # PID state should update
        self.assertNotEqual(self.force_module.integral, 0.0)
        self.assertNotEqual(self.force_module.prev_error, 0.0)

    def test_compute_action_scanning_state(self):
        """Test compute_action in SCANNING state."""
        # Set state to SCANNING
        self.sm_state.state = UltrasoundState.SCANNING

        # Set contact force to non-zero
        self.sm_state.contact_normal_force = torch.ones((1), device=self.device)

        # Compute action
        action, updated_state = self.force_module.compute_action(self.mock_env, self.sm_state)

        # Action should have a small positive value in z-direction (maintaining contact)
        self.assertGreater(action[0, 2].item(), 0)

        # State should not change
        self.assertEqual(updated_state.state, UltrasoundState.SCANNING)

        # PID state should update
        self.assertNotEqual(self.force_module.integral, 0.0)
        self.assertNotEqual(self.force_module.prev_error, 0.0)

    def test_pid_clamping(self):
        """Test that PID output is properly clamped."""
        # Set state to SCANNING
        self.sm_state.state = UltrasoundState.SCANNING

        # Set contact force to zero to get a large negative error
        self.sm_state.contact_normal_force = torch.zeros((1), device=self.device)

        # Set a very high Kp to generate a large output
        self.force_module.Kp = 100.0

        # Compute action
        action, _ = self.force_module.compute_action(self.mock_env, self.sm_state)

        # Action should be clamped to max value (0.01)
        self.assertLessEqual(action[0, 2].item(), 0.01)

    def test_pid_integration(self):
        """Test PID integration over multiple steps."""
        # Set state to SCANNING
        self.sm_state.state = UltrasoundState.SCANNING

        # Set contact force to zero
        self.sm_state.contact_normal_force = torch.zeros((1), device=self.device)

        # Set Ki to non-zero to test integration
        self.force_module.Ki = 0.1

        # Initial integral value
        initial_integral = self.force_module.integral

        # Compute action for first step
        self.force_module.compute_action(self.mock_env, self.sm_state)

        # Integral should decrease by exactly -0.04 (error * dt)
        first_integral = self.force_module.integral.item()
        self.assertAlmostEqual(first_integral, initial_integral - 0.04, places=5)

        # Compute action for second step
        self.force_module.compute_action(self.mock_env, self.sm_state)

        # Integral should decrease by exactly -0.04 again
        second_integral = self.force_module.integral.item()
        self.assertAlmostEqual(second_integral, first_integral - 0.04, places=5)

        # Overall, the integral should have decreased by -0.08 from initial
        self.assertAlmostEqual(second_integral, initial_integral - 0.08, places=5)


if __name__ == "__main__":
    unittest.main()
