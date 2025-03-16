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
from simulation.environments.state_machine.modules.path_planning_module import PathPlanningModule
from simulation.environments.state_machine.utils import PhantomScanPositions, SMState, UltrasoundState


class TestPathPlanningModule(unittest.TestCase):
    """Test cases for the PathPlanningModule class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Use CPU for testing
        self.device = "cpu"

        # Create the path planning module
        self.path_planning_module = PathPlanningModule(device=self.device, use_quaternion=False)

        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.unwrapped.num_envs = 1

        # Create actual tensor data instead of MagicMocks
        self.object_pos = torch.ones(3, device=self.device)
        self.object_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.env_origins = torch.zeros(3, device=self.device)

        # Set up the mock scene with proper return values
        object_data = MagicMock()
        object_data.root_pos_w = self.object_pos
        object_data.root_quat_w = self.object_quat

        # Configure the mock scene
        self.mock_env.unwrapped.scene = MagicMock()
        self.mock_env.unwrapped.scene.__getitem__.return_value.data = object_data
        self.mock_env.unwrapped.scene.env_origins = self.env_origins

        # Create a mock SMState
        self.sm_state = MagicMock(spec=SMState)
        self.sm_state.robot_obs = torch.zeros((1, 14), device=self.device)  # Assuming 14 dimensions
        self.sm_state.state = UltrasoundState.SETUP

    def test_initialization(self):
        """Test initialization of the path planning module."""
        # Test default initialization
        module = PathPlanningModule(device=self.device)
        self.assertEqual(module.device, self.device)
        # Check state_dim instead of use_quaternion
        self.assertEqual(module.state_dim, 6)  # Default is 6 (not using quaternion)

        # Check that positions are initialized correctly
        self.assertTrue(
            torch.allclose(
                module.start_pos, torch.tensor(PhantomScanPositions.SCAN_START, device=self.device).unsqueeze(0)
            )
        )
        self.assertTrue(
            torch.allclose(
                module.contact_pos, torch.tensor(PhantomScanPositions.SCAN_CONTACT, device=self.device).unsqueeze(0)
            )
        )
        self.assertTrue(
            torch.allclose(module.end_pos, torch.tensor(PhantomScanPositions.SCAN_END, device=self.device).unsqueeze(0))
        )

        # Check that derived values are calculated correctly
        self.assertTrue(torch.allclose(module.rel_target_offset, module.end_pos - module.contact_pos))
        self.assertTrue(torch.allclose(module.increment, module.rel_target_offset / module.scan_steps))
        self.assertEqual(module.current_step, 0)

        # Test with quaternion flag
        module_quat = PathPlanningModule(device=self.device, use_quaternion=True)
        self.assertEqual(module_quat.state_dim, 7)  # Should be 7 when using quaternion

    def test_reset(self):
        """Test reset functionality."""
        # Set current step to non-zero
        self.path_planning_module.current_step = 50
        # Modify increment
        self.path_planning_module.increment = torch.zeros_like(self.path_planning_module.increment)

        # Reset
        self.path_planning_module.reset()

        # Check that values are reset
        self.assertEqual(self.path_planning_module.current_step, 0)
        self.assertTrue(
            torch.allclose(
                self.path_planning_module.increment,
                self.path_planning_module.rel_target_offset / self.path_planning_module.scan_steps,
            )
        )

    def test_compute_action_setup_state(self):
        """Test compute_action in SETUP state."""
        # Set state to SETUP
        self.sm_state.state = UltrasoundState.SETUP

        # Mock quat_apply_yaw to return a tensor
        with patch(
            "simulation.environments.state_machine.modules.path_planning_module.quat_apply_yaw"
        ) as mock_quat_apply:
            # Return a tensor when called
            mock_quat_apply.return_value = torch.tensor([[0.1, 0.2, 0.3]], device=self.device)

            # Compute action
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # Check that action sets position correctly
            expected_position = self.object_pos - self.env_origins + mock_quat_apply.return_value
            self.assertTrue(torch.allclose(action[0, :3], expected_position[0]))

            # State should not change
            self.assertEqual(updated_state.state, UltrasoundState.SETUP)

            # Check that quat_apply_yaw was called with correct arguments
            mock_quat_apply.assert_called_once_with(self.object_quat, self.path_planning_module.start_pos)

    def test_compute_action_approach_state(self):
        """Test compute_action in APPROACH state."""
        # Set state to APPROACH
        self.sm_state.state = UltrasoundState.APPROACH

        # Mock quat_apply_yaw to return a tensor
        with patch(
            "simulation.environments.state_machine.modules.path_planning_module.quat_apply_yaw"
        ) as mock_quat_apply:
            # Return a tensor when called
            mock_quat_apply.return_value = torch.tensor([[0.4, 0.5, 0.6]], device=self.device)

            # Set robot position far from target
            self.sm_state.robot_obs[0, :3] = torch.tensor([10.0, 10.0, 10.0], device=self.device)

            # Compute action
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # Check that action sets position correctly
            expected_position = self.object_pos - self.env_origins + mock_quat_apply.return_value
            self.assertTrue(torch.allclose(action[0, :3], expected_position[0]))

            # State should not change because robot is far from target
            self.assertEqual(updated_state.state, UltrasoundState.APPROACH)

            # Now test with robot close to target
            self.sm_state.robot_obs[0, :3] = expected_position[0]  # Very close
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # State should change to CONTACT
            self.assertEqual(updated_state.state, UltrasoundState.CONTACT)

            # Check that quat_apply_yaw was called with correct arguments
            mock_quat_apply.assert_called_with(self.object_quat, self.path_planning_module.contact_pos)

    def test_compute_action_scanning_state(self):
        """Test compute_action in SCANNING state."""
        # Set state to SCANNING
        self.sm_state.state = UltrasoundState.SCANNING

        # Create return values for quat_apply_yaw
        world_organ_start_pos = torch.tensor([[0.7, 0.8, 0.9]], device=self.device)
        world_increment = torch.tensor([[0.01, 0.01, 0.01]], device=self.device)

        # Mock quat_apply_yaw to return tensors
        with patch(
            "simulation.environments.state_machine.modules.path_planning_module.quat_apply_yaw"
        ) as mock_quat_apply:
            # Set up side_effect to return our predefined values
            mock_quat_apply.side_effect = [world_organ_start_pos, world_increment]

            # Compute action
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # Check that action sets position correctly
            expected_position = self.object_pos - self.env_origins + world_organ_start_pos + world_increment
            self.assertTrue(torch.allclose(action[0, :3], expected_position[0]))

            # Check that current_step was incremented
            self.assertEqual(self.path_planning_module.current_step, 1)

            # Reset the mock for the next tests
            mock_quat_apply.reset_mock()
            mock_quat_apply.side_effect = None
            mock_quat_apply.return_value = world_organ_start_pos  # Use a single return value for remaining tests

            # Run for scan_steps iterations
            self.path_planning_module.current_step = self.path_planning_module.scan_steps - 1
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # Check that current_step was incremented
            self.assertEqual(self.path_planning_module.current_step, self.path_planning_module.scan_steps)

            # Run for hold_steps iterations
            self.path_planning_module.current_step = (
                self.path_planning_module.scan_steps + self.path_planning_module.hold_steps - 1
            )
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # Check that current_step was incremented
            self.assertEqual(
                self.path_planning_module.current_step,
                self.path_planning_module.scan_steps + self.path_planning_module.hold_steps,
            )

            # Run one more iteration to complete the scan
            action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

            # State should change to DONE
            self.assertEqual(updated_state.state, UltrasoundState.DONE)

    def test_compute_action_other_states(self):
        """Test compute_action in other states."""
        # Set state to DONE
        self.sm_state.state = UltrasoundState.DONE

        # Compute action
        action, updated_state = self.path_planning_module.compute_action(self.mock_env, self.sm_state)

        # Action should be zeros
        self.assertTrue(torch.allclose(action, torch.zeros((1, 6), device=self.device)))

        # State should not change
        self.assertEqual(updated_state.state, UltrasoundState.DONE)


if __name__ == "__main__":
    unittest.main()
