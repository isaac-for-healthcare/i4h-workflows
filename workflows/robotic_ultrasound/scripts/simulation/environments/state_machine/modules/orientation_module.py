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

import torch
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from simulation.environments.state_machine.utils import RobotQuaternions, SMState, UltrasoundState

from .base_module import BaseControlModule


class OrientationControlModule(BaseControlModule):
    """Module for orientation control."""

    def __init__(self, device: str = "cuda:0", use_quaternion: bool = False):
        """Initialize orientation control module.

        Args:
            device: Torch device
            use_quaternion: Whether to use quaternion for orientation
        """
        self.down_quaternion = torch.tensor(RobotQuaternions.DOWN, device=device)
        super().__init__(device, use_quaternion)

    def compute_action(self, env, sm_state: SMState) -> tuple[torch.Tensor, SMState]:
        """Compute orientation control action.

        Args:
            env: Environment instance
            sm_state: Current state machine state

        Returns:
            Tuple of computed action and updated state
        """
        base_action = self.get_base_action(env)

        # Check for state changes
        if str(sm_state.state) == str(UltrasoundState.SETUP):
            current_quat = sm_state.robot_obs[0, 3:]
            # Use quaternion distance instead of direct subtraction
            quat_distance = self.quaternion_distance(current_quat, self.down_quaternion)
            if quat_distance < 0.1:  # Threshold in radians (about 5.7 degrees)
                sm_state.state = UltrasoundState.APPROACH

        # If not making contact, keep the robot down
        if str(sm_state.state) != str(UltrasoundState.SCANNING):
            base_action[:, 3:] = self.down_quaternion
        else:
            normal_force = sm_state.contact_normal_force
            if torch.any(normal_force != 0):  # Only adjust orientation if there's contact
                roll, pitch, _ = euler_xyz_from_quat(self.down_quaternion.unsqueeze(0))

                # Get yaw relative to torso
                yaw = self.get_torso_yaw(env)
                # Convert to quaternion
                # Note: pitch and roll are swapped due to the EE frame being upside down
                #       during the scan.
                new_quat = quat_from_euler_xyz(pitch, roll, yaw)
                base_action[:, 3:] = new_quat

        return base_action, sm_state

    def reset(self) -> None:
        """Reset module state."""
        pass

    def get_torso_yaw(self, env):
        """Get the torso yaw from the environment."""
        torso_data = env.unwrapped.scene["organs"].data
        torso_quat = torso_data.root_quat_w
        _, _, yaw = euler_xyz_from_quat(torso_quat)
        # Rotate yaw by 180 degrees (π radians)
        yaw -= torch.pi
        return yaw

    def quaternion_distance(self, q1, q2):
        """Calculate the angular distance between two quaternions.

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Angular distance in radians
        """
        # Ensure unit quaternions
        q1_normalized = q1 / torch.norm(q1)
        q2_normalized = q2 / torch.norm(q2)

        # Compute dot product
        dot_product = torch.sum(q1_normalized * q2_normalized)

        # Handle numerical errors
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Account for quaternion double covering (q and -q represent same rotation)
        dot_product = torch.abs(dot_product)

        # Calculate angle
        return 2.0 * torch.acos(dot_product)
