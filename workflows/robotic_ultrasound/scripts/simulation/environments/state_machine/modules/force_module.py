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
from simulation.environments.state_machine.utils import SMState, UltrasoundState

from .base_module import BaseControlModule


class ForceControlModule(BaseControlModule):
    """Module for force-based control using a PID controller for contact force."""

    def __init__(
        self,
        device: str = "cuda:0",
        use_quaternion: bool = False,
        desired_force_sum: float = 0.30,
        Kp: float = 0.5,
        Ki: float = 0.0,
        Kd: float = 0.0,
    ):
        """Initialize force control module.

        Args:
            device: Torch device
            use_quaternion: Whether to use quaternion for orientation
            desired_force_sum: The desired sum of absolute forces
            Kp: Proportional gain for PID
            Ki: Integral gain for PID
            Kd: Derivative gain for PID
        """
        super().__init__(device, use_quaternion)
        self.desired_force_sum = desired_force_sum
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # PID bookkeeping
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 1.0
        # States when force should be applied
        self.active_states = [str(UltrasoundState.SCANNING), str(UltrasoundState.CONTACT)]

    def compute_action(self, env, sm_state: SMState) -> tuple[torch.Tensor, SMState]:
        """Compute force-based control action using PID controller.

        Args:
            env: Environment instance
            sm_state: Current state machine state

        Returns:
            Tuple of computed action and updated state
        """
        base_action = torch.zeros(env.unwrapped.num_envs, self.state_dim, device=self.device)
        force_sum = torch.sum(torch.abs(sm_state.contact_normal_force))

        # Check for state changes
        if str(sm_state.state) == str(UltrasoundState.CONTACT) and force_sum > 0:
            sm_state.state = UltrasoundState.SCANNING

        # If making contact, apply force
        if str(sm_state.state) in self.active_states:
            # Compute PID terms
            error = 0.001 if force_sum > 0 else -0.04
            error = torch.tensor(error, device=self.device)

            self.integral += error * self.dt
            derivative = (error - self.prev_error) / self.dt

            pid_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
            pid_output = torch.clamp(pid_output, max=0.01)
            base_action[:, 2] = pid_output
            self.prev_error = error

        return base_action, sm_state

    def reset(self) -> None:
        """Reset PID controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
