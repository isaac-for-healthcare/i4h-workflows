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

from abc import ABC, abstractmethod

import torch


class BaseControlModule(ABC):
    """Abstract base class for all control modules."""

    def __init__(self, device: str = "cuda:0", use_quaternion: bool = False):
        """Initialize the base control module.

        Args:
            device (str): Device to run computations on
            use_quaternion (bool): Whether to use quaternion for orientation control, else use euler angles
        """
        self.device = device
        self.state_dim = 7 if use_quaternion else 6

    @abstractmethod
    def compute_action(self, env, state) -> torch.Tensor:
        """Compute the control action for this module.

        Args:
            env: The environment instance
            state: The current state of the state machine

        Returns:
            torch.Tensor: The computed action
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the module's internal state."""
        pass

    def get_base_action(self, env):
        return torch.zeros(env.unwrapped.num_envs, self.state_dim, device=self.device)
