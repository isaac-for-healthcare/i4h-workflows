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
import time

import numpy as np
import pygame
from common.utils import get_ntp_offset
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.gamepad_event import GamepadEvent


class GamepadOp(Operator):
    """
    Operator to interface with Gamepad/Joystick.
    """

    def __init__(self, fragment, device_idx, *args, **kwargs):
        """
        Initialize the Gamepad operator.

        Parameters:
        - device_idx (int): device index.
        """
        self.device_idx = device_idx
        self.ntp_offset_time = get_ntp_offset()
        self.joystick = None

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise "No joystick connected"

        self.joystick = pygame.joystick.Joystick(self.device_idx)
        self.joystick.init()
        print(f"Joystick initialized: {self.joystick.get_name()}")

    def compute(self, op_input, op_output, context):
        event = pygame.event.wait()

        ts = int((time.time() + self.ntp_offset_time) * 1000)
        e = GamepadEvent(
            ts=ts,
            type=event.type,
            os=os.name,
            name=self.joystick.get_name(),
            axes=[self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())],
            buttons=[self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())],
            hats=np.array([self.joystick.get_hat(i) for i in range(self.joystick.get_numhats())], dtype=float)
            .flatten()
            .tolist(),
        )

        op_output.emit(e, "output")
