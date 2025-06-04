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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict

# Controller settings
STICK_DEADZONE = 0.1
TOOL_HOME_TIMEOUT = 3.0  # seconds


# Light ring states
@dataclass
class LightRingState:
    red: int
    green: int
    blue: int
    effect: str = "On"
    effect_param: float = 31.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "effect": self.effect,
            "effect_param": self.effect_param,
        }


# Light ring states
LIGHT_RING_SOLID_RED = LightRingState(red=255, green=0, blue=0)
LIGHT_RING_SOLID_BLUE = LightRingState(red=0, green=0, blue=255)
LIGHT_RING_SOLID_GREEN = LightRingState(red=0, green=255, blue=0)


class ControlMode(Enum):
    """Available control modes for the gamepad."""

    CARTESIAN = auto()
    POLAR = auto()

    def cycle(self) -> "ControlMode":
        """Return the next control mode in the cycle."""
        if self == ControlMode.CARTESIAN:
            return ControlMode.POLAR
        return ControlMode.CARTESIAN

    def lightring_state(self) -> LightRingState:
        """Get the light ring state associated with this control mode."""
        if self == ControlMode.CARTESIAN:
            return LIGHT_RING_SOLID_BLUE
        return LIGHT_RING_SOLID_GREEN


# Default poses and positions
MIRA_READY_POSE: Dict[str, Any] = {
    "left": [15.7, 80.6, 45.0, 98.9, 0.0, 0.0],
    "right": [15.7, 80.6, 45.0, 98.9, 0.0, 0.0],
}

LINUX_XBOX_MAP = {
    "South": lambda o: o.buttons[0],
    "East": lambda o: o.buttons[1],
    "West": lambda o: o.buttons[2],
    "North": lambda o: o.buttons[3],
    "Select": lambda o: o.buttons[6],
    "Start": lambda o: o.buttons[7],
    "Back": lambda o: o.buttons[11],
    "Left Trigger": lambda o: o.buttons[4],
    "Right Trigger": lambda o: o.buttons[5],
    "Left Bumper": lambda o: o.axes[2] > 0,
    "Right Bumper": lambda o: o.axes[5] > 0,
    "Left Stick": lambda o: o.buttons[9],
    "Right Stick": lambda o: o.buttons[10],
    "Left Stick X": lambda o: o.axes[0],
    "Left Stick Y": lambda o: o.axes[1],
    "Right Stick X": lambda o: o.axes[3],
    "Right Stick Y": lambda o: o.axes[4],
    "Dpad X": lambda o: o.hats[0],
    "Dpad Y": lambda o: o.hats[1],
}

LINUX_PRO_CONTROLLER_MAP = {
    "South": lambda o: o.buttons[0],
    "East": lambda o: o.buttons[1],
    "West": lambda o: o.buttons[3],
    "North": lambda o: o.buttons[2],
    "Select": lambda o: o.buttons[4],
    "Start": lambda o: o.buttons[11],
    "Back": lambda o: o.buttons[9],
    "Minus": lambda o: o.buttons[9],
    "Plus": lambda o: o.buttons[10],
    "Left Trigger": lambda o: o.buttons[5],
    "Right Trigger": lambda o: o.buttons[6],
    "Left Bumper": lambda o: o.buttons[7],
    "Right Bumper": lambda o: o.buttons[8],
    "Left Stick": lambda o: o.buttons[12],
    "Right Stick": lambda o: o.buttons[13],
    "Left Stick X": lambda o: o.axes[0],
    "Left Stick Y": lambda o: o.axes[1],
    "Right Stick X": lambda o: o.axes[2],
    "Right Stick Y": lambda o: o.axes[3],
    "Dpad X": lambda o: o.hats[0],
    "Dpad Y": lambda o: o.hats[1],
}

WINDOWS_XBOX_MAP = {
    "South": lambda o: o.buttons[0],
    "East": lambda o: o.buttons[1],
    "West": lambda o: o.buttons[2],
    "North": lambda o: o.buttons[3],
    "Select": lambda o: o.buttons[6],
    "Start": lambda o: o.buttons[7],
    "Back": lambda o: o.buttons[11],
    "Left Trigger": lambda o: o.buttons[4],
    "Right Trigger": lambda o: o.buttons[5],
    "Left Bumper": lambda o: o.axes[4] > 0,
    "Right Bumper": lambda o: o.axes[5] > 0,
    "Left Stick": lambda o: o.buttons[8],
    "Right Stick": lambda o: o.buttons[9],
    "Left Stick X": lambda o: o.axes[0],
    "Left Stick Y": lambda o: o.axes[1],
    "Right Stick X": lambda o: o.axes[2],
    "Right Stick Y": lambda o: o.axes[3],
    "Dpad X": lambda o: o.hats[0],
    "Dpad Y": lambda o: o.hats[1],
}

WINDOWS_PRO_CONTROLLER_MAP = {
    "South": lambda o: o.buttons[1],
    "East": lambda o: o.buttons[0],
    "West": lambda o: o.buttons[3],
    "North": lambda o: o.buttons[2],
    "Select": lambda o: o.buttons[15],
    "Start": lambda o: o.buttons[5],
    "Back": lambda o: o.buttons[4],
    "Minus": lambda o: o.buttons[4],
    "Plus": lambda o: o.buttons[6],
    "Left Trigger": lambda o: o.buttons[9],
    "Right Trigger": lambda o: o.buttons[10],
    "Left Bumper": lambda o: o.axes[4] > 0,
    "Right Bumper": lambda o: o.axes[5] > 0,
    "Left Stick": lambda o: o.buttons[7],
    "Right Stick": lambda o: o.buttons[8],
    "Left Stick X": lambda o: o.axes[0],
    "Left Stick Y": lambda o: o.axes[1],
    "Right Stick X": lambda o: o.axes[2],
    "Right Stick Y": lambda o: o.axes[3],
    "Dpad X": lambda o: 1 if o.buttons[13] else -1 if o.buttons[14] else 0,
    "Dpad Y": lambda o: 1 if o.buttons[11] else -1 if o.buttons[12] else 0,
}
