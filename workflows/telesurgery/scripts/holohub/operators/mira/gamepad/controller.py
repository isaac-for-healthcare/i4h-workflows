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

import queue
import time
from typing import Any, Dict

import numpy as np
import pygame
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.gamepad_event import GamepadEvent

from .constants import (
    LIGHT_RING_SOLID_RED,
    LINUX_PRO_CONTROLLER_MAP,
    LINUX_XBOX_MAP,
    MIRA_READY_POSE,
    STICK_DEADZONE,
    TOOL_HOME_TIMEOUT,
    WINDOWS_PRO_CONTROLLER_MAP,
    WINDOWS_XBOX_MAP,
    ControlMode,
)


class GamepadControllerOp(Operator):
    def __init__(self, fragment, *args, control_mode: ControlMode = ControlMode.POLAR, **kwargs):
        self.control_mode = control_mode
        self.reset_grips = False
        self.west_east_pressed_time: Any = None

        # Button states
        self.was_west_pressed = False
        self.was_east_pressed = False
        self.was_select_pressed = False
        self.was_start_pressed = False

        # Gripper states
        self.left_gripper_open = False
        self.right_gripper_open = False

        self.q: queue.Queue = queue.Queue()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def start(self):
        self.rpc("set_lightring", LIGHT_RING_SOLID_RED.to_dict())

    def compute(self, op_input, op_output, context):
        if not self.q.empty():
            msg = self.q.get()
            op_output.emit(msg, "output")
            return

        event = op_input.receive("input")
        self.on_event(event)

    def rpc(self, command: str, params: Dict[str, Any] | Any = None):
        msg = {"jsonrpc": "2.0", "method": command, "params": params, "id": 0}
        self.q.put(msg)

    def on_event(self, event: GamepadEvent) -> None:
        if event.type == pygame.JOYDEVICEADDED:
            self.rpc("set_lightring", self.control_mode.lightring_state().to_dict())
            return
        if event.type == pygame.JOYDEVICEREMOVED:
            self.rpc("set_lightring", LIGHT_RING_SOLID_RED.to_dict())
            return

        is_xbox = "xbox" in event.name.lower()
        if event.os == "posix":
            map = LINUX_XBOX_MAP if is_xbox else LINUX_PRO_CONTROLLER_MAP
        else:
            map = WINDOWS_XBOX_MAP if is_xbox else WINDOWS_PRO_CONTROLLER_MAP

        # extend buttons to 16 (for mapping)
        buttons = list(event.buttons)
        buttons.extend([False] * (16 - len(buttons)))
        event.buttons = buttons

        # Process button presses
        # north_pressed = map["North"](event)
        west_pressed = map["West"](event)
        east_pressed = map["East"](event)
        # south_pressed = map["South"](event)

        select_pressed = map["Select"](event)
        start_pressed = map["Start"](event)

        # Cycle through control modes when select is pressed
        if select_pressed and not self.was_select_pressed:
            self.control_mode = ControlMode.CARTESIAN if self.control_mode == ControlMode.POLAR else ControlMode.POLAR
            self.rpc("set_lightring", self.control_mode.lightring_state().to_dict())

        # Toggle left gripper when west is released
        if not west_pressed and self.was_west_pressed:
            self.rpc("set_left_gripper", 0 if self.left_gripper_open else 1)
            self.left_gripper_open = not self.left_gripper_open

        # Toggle right gripper when east is released
        if not east_pressed and self.was_east_pressed:
            self.rpc("set_right_gripper", 0 if self.right_gripper_open else 1)
            self.right_gripper_open = not self.right_gripper_open

        # Handle long press for tool homing
        if west_pressed and east_pressed:
            current_time = time.time()
            if not self.west_east_pressed_time:
                self.west_east_pressed_time = current_time
            elif not self.reset_grips and (current_time - self.west_east_pressed_time > TOOL_HOME_TIMEOUT):
                print("Requesting tool homing")
                self.left_gripper_open = False
                self.right_gripper_open = False
                self.rpc("request_home_tools")
                self.reset_grips = True
        else:
            self.reset_grips = False
            self.west_east_pressed_time = None

        # Process movement controls based on control mode
        if start_pressed and not self.was_start_pressed:
            print("Resetting pose")
            self.rpc("set_mira_pose", params=MIRA_READY_POSE)
        else:
            self._process_movement_controls(map, event)

        # Update button states
        self.was_select_pressed = select_pressed
        self.was_start_pressed = start_pressed
        self.was_west_pressed = west_pressed
        self.was_east_pressed = east_pressed

    def _process_movement_controls(self, map, event: GamepadEvent) -> None:
        left_x = map["Left Stick X"](event)
        left_y = map["Left Stick Y"](event)
        left_bottom_trigger = map["Left Bumper"](event)
        left_top_trigger = map["Left Trigger"](event)

        right_x = map["Right Stick X"](event)
        right_y = map["Right Stick Y"](event)
        right_bottom_trigger = map["Right Bumper"](event)
        right_top_trigger = map["Right Trigger"](event)

        # Get stick buttons
        left_stick_button = map["Left Stick"](event)
        right_stick_button = map["Right Stick"](event)

        if not (left_stick_button or right_stick_button):
            # Process based on control mode
            if self.control_mode == ControlMode.CARTESIAN:
                left_cartesian = np.zeros(4)
                right_cartesian = np.zeros(4)

                # Change left arm X-axis direction based on trigger
                if left_top_trigger and not left_bottom_trigger:
                    left_cartesian[0] = 1
                elif left_bottom_trigger and not left_top_trigger:
                    left_cartesian[0] = -1

                # Change right arm X-axis direction based on trigger
                if right_top_trigger and not right_bottom_trigger:
                    right_cartesian[0] = 1
                elif right_bottom_trigger and not right_top_trigger:
                    right_cartesian[0] = -1

                if abs(left_x) > STICK_DEADZONE or abs(left_y) > STICK_DEADZONE:
                    if left_top_trigger and left_bottom_trigger:
                        # Change left arm elbow angle if both left triggers are pressed
                        left_cartesian[3] = left_x
                    else:
                        # Change left arm X and Y axes based on stick values
                        left_cartesian[1] = -left_x
                        left_cartesian[2] = left_y

                if abs(right_x) > STICK_DEADZONE or abs(right_y) > STICK_DEADZONE:
                    if right_top_trigger and right_bottom_trigger:
                        # Change right arm elbow angle if both right triggers are pressed
                        right_cartesian[3] = -right_x
                    else:
                        # Change right arm X and Y axes based on stick values
                        right_cartesian[1] = -right_x
                        right_cartesian[2] = right_y

                # Send rpc if any changes are detected
                if left_cartesian.any() or right_cartesian.any():
                    self.rpc(
                        "set_mira_cartesian_delta",
                        params={"left": left_cartesian.tolist(), "right": right_cartesian.tolist()},
                    )
            else:  # POLAR mode
                left_polar = np.zeros(4)
                right_polar = np.zeros(4)

                # Change left arm X-axis direction based on trigger
                if left_top_trigger and not left_bottom_trigger:
                    left_polar[2] = 1
                elif left_bottom_trigger and not left_top_trigger:
                    left_polar[2] = -1

                # Change right arm X-axis direction based on trigger
                if right_top_trigger and not right_bottom_trigger:
                    right_polar[2] = 1
                elif right_bottom_trigger and not right_top_trigger:
                    right_polar[2] = -1

                if abs(left_x) > STICK_DEADZONE or abs(left_y) > STICK_DEADZONE:
                    if left_top_trigger and left_bottom_trigger:
                        # Change left arm elbow angle if both left triggers are pressed
                        left_polar[3] = left_x
                    else:
                        # Change left arm X and Y axes based on stick values
                        left_polar[1] = -left_x
                        left_polar[0] = -left_y

                if abs(right_x) > STICK_DEADZONE or abs(right_y) > STICK_DEADZONE:
                    if right_top_trigger and right_bottom_trigger:
                        # Change right arm elbow angle if both right triggers are pressed
                        right_polar[3] = -right_x
                    else:
                        # Change right arm X and Y axes based on stick values
                        right_polar[1] = -right_x
                        right_polar[0] = -right_y

                # Send rpc if any changes are detected
                if left_polar.any() or right_polar.any():
                    self.rpc(
                        "set_mira_polar_delta", params={"left": left_polar.tolist(), "right": right_polar.tolist()}
                    )
        elif left_stick_button or right_stick_button:
            left_roll, right_roll = 0, 0
            if left_stick_button:
                left_roll = -left_y * 3
            if right_stick_button:
                right_roll = right_y * 3

            self.rpc("set_mira_roll_delta", params={"left": left_roll, "right": right_roll})

        dpad_y = map["Dpad Y"](event)
        dpad_x = map["Dpad X"](event)

        if dpad_y != 0 or dpad_x != 0:
            self.rpc("set_camera_pose_delta", params={"north": dpad_y, "east": dpad_x})
