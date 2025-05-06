# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
from dds_hid_subscriber._dds_hid_subscriber import HIDDeviceType


class HIDController:
    @property
    def controller_data(self):
        return self._controller_data

    @property
    def default_joint_positions(self):
        return self._controller_data._default_joint_positions

    @property
    def target_joint_positions(self):
        return self._controller_data._target_joint_positions

    @property
    def joint_names(self):
        return self._controller_data.joint_name_to_index.keys()

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._controller_data = ControllerData()
        self.deadzone = 0.05  # Define deadzone threshold

    def forward(self, dt: float):
        """Move the robot to the desired joint positions based on velocity * dt.

        Args:
            current_joint_positions: Current joint positions of the robot
            dt: Simulation time step
        """

        # Speed factor now represents velocity (radians per second)
        speed_factor = 1.0  # Adjust this value as needed
        joint_positions = self._controller_data._target_joint_positions.copy()

        # Apply continuous movement based on joystick velocity inputs
        for joint_name, value in self._controller_data.joystick_values.items():
            # Use the defined deadzone from __init__
            if abs(value) > self.deadzone:
                joint_index = self._controller_data.joint_name_to_index[joint_name]
                # Calculate position change: velocity * time
                joint_positions[joint_index] += value * speed_factor * dt

        # Apply controller data joint positions using vectorized clamping
        # Clamp the controller's joint positions based on the defined limits
        self._controller_data._target_joint_positions = np.clip(
            joint_positions,
            a_min=self._controller_data.joint_limits_lower,
            a_max=self._controller_data.joint_limits_upper,
        )

        return self._controller_data._target_joint_positions

    def reset_to_default_joint_positions(self):
        self._controller_data._target_joint_positions = self._controller_data._default_joint_positions.copy()

    def handle_hid_event(self, events: str):
        if events is None or len(events) == 0:
            return

        for event in events:
            event_handled = False
            match event.device_type:
                case HIDDeviceType.JOYSTICK:
                    match event.event_type:
                        case 1:  # buttons
                            match event.number:
                                case 5:  # left - controls wrist_3 (negative direction)
                                    new_value = -1.0 if event.value != 0 else 0.0
                                    self._controller_data.set_joystick_value("wrist_3", new_value)
                                    event_handled = True
                                case 6:  # right - controls wrist_3 (positive direction)
                                    new_value = 1.0 if event.value != 0 else 0.0
                                    self._controller_data.set_joystick_value("wrist_3", new_value)
                                    event_handled = True
                                case 7 | 8 | 9:  # NV, Circle, L Triangle, R Triangle
                                    self.reset_to_default_joint_positions()
                                    event_handled = True

                                # NV button (4), Circle (9), L Triangle (7), R Triangle (8) have no function
                        case 2:  # Joysticks and D-Pad (Axes)
                            # Normalize value to [-1.0, 1.0] for axes
                            # Axis values are typically -32767 to 32767
                            normalized_value = event.value / 32767.0

                            # # Apply deadzone check
                            # if abs(normalized_value) < self.deadzone:
                            #     normalized_value = 0.0

                            match event.number:
                                case 0:  # Left Joystick X - controls shoulder_pan
                                    self._controller_data.set_joystick_value("shoulder_pan", normalized_value)
                                    event_handled = True
                                case 1:  # Left Joystick Y - controls shoulder_lift (Inverted)
                                    # Invert Y axis if necessary (common for flight stick style controls)
                                    self._controller_data.set_joystick_value("shoulder_lift", -normalized_value)
                                    event_handled = True
                                case 2:  # Right Joystick X - controls elbow
                                    self._controller_data.set_joystick_value("elbow", normalized_value)
                                    event_handled = True
                                case 5:  # Right Joystick Y - controls wrist_1 (Inverted)
                                    # Invert Y axis if necessary
                                    self._controller_data.set_joystick_value("wrist_1", -normalized_value)
                                    event_handled = True
                                case 3:  # ZR (Right Trigger) - typically axis 5 in SDL, might vary
                                    self._controller_data.set_joystick_value("wrist_2", abs(normalized_value))
                                    event_handled = True
                                case 4:  # ZL (Left Trigger) - typically axis 2 in SDL, might vary
                                    self._controller_data.set_joystick_value("wrist_2", -abs(normalized_value))
                                    event_handled = True
                                case 6 | 7:  # ZR/ZL - stop wrist_2 movement
                                    if normalized_value == -1.0:
                                        self._controller_data.set_joystick_value("wrist_2", 0.0)
                                        event_handled = True

            if not event_handled:
                self._logger.warning(f"Unhandled event: {event}")
                print(f"\tdevice type: {event.device_type} - type = {type(event.device_type)}")
                print(f"\tevent type: {event.event_type} - type = {type(event.event_type)}")
                print(f"\tevent number: {event.number} - type = {type(event.number)}")
                print(f"\tevent value: {event.value} - type = {type(event.value)}")


class ControllerData:
    def __init__(self):
        """Stores controller data for robot and joint selection and control.

        Maintains state information for robot selection, joint selection,
        and incremental joint position changes based on HID input events.
        Also tracks timing information for latency measurement.
        """

        # Define joint limits between -180 and +180 degrees (-π to +π radians)
        # Special case for shoulder_lift: -240 to 65 degrees (-4.18879 to 1.13446 radians)
        self.joint_limits_lower = np.array([-3.14159, -4.18879, -3.14159, -3.14159, -3.14159, -3.14159])
        self.joint_limits_upper = np.array([3.14159, 1.13446, 3.14159, 3.14159, 3.14159, 3.14159])

        # Joint control
        self._default_joint_positions: torch.Tensor = np.array(
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]
        )

        self._target_joint_positions: torch.Tensor = self._default_joint_positions.copy()

        # Map joint names to indices
        self.joint_name_to_index = {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow": 2,
            "wrist_1": 3,
            "wrist_2": 4,
            "wrist_3": 5,
        }

        self.joystick_values = {
            "shoulder_pan": 0.0,  # Left Joystick X
            "shoulder_lift": 0.0,  # Left Joystick Y
            "elbow": 0.0,  # Right Joystick X
            "wrist_1": 0.0,  # Right Joystick Y
            "wrist_2": 0.0,  # ZR/ZL
            "wrist_3": 0.0,  # Left/Right buttons
        }

        # store the latest input command received for performance measurement
        # ensure thread safe access to the latest message
        self.latest_input_command = None
        self.unique_message_ids = set()
        self.last_message_id = 0
        self.frame_id = 0
        self.message_loss_count = 0

    def set_joystick_value(self, joint_name: str, value: float):
        """Set the current value for a joystick axis.

        Args:
            joint_name: Name of the joint controlled by this joystick axis
            value: Normalized joystick value between -1.0 and 1.0
        """
        self.joystick_values[joint_name] = value

    def debug_print(self, message):
        print("+===================+")
        print(message)
        print("+===================+")
