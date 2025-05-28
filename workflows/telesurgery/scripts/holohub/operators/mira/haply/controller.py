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

import json
import time

import numpy as np
from common.utils import get_ntp_offset
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from scipy.spatial.transform import Rotation
from websocket import create_connection


class HaplyControllerOp(Operator):
    """
    Operator to interface with Haply Inverse.
    """

    def __init__(self, fragment, uri, *args, **kwargs):
        """
        Initialize the Haply operator.

        Parameters:
        - uri (str): WebSocket URI for Inverse Service 3.1.
        """
        self.uri = uri
        self.ntp_offset_time = get_ntp_offset()
        self.ws = None

        self.first_message = True
        self.inverse3_device_id = None
        self.force = {"x": 0, "y": 0, "z": 0}  # Forces to send to the Inverse3 device.
        self.previous_timestamp = time.time()
        self.previous_position = None
        self.previous_angles = None
        self.previous_buttons = {"a": False, "b": False, "c": False}
        self.left_arm = True
        self.sleep = True
        self.last_toggled = self.previous_timestamp
        # 1 is open, 0 is closed
        self.left_grip = 1
        self.right_grip = 1

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self.ws = create_connection(self.uri)

    def cleanup(self) -> None:
        if self.ws:
            self.ws.close()
        self.ws = None

    def compute(self, op_input, op_output, context):
        data = json.loads(self.ws.recv())
        self.on_event(data, op_output)

    def get_angles(self, quat):
        if not quat:
            return None

        x, y, z, w = quat["x"], quat["y"], quat["z"], quat["w"]
        r = Rotation.from_quat([x, y, z, w])
        angles = r.as_euler("xyz", degrees=True)
        return {"x": angles[0], "y": angles[1], "z": angles[2]}

    def on_event(self, data, op_output):
        # Get devices list from the data
        inverse3_devices = data.get("inverse3", [])
        verse_grip_devices = data.get("wireless_verse_grip", [])

        # Get the first device from the list
        inverse3_data = inverse3_devices[0] if inverse3_devices else {}
        verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}

        # Handle the first message to get device IDs and extra information
        if self.first_message:
            self.first_message = False
            if not inverse3_data:
                print("No Inverse3 device found.")
                return

            if not verse_grip_data:
                print("No Wireless Verse Grip device found.")

            # Store device ID for sending forces
            self.inverse3_device_id = inverse3_data.get("device_id")

            # Get handedness from Inverse3 device config data (only available in the first message)
            handedness = inverse3_devices[0].get("config", {}).get("handedness")
            print(f"Inverse3 device ID: {self.inverse3_device_id}, Handedness: {handedness}")

            if verse_grip_data:
                print(f'Wireless Verse Grip device ID: {verse_grip_data.get("device_id")}')

        # Extract position, velocity from Inverse3 device state
        position = inverse3_data["state"].get("cursor_position", {})
        # velocity = inverse3_data["state"].get("cursor_velocity", {})

        # Extract buttons and orientation from Wireless Verse Grip device state (or default if not found)
        buttons = verse_grip_data.get("state", {}).get("buttons", {})
        orientation = verse_grip_data.get("state", {}).get("orientation", {})
        angles = self.get_angles(orientation)

        # Must send forces to receive state updates (even if forces are 0)
        request_msg = {
            "inverse3": [
                {
                    "device_id": self.inverse3_device_id,
                    "commands": {
                        "set_cursor_force": {
                            "values": self.force,
                        },
                    },
                }
            ]
        }

        # Send the force command message to the server
        self.ws.send(json.dumps(request_msg))

        if self.previous_position is None or self.previous_angles is None:
            self.previous_position = position
            self.previous_angles = angles
            return

        # Toggle sleep mode if both b and c are pressed
        if buttons["a"] and buttons["c"] and (self.current_time - self.last_toggled > 1):
            self.sleep = not self.sleep
            self.last_toggled = self.current_time

        if self.sleep:
            return

        # Toggle control of which arm since we only have one haply device
        if buttons["a"] and (self.current_time - self.last_toggled > 1):
            self.left_arm = not self.left_arm
            self.last_toggled = self.current_time

        dx = position["x"] - self.previous_position["x"]
        dy = position["y"] - self.previous_position["y"]
        dz = position["z"] - self.previous_position["z"]

        droll = angles["x"] - self.previous_angles["x"]
        dyaw = angles["z"] - self.previous_angles["z"]
        da = 0

        scale = 1
        if buttons["b"] and (self.current_time - self.last_toggled > 0.5):
            if self.left_arm:
                self.left_grip = not self.left_grip
            else:
                self.right_grip = not self.right_grip

            self.move_gripper(op_output, self.left_arm, self.left_grip, self.right_grip)
            self.last_toggled = self.current_time

        if buttons["c"]:
            dx, dy, dz = 0, 0, 0
            if abs(droll) > 1:
                da = -1 if droll > 0 else 1
            if abs(droll) > 0.5:
                self.move_to(op_output, dx, dy, dz, da, scale, self.left_arm)

        else:
            if np.linalg.norm([dx, dy, dz]) > 0.0001 or abs(droll) > 0.5:
                self.move_to(op_output, dx, dy, dz, da, scale, self.left_arm)

            if abs(dyaw) > 0.5:
                self.move_angle(op_output, dyaw, self.left_arm)

        self.previous_position = position
        self.previous_angles = angles
        self.previuos_buttons = buttons

    def move_to(self, op_output, dx, dy, dz, da, r, left_arm):
        scale = 700 * r

        # The x and y axis should be swapped
        dy, dx, dz = -scale * dx, scale * dy, scale * dz
        da = r * da

        request = {
            "jsonrpc": "2.0",
            "method": "set_mira_cartesian_delta",
            "params": {
                "left": [dx, dy, dz, da] if left_arm else [0, 0, 0, 0],
                "right": [0, 0, 0, 0] if left_arm else [dx, dy, dz, da],
            },
            "id": 1,
        }
        op_output.emit(json.dumps(request), "output")

    def move_angle(self, op_output, droll, left_arm):
        scale = 1
        request = {
            "jsonrpc": "2.0",
            "method": "set_mira_roll_delta",
            "params": {
                "left": scale * droll if left_arm else 0,
                "right": 0 if left_arm else scale * droll,
            },
            "id": 1,
        }
        op_output.emit(json.dumps(request), "output")

    def move_gripper(self, op_output, left_arm, left_grip, right_grip):
        request = {
            "jsonrpc": "2.0",
            "method": "set_left_gripper" if left_arm else "set_right_gripper",
            "params": 1.0 * left_grip if left_arm else right_grip,
            "id": 1,
        }
        op_output.emit(json.dumps(request), "output")
