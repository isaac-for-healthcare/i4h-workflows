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

import asyncio  # To run async loops
import json
import logging
import time

import numpy as np
import websockets  # Required for device communication
from scipy.spatial.transform import Rotation


async def connect(host: str, port: int):
    """Connect to the api's websocket server."""
    uri = f"ws://{host}:{port}"

    try:
        return await asyncio.wait_for(
            websockets.connect(uri, ping_interval=None, ping_timeout=None),  # Disable heartbeat  # Disable ping timeout
            timeout=1.0,
        )
    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
        logging.error(f"Failed to connect to api at {uri}")
        raise ConnectionError(f"Failed to connect to api: {e}")
    except Exception as e:
        logging.error(f"Failed to connect to api at {uri}")
        raise ConnectionError(f"Failed to connect to api: {e}")


async def move_to(websocket, dx, dy, dz, da, r, left_arm):
    scale = 700 * r

    # The x and y axis should be swapped
    dy, dx, dz = -scale * dx, scale * dy, scale * dz
    da = r * da

    if left_arm:
        params = {"left": [dx, dy, dz, da], "right": [0, 0, 0, 0]}
    else:
        params = {
            "left": [0, 0, 0, 0],
            "right": [dx, dy, dz, da],
        }

    request = {"jsonrpc": "2.0", "method": "set_mira_cartesian_delta", "params": params, "id": 1}

    await websocket.send(json.dumps(request))
    await asyncio.sleep(0.01)  # 10ms delay


async def move_angle(websocket, droll, left_arm):
    scale = 1

    # The x and y axis should be swapped
    da = scale * droll

    if left_arm:
        params = {
            "left": da,
            "right": 0,
        }
    else:
        params = {
            "left": 0,
            "right": da,
        }

    request = {"jsonrpc": "2.0", "method": "set_mira_roll_delta", "params": params, "id": 1}

    await websocket.send(json.dumps(request))
    await asyncio.sleep(0.01)  # 10ms delay


async def move_gripper(websocket, left_arm, left_grip, right_grip):
    if left_arm:
        request = {"jsonrpc": "2.0", "method": "set_left_gripper", "params": 1.0 * left_grip, "id": 1}
    else:
        request = {"jsonrpc": "2.0", "method": "set_right_gripper", "params": 1.0 * right_grip, "id": 1}

    await websocket.send(json.dumps(request))
    await asyncio.sleep(0.01)  # 10ms delay


def get_angles(quat):
    if not quat:
        return None

    x, y, z, w = quat["x"], quat["y"], quat["z"], quat["w"]
    r = Rotation.from_quat([x, y, z, w])
    angles = r.as_euler("xyz", degrees=True)
    return {"x": angles[0], "y": angles[1], "z": angles[2]}


# Main asynchronous loop
async def main(uri, api_host, api_port):
    first_message = True
    inverse3_device_id = None
    force = {"x": 0, "y": 0, "z": 0}  # Forces to send to the Inverse3 device.
    previous_timestamp = time.time()
    previous_position = None
    previous_angles = None
    # previous_buttons = {"a": False, "b": False, "c": False}
    left_arm = True
    sleep = True
    last_toggled = previous_timestamp
    # 1 is open, 0 is closed
    left_grip = 1
    right_grip = 1

    # make connection to MIRA server
    websocket = await connect(api_host, api_port)
    # websocket = await connect("10.137.144.206", "8081")

    # Haptic loop
    async with websockets.connect(uri) as ws:
        # Wait 1 second for device to warm up
        while True:
            # Receive data from the device
            current_time = time.time()
            if current_time - previous_timestamp < 0.01:
                await asyncio.sleep(0.001)
                continue

            previous_timestamp = current_time

            response = await ws.recv()
            data = json.loads(response)

            # Get devices list from the data
            inverse3_devices = data.get("inverse3", [])
            verse_grip_devices = data.get("wireless_verse_grip", [])

            # Get the first device from the list
            inverse3_data = inverse3_devices[0] if inverse3_devices else {}
            verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}

            # Handle the first message to get device IDs and extra information
            if first_message:
                first_message = False

                if not inverse3_data:
                    print("No Inverse3 device found.")
                    break
                if not verse_grip_data:
                    print("No Wireless Verse Grip device found.")

                # Store device ID for sending forces
                inverse3_device_id = inverse3_data.get("device_id")

                # Get handedness from Inverse3 device config data (only available in the first message)
                handedness = inverse3_devices[0].get("config", {}).get("handedness")

                print(f"Inverse3 device ID: {inverse3_device_id}, Handedness: {handedness}")

                if verse_grip_data:
                    print(f'Wireless Verse Grip device ID: {verse_grip_data.get("device_id")}')

            # Extract position, velocity from Inverse3 device state
            position = inverse3_data["state"].get("cursor_position", {})
            # velocity = inverse3_data["state"].get("cursor_velocity", {})

            # Extract buttons and orientation from Wireless Verse Grip device state (or default if not found)
            buttons = verse_grip_data.get("state", {}).get("buttons", {})
            orientation = verse_grip_data.get("state", {}).get("orientation", {})
            angles = get_angles(orientation)

            # Must send forces to receive state updates (even if forces are 0)
            request_msg = {
                "inverse3": [{"device_id": inverse3_device_id, "commands": {"set_cursor_force": {"values": force}}}]
            }

            # Send the force command message to the server
            await ws.send(json.dumps(request_msg))

            if previous_position is None or previous_angles is None:
                previous_position = position
                previous_angles = angles
                continue

            # Toggle sleep mode if both b and c are pressed
            if buttons.get("a") and buttons.get("c") and (current_time - last_toggled > 1):
                sleep = not sleep
                last_toggled = current_time

            if sleep:
                continue

            # Toggle control of which arm since we only have one haply device
            if buttons.get("a") and (current_time - last_toggled > 1):
                left_arm = not left_arm
                last_toggled = current_time

            dx = position["x"] - previous_position["x"]
            dy = position["y"] - previous_position["y"]
            dz = position["z"] - previous_position["z"]

            droll = angles["x"] - previous_angles["x"]
            dyaw = angles["z"] - previous_angles["z"]
            da = 0

            scale = 1

            if buttons.get("b") and (current_time - last_toggled > 0.5):
                if left_arm:
                    left_grip = not left_grip
                else:
                    right_grip = not right_grip
                await move_gripper(websocket, left_arm, left_grip, right_grip)
                last_toggled = current_time

            if buttons.get("c"):
                dx, dy, dz = 0, 0, 0
                if abs(droll) > 1:
                    da = -1 if droll > 0 else 1
                if abs(droll) > 0.5:
                    await move_to(websocket, dx, dy, dz, da, scale, left_arm)

            else:
                if np.linalg.norm([dx, dy, dz]) > 0.0001 or abs(droll) > 0.5:
                    await move_to(websocket, dx, dy, dz, da, scale, left_arm)

                if abs(dyaw) > 0.5:
                    await move_angle(websocket, dyaw, left_arm)

            previous_position = position
            previous_angles = angles
            # previuos_buttons = buttons


# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main("ws://localhost:10001", "10.137.145.163", 8081))
