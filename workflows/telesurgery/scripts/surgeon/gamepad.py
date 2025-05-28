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

import argparse

from holohub.operators.hid.gamepad import GamepadOp
from holohub.operators.mira.api_client import ApiClientOp
from holohub.operators.mira.gamepad.controller import GamepadControllerOp
from holoscan.core import Application


class App(Application):
    """Application to run the HID operator and process its input."""

    def __init__(self, device_idx, api_host, api_port):
        self.device_idx = device_idx
        self.api_host = api_host
        self.api_port = api_port

        super().__init__()

    def compose(self):
        event = GamepadOp(
            self,
            name="gamepad",
            device_idx=self.device_idx,
        )
        controller = GamepadControllerOp(
            self,
            name="gamepad_controller",
        )
        client = ApiClientOp(
            self,
            name="api_client",
            host=self.api_host,
            port=self.api_port,
        )

        self.add_flow(event, controller, {("output", "input")})
        self.add_flow(controller, client, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the hid application")
    parser.add_argument("--device_idx", type=int, default=0, help="device index case of multiple joysticks")
    parser.add_argument("--api_host", type=str, default="10.137.145.163", help="api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="api server port")

    args = parser.parse_args()

    app = App(
        device_idx=args.device_idx,
        api_host=args.api_host,
        api_port=args.api_port,
    )
    app.run()


if __name__ == "__main__":
    main()
