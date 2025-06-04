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
import os

from holohub.operators.mira.haply.controller import HaplyControllerOp
from holohub.operators.sink import NoOp
from holoscan.core import Application


class App(Application):
    """Application to run the HID operator and process its input."""

    def __init__(self, uri, api_host, api_port):
        self.uri = uri
        self.api_host = api_host
        self.api_port = api_port

        super().__init__()

    def compose(self):
        haply = HaplyControllerOp(
            self,
            name="haply_controller",
            uri=self.uri,
            api_host=self.api_host,
            api_port=self.api_port,
        )
        sink = NoOp(self)

        self.add_flow(haply, sink, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the hid application")
    parser.add_argument("--uri", type=str, default="ws://localhost:10001", help="haply inverse local uri")
    parser.add_argument("--api_host", type=str, default=os.environ.get("PATIENT_IP"), help="api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="api server port")

    args = parser.parse_args()

    app = App(
        uri=args.uri,
        api_host=args.api_host,
        api_port=args.api_port,
    )
    app.run()


if __name__ == "__main__":
    main()
