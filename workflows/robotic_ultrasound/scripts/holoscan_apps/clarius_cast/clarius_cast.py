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

from holoscan.core import Application
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator

from holoscan_i4h.operators.clarius_cast.clarius_cast import ClariusCastOp
from holoscan_i4h.operators.no_op.no_op import NoOp


class ClariusCastApp(Application):
    """Application for streaming Ultrasound image data using Clarius Cast APIs"""

    def __init__(self, ip, port, domain_id, height, width, topic_out, show_holoviz):
        """
        Initializes the ClariusCastApp application.

        Parameters:
            ip: IP address of the Clarius probe.
            port: Port number for the Clarius probe.
            domain_id: DDS domain ID.
            height: Height of the image in pixels.
            width: Width of the image in pixels.
            topic_out: The DDS topic name for publishing ultrasound data.
            show_holoviz: Flag to indicate if Holoviz should be shown.
        """
        self.ip = ip
        self.port = port
        self.domain_id = domain_id
        self.height = height
        self.width = width
        self.topic_out = topic_out
        self.show_holoviz = show_holoviz
        super().__init__()

    def compose(self):
        """Compose the operators and define the workflow."""
        clarius_cast = ClariusCastOp(
            self,
            name="clarius_cast",
            ip=self.ip,
            port=self.port,
            domain_id=self.domain_id,
            height=self.height,
            width=self.width,
            topic_out=self.topic_out,
        )

        pool = UnboundedAllocator(self, name="pool")
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="Clarius Cast",
            width=self.width,
            height=self.height,
        )
        noop = NoOp(self)

        # Define the workflow
        if self.show_holoviz:
            self.add_flow(clarius_cast, holoviz, {("output", "receivers")})
        else:
            self.add_flow(clarius_cast, noop)


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the Clarius Cast application")
    parser.add_argument("--test", action="store_true", help="show holoviz")
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.1.1",
        help="IP address of Clarius probe",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5828,
        help="port # for Clarius probe",
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=int(os.environ.get("OVH_DDS_DOMAIN_ID", 1)),
        help="domain id",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("OVH_HEIGHT", 480)),
        help="height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("OVH_WIDTH", 640)),
        help="width",
    )
    parser.add_argument(
        "--topic_out",
        type=str,
        default="topic_ultrasound_data",
        help="topic name to publish generated ultrasound data",
    )
    args = parser.parse_args()

    app = ClariusCastApp(args.ip, args.port, args.domain_id, args.height, args.width, args.topic_out, args.test)
    app.run()


if __name__ == "__main__":
    main()
