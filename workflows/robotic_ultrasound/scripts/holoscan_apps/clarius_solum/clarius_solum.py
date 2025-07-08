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

from holoscan_i4h.operators.clarius_solum.clarius_solum import ClariusSolumOp
from holoscan_i4h.operators.no_op.no_op import NoOp


class ClariusSolumApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:
    - ClariusSolumOp
    - HolovizOp

    The ClariusSolumOp reads a video file and sends the frames to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def __init__(self, ip, port, cert, model, app, domain_id, height, width, topic_out, show_holoviz):
        """Initializes the ClariusSolumApp application.

        Parameters:
            ip: IP address of the Clarius probe.
            port: Port number for Clarius probe.
            cert: Path to the probe certificate.
            model: The Clarius probe model name.
            app: The ultrasound application to perform.
            domain_id: Domain ID for DDS communication.
            height: Height of the image in pixels.
            width: Width of the image in pixels.
            topic_out: The DDS topic to publish ultrasound data.
            show_holoviz: Flag to enable visualization.
        """
        self.ip = ip
        self.port = port
        self.cert = cert
        self.model = model
        self.app = app
        self.domain_id = domain_id
        self.height = height
        self.width = width
        self.topic_out = topic_out
        self.show_holoviz = show_holoviz
        super().__init__()

    def compose(self):
        """Compose the operators and define the workflow."""
        clarius_solum = ClariusSolumOp(
            self,
            name="clarius_solum",
            ip=self.ip,
            port=self.port,
            cert=self.cert,
            model=self.model,
            app=self.app,
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
            window_title="Clarius Solum",
            width=self.width,
            height=self.height,
        )
        noop = NoOp(self)

        # Define the workflow
        if self.show_holoviz:
            self.add_flow(clarius_solum, holoviz, {("output", "receivers")})
        else:
            self.add_flow(clarius_solum, noop)


def main():
    """Parse command-line arguments and run the application."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the Clarius Solum application")
    parser.add_argument("--test", action="store_true", help="show holoviz")
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.68.50",
        help="IP address of Clarius probe",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="port # for Clarius probe",
    )
    parser.add_argument(
        "--cert",
        type=str,
        default=os.environ.get("CLARIUS_CERTIFICATE", f"{cwd}/ClariusOne.cert"),
        help="The required certificate to use Clarius Solum APIs",
    )
    # Only support C3HD3 for now
    parser.add_argument(
        "--model",
        choices=["C3HD3"],
        default="C3HD3",
        help="The model of the Clarius Probe",
    )
    parser.add_argument(
        "--app",
        choices=[
            "abdomen",
            "bladder",
            "cardiac",
            "lung",
            "msk",
            "msk_hip",
            "msk_shoulder",
            "msk_spine",
            "nerve",
            "obgyn",
            "oncoustics_liver",
            "pelvic",
            "prostate",
            "research",
            "superficial",
            "vascular",
        ],
        default="abdomen",
        help="The Ultrasound Application to run",
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

    app = ClariusSolumApp(
        args.ip,
        args.port,
        args.cert,
        args.model,
        args.app,
        args.domain_id,
        args.height,
        args.width,
        args.topic_out,
        args.test,
    )
    app.run()


if __name__ == "__main__":
    main()
