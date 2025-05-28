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

import holoscan
import numpy as np
from holoscan.core import Application, Operator
from holoscan.core._core import OperatorSpec
from holoscan.operators.holoviz import HolovizOp
from holoscan.resources import UnboundedAllocator
from holoscan.conditions import CountCondition
from operators.realsense_camera_dds.realsense_camera_dds import RealsenseCameraDDSOp

class NoOp(Operator):
    """A sink operator that takes input and discards them."""

    def __init__(self, fragment, depth, *args, **kwargs):
        """Initialize the operator."""
        self.depth = depth
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Define input ports for color and optional depth."""
        spec.input("color")
        if self.depth:
            spec.input("depth")

    def compute(self, op_input, op_output, context):
        """Receive and discard input frames."""
        op_input.receive("color")
        if self.depth:
            op_input.receive("depth")


class RealsenseApp(Application):
    """Application to run the RealSense operator and process its output."""

    def __init__(self, domain_id, height, width, topic_rgb, topic_depth, device_idx, framerate, show_holoviz, count):
        self.domain_id = domain_id
        self.height = height
        self.width = width
        self.topic_rgb = topic_rgb
        self.topic_depth = topic_depth
        self.device_idx = device_idx
        self.framerate = framerate
        self.show_holoviz = show_holoviz
        self.count = count
        super().__init__()

    def compose(self):
        """Create and connect application operators."""
        camera = RealsenseCameraDDSOp(
            self,
            CountCondition(self, self.count),
            name="realsense",
            domain_id=self.domain_id,
            height=self.height,
            width=self.width,
            topic_rgb=self.topic_rgb,
            topic_depth=self.topic_depth,
            device_idx=self.device_idx,
            framerate=self.framerate,
            show_holoviz=self.show_holoviz,
        )

        if self.show_holoviz:
            holoviz = HolovizOp(
                self,
                allocator=UnboundedAllocator(self, name="pool"),
                name="holoviz",
                window_title="Realsense Camera",
                width=self.width,
                height=self.height,
            )
            self.add_flow(camera, holoviz, {("color", "receivers")})
        else:
            port_map = {("color", "color"), ("depth", "depth")} if self.topic_depth else {("color", "color")}
            self.add_flow(camera, NoOp(self, True if self.topic_depth else None), port_map)


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the RealSense camera application")
    parser.add_argument("--test", action="store_true", help="show holoviz")
    parser.add_argument(
        "--count",
        type=int,
        default=-1,
        help="Number of frames to run",
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
        "--framerate",
        type=int,
        default=30,
        help="frame rate",
    )
    parser.add_argument(
        "--device_idx",
        type=int,
        default=None,
        help="device index case of multiple cameras",
    )
    parser.add_argument(
        "--topic_rgb",
        type=str,
        default="topic_room_camera_data_rgb",
        help="topic name to produce camera rgb",
    )
    parser.add_argument(
        "--topic_depth",
        type=str,
        default=None,  # "topic_room_camera_data_depth",
        help="topic name to produce camera depth",
    )

    args = parser.parse_args()
    app = RealsenseApp(
        args.domain_id,
        args.height,
        args.width,
        args.topic_rgb,
        args.topic_depth,
        args.device_idx,
        args.framerate,
        args.test,
        args.count,
    )
    app.run()


if __name__ == "__main__":
    main()
