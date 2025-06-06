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

import numpy as np
import pyrealsense2 as rs
import holoscan
import rti.connextdds as dds
from dds.schemas.camera_info import CameraInfo
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class RealsenseOp(Operator):
    """
    Operator to interface with an Intel RealSense camera.
    Captures RGB and depth frames, optionally publishing them via DDS.
    """

    def __init__(
        self,
        fragment,
        *args,
        domain_id,
        width,
        height,
        topic_rgb,
        topic_depth,
        device_idx,
        framerate,
        show_holoviz,
        **kwargs,
    ):
        """
        Initialize the RealSense operator.

        Parameters:
        - domain_id (int): DDS domain ID.
        - width (int): Width of the camera stream.
        - height (int): Height of the camera stream.
        - topic_rgb (str): DDS topic for RGB frames.
        - topic_depth (str): DDS topic for depth frames.
        - device_idx (int): Camera device index.
        - framerate (int): Frame rate for the camera stream.
        - show_holoviz (bool): Whether to display frames using Holoviz.
        """
        self.domain_id = domain_id
        self.width = width
        self.height = height
        self.topic_rgb = topic_rgb
        self.topic_depth = topic_depth
        self.device_idx = device_idx
        self.framerate = framerate
        self.show_holoviz = show_holoviz
        super().__init__(fragment, *args, **kwargs)

        self.pipeline = rs.pipeline()
        self.rgb_writer = None
        self.depth_writer = None

    def setup(self, spec: OperatorSpec):
        """Define the output ports for the operator."""
        if self.topic_rgb:
            spec.output("color")
        if self.topic_depth and not self.show_holoviz:
            spec.output("depth")

    def start(self):
        """
        Configure and start the RealSense camera pipeline.
        Sets up DDS writers if topics are provided.
        """
        config = rs.config()
        context = rs.context()
        for device in context.query_devices():
            print(f"+++ Available device: {device}")

        if self.device_idx is not None:
            if self.device_idx < len(context.query_devices()):
                config.enable_device(context.query_devices()[self.device_idx].get_info(rs.camera_info.serial_number))
            else:
                print(f"Ignoring input device_idx: {self.device_idx}")

        dp = dds.DomainParticipant(domain_id=self.domain_id)
        if self.topic_rgb:
            print("Enabling color...")
            self.rgb_writer = dds.DataWriter(dp.implicit_publisher, dds.Topic(dp, self.topic_rgb, CameraInfo))
            config.enable_stream(
                rs.stream.color,
                width=self.width,
                height=self.height,
                format=rs.format.rgba8,
                framerate=self.framerate,
            )
        if self.topic_depth:
            print("Enabling depth...")
            config.enable_stream(
                rs.stream.depth,
                width=self.width,
                height=self.height,
                format=rs.format.z16,
                framerate=self.framerate,
            )
            self.depth_writer = dds.DataWriter(dp.implicit_publisher, dds.Topic(dp, self.topic_depth, CameraInfo))
        self.pipeline.start(config)

    def compute(self, op_input, op_output, context):
        """
        Capture frames from the RealSense camera and publish them via DDS.
        """
        frames = self.pipeline.wait_for_frames()
        color = None
        if self.rgb_writer:
            color = np.asanyarray(frames.get_color_frame().get_data())
            self.rgb_writer.write(CameraInfo(data=color.tobytes(), width=self.width, height=self.height))
            op_output.emit({"color": holoscan.as_tensor(color)} if self.show_holoviz else True, "color")
        if self.depth_writer:
            depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) / 1000.0
            self.depth_writer.write(CameraInfo(data=depth.tobytes(), width=self.width, height=self.height))
            if not self.show_holoviz:
                op_output.emit({"depth": holoscan.as_tensor(depth)}, "depth")


