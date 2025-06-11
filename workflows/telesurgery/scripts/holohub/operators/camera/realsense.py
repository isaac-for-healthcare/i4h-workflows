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

import time

import numpy as np
import pyrealsense2 as rs
from common.utils import get_ntp_offset
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.camera_stream import CameraStream


class RealsenseToCameraStreamOp(Operator):
    """
    Operator to interface with an Intel RealSense camera.
    Captures RGB or depth frames.
    """

    def __init__(
        self,
        fragment,
        width: int,
        height: int,
        device_idx: int,
        framerate: int,
        stream_type: str,
        stream_format: str,
        *args,
        **kwargs,
    ):
        """
        Initialize the RealSense operator.

        Parameters:
        - width (int): Width of the camera stream.
        - height (int): Height of the camera stream.
        - device_idx (int): Camera device index.
        - framerate (int): Frame rate for the camera stream.
        - stream_type (str): Stream Type (color|depth).
        - stream_format (str): Stream format [pyrealsense2.format].
        """
        self.width = width
        self.height = height
        self.device_idx = device_idx
        self.framerate = framerate

        self.stream_type = rs.stream.depth if stream_type == "depth" else rs.stream.color
        self.stream_format = (
            rs.format.__members__[stream_format]
            if stream_format
            else rs.format.z16
            if stream_type == "depth"
            else rs.format.rgba8
        )
        self.ntp_offset_time = get_ntp_offset()

        super().__init__(fragment, *args, **kwargs)
        self.pipeline = rs.pipeline()

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def stop(self):
        self.pipeline.stop()

    def start(self):
        config = rs.config()
        context = rs.context()

        try:
            for i, device in enumerate(context.query_devices()):
                try:
                    print(f"(RealSense): {i}: Available device: {device}")
                except Exception:
                    print(f"(RealSense): Failed to query device {i} (Ignoring)")
        except Exception:
            print("(RealSense): FAILED TO QUERY DEVICES (Ignoring)")

        if self.device_idx is not None:
            if self.device_idx < len(context.query_devices()):
                config.enable_device(context.query_devices()[self.device_idx].get_info(rs.camera_info.serial_number))
            else:
                print(f"(RealSense): Ignoring input device_idx: {self.device_idx}")

        print(
            f"(RealSense): Enable type: {self.stream_type}; format: {self.stream_format}; "
            f"resolution: {self.width}x{self.height}; framerate: {self.framerate}"
        )
        config.enable_stream(
            stream_type=self.stream_type,
            format=self.stream_format,
            width=self.width,
            height=self.height,
            framerate=self.framerate,
        )
        self.pipeline.start(config)

    def compute(self, op_input, op_output, context):
        frames = self.pipeline.wait_for_frames()

        ts = int((time.time() + self.ntp_offset_time) * 1000)
        if self.stream_type == rs.stream.depth:
            data = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
        else:
            data = np.asanyarray(frames.get_color_frame().get_data())

        stream = CameraStream(
            ts=ts,
            type=self.stream_type.value,
            format=self.stream_format.value,
            width=self.width,
            height=self.height,
            data=data,
        )
        op_output.emit(stream, "output")
