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

import queue
import time

from common.utils import get_ntp_offset
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.camera_stream import CameraStream


class IsaacSimToCameraStreamOp(Operator):
    """
    Operator to capture video using OpenCV.
    Process RGB frames from Isaac SIM.
    """

    def __init__(self, fragment, width: int, height: int, *args, **kwargs):
        """
        Initialize the RealSense operator.

        Parameters:
        - width (int): Width of the camera stream.
        - height (int): Height of the camera stream.
        - stream_type (str): Stream Type (color|depth).
        - stream_format (str): Stream format [pyrealsense2.format].
        """
        self.width = width
        self.height = height
        self.stream_type = 2  # color
        self.stream_format = 7  # rgba
        self.ntp_offset_time = get_ntp_offset()

        self.q: queue.Queue = queue.Queue()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        pass

    def stop(self):
        self.q.empty()

    def on_new_frame_rcvd(self, data, frame_num):
        ts = int((time.time() + self.ntp_offset_time) * 1000)
        stream = CameraStream(
            ts=ts,
            type=self.stream_type,
            format=self.stream_format,
            width=self.width,
            height=self.height,
            data=data,
            frame_num=frame_num,
        )
        self.q.put(stream)

    def compute(self, op_input, op_output, context):
        stream = self.q.get()
        op_output.emit(stream, "output")
