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

import cupy as cp
from common.utils import get_ntp_offset
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.camera_stream import CameraStream


# Convert input video frame to CameraStream
class VideoToCameraStreamOp(Operator):
    def __init__(self, fragment, width, height, *args, **kwargs):
        self.width = width
        self.height = height
        self.ntp_offset_time = get_ntp_offset()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        data = cp.asarray(op_input.receive("input")[""])

        ts = int((time.time() + self.ntp_offset_time) * 1000)

        stream = CameraStream(
            ts=ts,
            type=2,
            format=7,
            width=self.width,
            height=self.height,
            data=data,
        )
        op_output.emit(stream, "output")
