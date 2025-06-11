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

from holoscan.core import Operator, Tensor
from holoscan.core._core import OperatorSpec
from nvjpeg import NvJpeg
from schemas.camera_stream import CameraStream


class NVJpegDecoderOp(Operator):
    """
    Operator to decode RGB data to JPEG using NVJpeg.
    """

    def __init__(self, fragment, skip, *args, **kwargs):
        self.skip = skip
        self.nvjpeg = None

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")
        spec.output("image")

    def start(self):
        self.nvjpeg = NvJpeg()

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        start = time.time()
        if not self.skip:
            stream.data = self.nvjpeg.decode(stream.data)
        stream.decode_latency = (time.time() - start) * 1000

        op_output.emit(stream, "output")
        op_output.emit({"": Tensor.as_tensor(stream.data)}, "image")
