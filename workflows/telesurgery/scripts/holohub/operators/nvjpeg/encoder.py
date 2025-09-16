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

import cupy
import numpy as np
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from nvjpeg import NvJpeg
from schemas.camera_stream import CameraStream


class NVJpegEncoderOp(Operator):
    """
    Operator to encode RGB data to JPEG using NVJpeg.
    """

    def __init__(self, fragment, skip, quality, *args, **kwargs):
        """
        Initialize the Jpeg Encode operator.

        Parameters:
        - jpeg_quality (int): Quality of the JPEG compression (between 1 and 100).
        """
        self.skip = skip
        self.quality = max(1, min(100, quality))
        self.nvjpeg = None

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def start(self):
        self.nvjpeg = NvJpeg()
        print(f"NVJpegEncoder: enabled {not self.skip}; quality: {self.quality}")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        start = time.time()
        if isinstance(stream.data, cupy.ndarray):
            stream.data = cupy.asnumpy(stream.data)

        if self.skip:
            stream.data = stream.data.tobytes() if isinstance(stream.data, np.ndarray) else stream.data
            stream.compress_ratio = 1
        else:
            original_len = np.prod(stream.data.shape) if isinstance(stream.data, np.ndarray) else len(stream.data)
            stream.data = self.nvjpeg.encode(stream.data[:, :, :3], self.quality)
            stream.compress_ratio = original_len / len(stream.data)
        stream.encode_latency = (time.time() - start) * 1000

        op_output.emit(stream, "output")
