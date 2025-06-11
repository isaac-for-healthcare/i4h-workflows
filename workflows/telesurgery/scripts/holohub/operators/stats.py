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
from common.utils import get_ntp_offset
from holoscan.core import Operator, OperatorSpec
from schemas.camera_stream import CameraStream


class CameraStreamStats(Operator):
    """A operator that takes CameraStream input and produces stats."""

    def __init__(self, fragment, interval_ms, *args, **kwargs):
        self.prev_ts = 0
        self.interval_ms = interval_ms

        self.sync_offset_time = get_ntp_offset()
        self.prev_ts = 0
        self.time_diff = []
        self.time_encode = []
        self.time_decode = []
        self.compress_ratio = []
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        ts = int((time.time() + self.sync_offset_time) * 1000)
        self.time_diff.append(ts - stream.ts)
        self.time_encode.append(stream.encode_latency)
        self.time_decode.append(stream.decode_latency)
        self.compress_ratio.append(stream.compress_ratio)

        if ts - self.prev_ts > self.interval_ms:
            f1 = len(self.time_diff)
            l1 = np.min(self.time_diff)
            l2 = np.max(self.time_diff)
            l3 = round(np.mean(self.time_diff))

            e1 = np.mean(self.time_encode)
            d1 = np.mean(self.time_decode)
            c1 = np.mean(self.compress_ratio)
            s1 = stream.width * stream.height * (4 if stream.type == 2 else 1)

            # Print the results
            print(
                f"fps: {f1:02d}, "
                f"min: {l1:02d}, max: {l2:03d}, avg: {l3:03d}, "
                f"encode: {e1:02.2f}, decode: {d1:02.2f}, "
                f"compress: {c1:03.1f}x, size: {s1:,}"
            )

            self.time_diff.clear()
            self.time_encode.clear()
            self.time_decode.clear()
            self.compress_ratio.clear()
            self.prev_ts = ts
