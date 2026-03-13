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
from holoscan.core import Operator, OperatorSpec
from schemas.camera_stream import CameraStream


class StreamliftStatsOp(Operator):
    def __init__(self, fragment, upsampling, interval_ms=1000, *args, **kwargs):
        self.prev_ts = 0
        self.interval_ms = interval_ms
        self.stype = "up" if upsampling else "down"
        self.metric_keys = [
            f"streamlift_{self.stype}_total_latency_ms",
            f"streamlift_{self.stype}_h2d_latency_ms",
            f"streamlift_{self.stype}_sampling_latency_ms",
            f"streamlift_{self.stype}_d2h_latency_ms",
        ]

        self.metrics = {}
        for metric_key in self.metric_keys:
            self.metrics[metric_key] = []
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")

        ts = int(time.time() * 1000)
        for metric_key in self.metric_keys:
            self.metrics[metric_key].append(self.metadata.get(metric_key, 0))

        if ts - self.prev_ts > self.interval_ms:
            fps = len(self.metrics.get(self.metric_keys[0]))

            v = []
            for metric_key in self.metric_keys:
                min = np.min(self.metrics[metric_key])
                max = np.max(self.metrics[metric_key])
                avg = np.mean(self.metrics[metric_key])
                v.append([min, max, avg])

            # Print the results
            print(
                f"StreamLift{self.stype.capitalize()}Op:: fps: {fps:02d}, "
                f"min: {v[0][0]:.2f}, max: {v[0][1]:.2f}, avg: {v[0][2]:.2f}, "
                f"hmin: {v[1][0]:.2f}, hmax: {v[1][1]:.2f}, havg: {v[1][2]:.2f}, "
                f"smin: {v[2][0]:.2f}, smax: {v[2][1]:.2f}, savg: {v[2][2]:.2f}, "
                f"dmin: {v[3][0]:.2f}, dmax: {v[3][1]:.2f}, davg: {v[3][2]:.2f}"
            )

            for metric_key in self.metric_keys:
                self.metrics[metric_key].clear()
            self.prev_ts = ts


class CameraStreamUpdateDimOp(Operator):
    """Operator to update a camera width/height."""

    def __init__(self, fragment, width, height, *args, **kwargs):
        self.width = width
        self.height = height

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        stream.width = self.width
        stream.height = self.height
        op_output.emit(stream, "output")
