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

import cupy as cp
from holoscan.core import Operator
from schemas.camera_stream import CameraStream


class CameraStreamMergeOp(Operator):
    """Operator to merge a camera stream into a single stream."""

    def __init__(self, fragment, for_encoder=True, *args, **kwargs):
        self.for_encoder = for_encoder

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")
        spec.input("image")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        camera_tensor = op_input.receive("image")[""]
        assert isinstance(stream, CameraStream)

        if self.for_encoder:
            stream.data = cp.asarray(camera_tensor).get().tobytes()
            stream.encode_latency = self.metadata.get("video_encoder_encode_latency_ms", 0)
            stream.compress_ratio = self.metadata.get("video_encoder_compress_ratio", 0)
        else:
            stream.decode_latency = self.metadata.get("video_decoder_decode_latency_ms", 0)
        op_output.emit(stream, "output")
