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
import numpy as np
import time
import rti.connextdds
from holoscan.core import Operator, Tensor
from common.utils import get_ntp_offset
from schemas.camera_stream import CameraStream


class CameraStreamSplitOp(Operator):
    """Operator to split a camera stream into two streams."""

    def __init__(self, fragment, for_encoder=True, *args, **kwargs):
        self.for_encoder = for_encoder
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")
        spec.output("output")
        spec.output("image")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        if isinstance(stream.data, np.ndarray):
            # For NVC encoder: move data to GPU
            camera_data = cp.asarray(stream.data)
        elif isinstance(stream.data, rti.connextdds.Uint8Seq):
            # For NVC decoder: convert DDS sequence to numpy array
            camera_data = np.reshape(stream.data, (len(stream.data),))
        else:
            camera_data = stream.data
        if not self.for_encoder:
            self.ntp_offset_time = get_ntp_offset()
            stream.postdds = (time.time() + self.ntp_offset_time)
        op_output.emit(stream, "output")
        op_output.emit({"": Tensor.as_tensor(camera_data)}, "image")
