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


class MergeSideBySideOp(Operator):
    """
    Holoscan Operator to merge a side-by-side image in a CameraStream to a line-by-line image.
    Receives a CameraStream and an image, emits a CameraStream with merged image data.
    """
    def _merge_side_by_side(self, img):
        """
        Merge a side-by-side image (left|right) into a line-by-line (interleaved) image using CuPy.
        Input:  img (H, 2W, C) or (H, 2W) (side-by-side)
        Output: img_out (2H, W, C) or (2H, W) (line-by-line)
        """
        # Ensure input is a CuPy array
        img = cp.asarray(img)
        h, w2 = img.shape[:2]
        w = w2 // 2
        left = img[:, :w, ...]
        right = img[:, w:, ...]
        # Prepare output array
        out_shape = (h * 2, w) + img.shape[2:]
        out = cp.empty(out_shape, dtype=img.dtype)
        out[0::2] = left
        out[1::2] = right
        return out

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        camera_tensor = op_input.receive("input")[""]
        # Merge side-by-side to line-by-line
        merged = self._merge_side_by_side(camera_tensor)
        op_output.emit(merged, "output")
