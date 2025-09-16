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


class MergeSideBySideOp(Operator):
    """
    Holoscan Operator to merge an up-down image in a CameraStream to a line-by-line image.
    The top half of the image is for the left eye and the bottom half is the right eye.
    """

    def _merge_side_by_side(self, img):
        """
        Merge an up-down image (up|down) into a line-by-line (interleaved) image using CuPy.
        Input:  img (2H, W, C) or (2H, W) (up-down)
        Output: img_out (H, 2W, C) or (H, 2W) (line-by-line)
        """
        # Ensure input is a CuPy array
        img = cp.asarray(img)
        h2, w = img.shape[:2]
        if h2 % 2 != 0:
            raise ValueError("Input image height must be even for up-down splitting.")
        h = h2 // 2
        up = img[:h, :, ...]
        down = img[h:, :, ...]
        # Prepare output array
        out_shape = (h * 2, w) + img.shape[2:]
        out = cp.empty(out_shape, dtype=img.dtype)
        out[0::2] = up
        out[1::2] = down
        return out

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        camera_tensor = op_input.receive("input")[""]
        # Merge side-by-side to line-by-line
        merged = self._merge_side_by_side(camera_tensor)
        op_output.emit(merged, "output")
