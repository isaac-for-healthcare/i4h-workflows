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

import unittest

import numpy as np
from holohub.operators.nvjpeg.decoder import NVJpegDecoderOp
from holohub.operators.nvjpeg.encoder import NVJpegEncoderOp
from holoscan.core import Application, Operator, OperatorSpec
from schemas.camera_stream import CameraStream


class RandomPixelOp(Operator):
    def __init__(self, fragment, width, height, *args, **kwargs):
        self.width = width
        self.height = height
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # randomly generate a 3 channel image of size width x height
        image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        bgr_image = image[..., ::-1].copy()
        camera_stream = CameraStream(data=bgr_image, width=self.width, height=self.height)
        op_output.emit(camera_stream, "output")


class AssertionOp(Operator):
    def __init__(self, fragment, assertion_callback, *args, **kwargs):
        self.assertion_callback = assertion_callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("camera")
        spec.input("stream")
        spec.input("expected")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("stream")
        expected = op_input.receive("expected")
        self.assertion_callback(stream, expected)


class TestNVJPEGOpsApplication(Application):
    def __init__(self, width, height, quality, assertion_callback):
        self.width = width
        self.height = height
        self.quality = quality
        self.assertion_callback = assertion_callback
        super().__init__()

    def compose(self):
        random_pixel_op = RandomPixelOp(self, name="random_pixel_op", width=self.width, height=self.height)
        nvjpeg_encoder_op = NVJpegEncoderOp(self, name="nvjpeg_encoder_op", skip=False, quality=self.quality)
        nvjpeg_decoder_op = NVJpegDecoderOp(self, name="nvjpeg_decoder_op", skip=False)
        assertion_op = AssertionOp(self, name="assertion_op", assertion_callback=self.assertion_callback)

        self.add_flow(random_pixel_op, nvjpeg_encoder_op, {("output", "input")})
        self.add_flow(nvjpeg_encoder_op, nvjpeg_decoder_op, {("output", "input")})
        self.add_flow(nvjpeg_decoder_op, assertion_op, {("camera", "camera"), ("output", "stream")})
        self.add_flow(random_pixel_op, assertion_op, {("output", "expected")})


class TestNVJPEGOperators(unittest.TestCase):
    def test_nvjpeg_operators(self):
        app = TestNVJPEGOpsApplication(width=100, height=100, quality=90, assertion_callback=self.assertion_callback)
        app.run()

    def assertion_callback(self, stream, expected):
        # compare two numpy arrays and ensure they are equal
        assert np.array_equal(stream.data, expected.data)
        assert stream.width == expected.width
        assert stream.height == expected.height
