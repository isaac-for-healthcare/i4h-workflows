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

import concurrent.futures
import gc
import time
import unittest

import numpy as np
from holohub.operators.camera.sim import IsaacSimToCameraStreamOp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec


class AssertionOp(Operator):
    def __init__(self, fragment, assertion_callback, *args, **kwargs):
        self.assertion_callback = assertion_callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("stream")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("stream")
        self.assertion_callback(stream)


class TestSimCameraSourceApplication(Application):
    def __init__(self, width, height, assertion_callback):
        self.width = width
        self.height = height
        self.assertion_callback = assertion_callback
        self.sim_camera_source_op = None
        super().__init__()

    def compose(self):
        print("compose")
        self.sim_camera_source_op = IsaacSimToCameraStreamOp(
            self, name="sim_camera_source_op", width=self.width, height=self.height, count=CountCondition(self, 1)
        )
        assertion_op = AssertionOp(
            self, name="assertion_op", assertion_callback=self.assertion_callback, count=CountCondition(self, 1)
        )

        self.add_flow(self.sim_camera_source_op, assertion_op, {("output", "stream")})

    def push_data(self, data):
        self.sim_camera_source_op.on_new_frame_rcvd(data, 0)


class TestSimCameraSource(unittest.TestCase):
    def setUp(self):
        self.assertion_complete = False
        self.app = TestSimCameraSourceApplication(width=100, height=100, assertion_callback=self.assertion_callback)
        self.future = self.app.run_async()
        while self.app.sim_camera_source_op is None:
            print("Waiting for sim_camera_source_op to be initialized")
            time.sleep(0.1)

    def tearDown(self):
        # Wait for the application to complete naturally
        try:
            # Since both operators have CountCondition(1), the app should finish quickly
            self.future.result(timeout=5)  # Wait up to 5 seconds for completion
        except concurrent.futures.TimeoutError:
            self.future.cancel()
            time.sleep(0.1)  # Give it a moment to clean up
        except Exception as e:
            print(f"Exception during tearDown: {e}")

        gc.collect()

    def test_sim_camera_source(self):
        self.width = 100
        self.height = 100
        self.expected_data = np.random.randint(0, 255, (self.height, self.width, 4), dtype=np.uint8)
        self.app.push_data(self.expected_data)
        while not self.assertion_complete:
            time.sleep(0.1)

    def assertion_callback(self, stream):
        assert np.array_equal(stream.data, self.expected_data)
        assert stream.width == self.width
        assert stream.height == self.height
        self.assertion_complete = True
