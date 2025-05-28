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

from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class DelayOp(Operator):
    """A delay operator that takes input and waits for few milliseconds."""

    def __init__(self, fragment, deplay_ms, *args, **kwargs):
        self.deplay_ms = deplay_ms
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("input")
        time.sleep(self.deplay_ms / 1000.0)
        op_output.emit(input, "output")
