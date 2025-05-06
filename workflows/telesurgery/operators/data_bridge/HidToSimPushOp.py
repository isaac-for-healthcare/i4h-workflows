# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Callable

from holoscan.core import Fragment, Operator, OperatorSpec


class HidToSimPushOp(Operator):
    def __init__(
        self,
        fragment: Fragment,
        hid_event_callback: Callable,
        *args,
        **kwargs,
    ):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.hid_event_callback = hid_event_callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("input")

        self.hid_event_callback(input)
