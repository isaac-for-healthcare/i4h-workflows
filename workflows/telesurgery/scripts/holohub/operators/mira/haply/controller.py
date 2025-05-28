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

import asyncio
import queue
import threading

from holohub.operators.mira.haply.controller_async import main
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class HaplyControllerOp(Operator):
    """
    Operator to interface with Haply Inverse.
    """

    def __init__(self, fragment, uri, api_host, api_port, *args, **kwargs):
        """
        Initialize the Haply operator.

        Parameters:
        - uri (str): WebSocket URI for Inverse Service 3.1.
        """
        self.uri = uri
        self.api_host = api_host
        self.api_port = api_port
        self.q: queue.Queue = queue.Queue()

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def run_in_thread(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(main(self.uri, self.api_host, self.api_port))

    def start(self):
        threading.Thread(target=self.run_in_thread).start()

    def compute(self, op_input, op_output, context):
        m = self.q.get()
        op_output.emit(m, "output")
