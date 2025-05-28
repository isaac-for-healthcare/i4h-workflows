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

import json
import queue
import threading

from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from websocket_server import WebsocketServer


class ApiServerOp(Operator):
    def __init__(self, fragment, host, port, *args, **kwargs):
        self.host = host
        self.port = port
        self.server = None
        self.q: queue.Queue = queue.Queue()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def on_message_received(self, client, server, message):
        self.q.put(json.loads(message))

    def start(self):
        self.server = WebsocketServer(host=self.host, port=self.port)
        self.server.set_fn_message_received(self.on_message_received)
        threading.Thread(target=self.listener).start()

    def listener(self):
        self.server.run_forever()

    def compute(self, op_input, op_output, context):
        m = self.q.get()
        op_output.emit(m, "output")

    def cleanup(self) -> None:
        self.server.shutdown()
