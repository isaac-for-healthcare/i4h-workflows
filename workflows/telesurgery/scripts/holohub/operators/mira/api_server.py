"""Gamepad controller implementation."""
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
