"""Gamepad controller implementation."""
import json
import threading

from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from websocket import create_connection


class ApiClientOp(Operator):
    def __init__(self, fragment, host, port, *args, **kwargs):
        self.uri = f"ws://{host}:{port}"
        self.ws = None
        self.rpc_id = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def start(self):
        print(f"Connecting to: {self.uri}")
        self.ws = create_connection(self.uri)
        print(f"Connected to api server: {self.uri}")
        threading.Thread(target=self.listener).start()

    def listener(self):
        print(f"Listening to messages from webserver: {self.uri}")
        while self.ws:
            try:
                message = self.ws.recv()
                json_message = json.loads(message)
                if "error" in json_message:
                    message = json_message["error"]["message"]
                    code = json_message["error"]["code"]
                    print(f"Received error: {message} (code: {code})")
                else:
                    print(f"Received message: {json_message}")
            except Exception as e:
                print(f"Error in message listener: {e}")

    def compute(self, op_input, op_output, context):
        try:
            message = op_input.receive("input")
            message["id"] = self.rpc_id

            self.ws.send(json.dumps(message))
            self.rpc_id += 1
        except Exception as e:
            print(f"Failed to send RPC: {e}")

    def cleanup(self) -> None:
        if self.ws:
            self.ws.close()
        self.ws = None
