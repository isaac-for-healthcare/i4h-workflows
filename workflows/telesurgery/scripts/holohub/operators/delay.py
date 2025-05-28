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
