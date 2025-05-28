from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class NoOp(Operator):
    """A sink operator that takes input and discards them."""

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        op_input.receive("input")
