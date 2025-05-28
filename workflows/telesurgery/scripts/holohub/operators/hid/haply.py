from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class HaplyOp(Operator):
    """
    Operator to interface with Haply (Connects to local haply server).
    """

    def __init__(self, fragment, *args, **kwargs):
        """
        Initialize the Haply operator.
        """

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        pass

    def compute(self, op_input, op_output, context):
        pass
