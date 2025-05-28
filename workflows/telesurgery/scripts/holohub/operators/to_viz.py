from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.camera_stream import CameraStream


class CameraStreamToViz(Operator):
    """A operator that takes CameraStream input and produces input for VIZ."""

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        op_output.emit({"image": stream.data}, "output")
