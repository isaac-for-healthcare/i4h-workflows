import time

from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from nvjpeg import NvJpeg
from schemas.camera_stream import CameraStream


class NVJpegDecoderOp(Operator):
    """
    Operator to decode RGB data to JPEG using NVJpeg.
    """

    def __init__(self, fragment, skip, *args, **kwargs):
        self.skip = skip
        self.nvjpeg = None

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def start(self):
        self.nvjpeg = NvJpeg()

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, CameraStream)

        start = time.time()
        if not self.skip:
            stream.data = self.nvjpeg.decode(stream.data)
        stream.decode_latency = (time.time() - start) * 1000

        op_output.emit(stream, "output")
