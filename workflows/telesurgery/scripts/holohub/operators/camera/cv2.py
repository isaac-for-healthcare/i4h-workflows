import time

import cv2
from common.utils import get_ntp_offset
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec
from schemas.camera_stream import CameraStream


class CV2VideoCaptureOp(Operator):
    """
    Operator to capture video using OpenCV.
    Captures RGB frames.
    """

    def __init__(self, fragment, width: int, height: int, device_idx: int, framerate: int, *args, **kwargs):
        """
        Initialize the RealSense operator.

        Parameters:
        - width (int): Width of the camera stream.
        - height (int): Height of the camera stream.
        - device_idx (int): Camera device index.
        - framerate (int): Frame rate for the camera stream.
        - stream_type (str): Stream Type (color|depth).
        - stream_format (str): Stream format [pyrealsense2.format].
        """
        self.width = width
        self.height = height
        self.device_idx = device_idx
        self.framerate = framerate

        self.stream_type = 2  # color
        self.stream_format = 5  # rgb
        self.ntp_offset_time = get_ntp_offset()

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self.cap = cv2.VideoCapture(self.device_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.framerate)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at /dev/video{self.device_idx}")

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()

    def compute(self, op_input, op_output, context):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ts = int((time.time() + self.ntp_offset_time) * 1000)
        stream = CameraStream(
            ts=ts,
            type=self.stream_type,
            format=self.stream_format,
            width=self.width,
            height=self.height,
            data=data,
        )
        op_output.emit(stream, "output")
