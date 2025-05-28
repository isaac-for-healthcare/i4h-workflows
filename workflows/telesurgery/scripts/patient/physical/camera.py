import argparse
import json

from holohub.operators.camera.cv2 import CV2VideoCaptureOp
from holohub.operators.camera.realsense import RealsenseOp
from holohub.operators.dds.publisher import DDSPublisherOp
from holohub.operators.nvjpeg.encoder import NVJpegEncoderOp
from holohub.operators.sink import NoOp
from holoscan.core import Application
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        camera: str,
        name: str,
        width: int,
        height: int,
        device_idx: int,
        framerate: int,
        stream_type,
        stream_format,
        dds_domain_id,
        dds_topic,
        encoder,
        encoder_params,
    ):
        self.camera = camera
        self.name = name
        self.width = width
        self.height = height
        self.device_idx = device_idx
        self.framerate = framerate
        self.stream_type = stream_type
        self.stream_format = stream_format
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.encoder = encoder
        self.encoder_params = encoder_params

        super().__init__()

    def compose(self):
        source = (
            RealsenseOp(
                self,
                name=self.name,
                width=self.width,
                height=self.height,
                device_idx=self.device_idx,
                framerate=self.framerate,
                stream_type=self.stream_type,
                stream_format=self.stream_format,
            )
            if self.camera == "realsense"
            else CV2VideoCaptureOp(
                self,
                name=self.name,
                width=self.width,
                height=self.height,
                device_idx=self.device_idx,
                framerate=self.framerate,
            )
        )

        jpeg = NVJpegEncoderOp(
            self,
            name="nvjpeg_encoder",
            skip=self.encoder != "nvjpeg",
            quality=self.encoder_params.get("quality", 90),
        )

        dds = DDSPublisherOp(
            self,
            name="dds_publisher",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=CameraStream,
        )

        sink = NoOp(self)

        self.add_flow(source, jpeg, {("output", "input")})
        self.add_flow(jpeg, dds, {("output", "input")})
        self.add_flow(dds, sink, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the camera application")
    parser.add_argument("--camera", type=str, default="cv2", choices=["realsense", "cv2"], help="camera type")
    parser.add_argument("--name", type=str, default="robot", help="camera name")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--device_idx", type=int, default=0, help="device index")
    parser.add_argument("--framerate", type=int, default=30, help="frame rate")
    parser.add_argument("--stream_type", type=str, default="color", choices=["color", "depth"])
    parser.add_argument("--stream_format", type=str, default="")
    parser.add_argument("--encoder", type=str, choices=["nvjpeg", "none"], default="nvjpeg")
    parser.add_argument("--encoder_params", type=str, default=json.dumps({"quality": 90}), help="encoder params")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")

    args = parser.parse_args()
    app = App(
        camera=args.camera,
        name=args.name,
        width=args.width,
        height=args.height,
        device_idx=args.device_idx,
        framerate=args.framerate,
        stream_type=args.stream_type,
        stream_format=args.stream_format,
        encoder=args.encoder,
        encoder_params=json.loads(args.encoder_params) if args.encoder_params else {},
        dds_domain_id=args.domain_id,
        dds_topic=args.topic if args.topic else f"telesurgery/{args.name}_camera/rgb",
    )
    app.run()


if __name__ == "__main__":
    main()
