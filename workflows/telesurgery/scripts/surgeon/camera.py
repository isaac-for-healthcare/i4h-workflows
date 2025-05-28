import argparse

from holohub.operators.dds.subscriber import DDSSubscriberOp
from holohub.operators.nvjpeg.decoder import NVJpegDecoderOp
from holohub.operators.stats import CameraStreamStats
from holohub.operators.to_viz import CameraStreamToViz
from holoscan.core import Application
from holoscan.operators.holoviz import HolovizOp
from holoscan.resources import UnboundedAllocator
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        width,
        height,
        decoder,
        dds_domain_id,
        dds_topic,
    ):
        self.width = width
        self.height = height
        self.decoder = decoder
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic

        super().__init__()

    def compose(self):
        dds = DDSSubscriberOp(
            self,
            name="dds_subscriber",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=CameraStream,
        )
        jpeg = NVJpegDecoderOp(
            self,
            name="nvjpeg_decoder",
            skip=self.decoder != "nvjpeg",
        )
        stats = CameraStreamStats(self, interval_ms=1000)
        stream_to_viz = CameraStreamToViz(self)
        viz = HolovizOp(
            self,
            allocator=UnboundedAllocator(self, name="pool"),
            name="holoviz",
            window_title="Camera",
            width=self.width,
            height=self.height,
        )

        self.add_flow(dds, jpeg, {("output", "input")})
        self.add_flow(jpeg, stats, {("output", "input")})
        self.add_flow(stats, stream_to_viz, {("output", "input")})
        self.add_flow(stream_to_viz, viz, {("output", "receivers")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the camera (rcv) application")
    parser.add_argument("--name", type=str, default="robot", help="camera name")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--decoder", type=str, choices=["nvjpeg", "none"], default="nvjpeg")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")

    args = parser.parse_args()
    app = App(
        width=args.width,
        height=args.height,
        decoder=args.decoder,
        dds_domain_id=args.domain_id,
        dds_topic=args.topic if args.topic else f"telesurgery/{args.name}_camera/rgb",
    )
    app.run()


if __name__ == "__main__":
    main()
