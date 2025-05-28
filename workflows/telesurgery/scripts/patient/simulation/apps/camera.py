from holohub.operators.camera.sim import IsaacSimCameraSourceOp
from holohub.operators.dds.publisher import DDSPublisherOp
from holohub.operators.nvjpeg.encoder import NVJpegEncoderOp
from holohub.operators.sink import NoOp
from holoscan.core import Application
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        width: int,
        height: int,
        dds_domain_id,
        dds_topic,
        jpeg_compress,
        jpeg_quality,
    ):
        self.width = width
        self.height = height
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.jpeg_compress = jpeg_compress
        self.jpeg_quality = jpeg_quality

        self.source: IsaacSimCameraSourceOp | None = None

        super().__init__()

    def on_new_frame(self, data, frame_num):
        if self.source is not None:
            self.source.on_new_frame(data, frame_num)
        else:
            print(f"Discarding an incoming frame: {frame_num}!")

    def compose(self):
        self.source = IsaacSimCameraSourceOp(
            self,
            name="sim_camera",
            width=self.width,
            height=self.height,
        )

        jpeg = NVJpegEncoderOp(
            self,
            name="nvjpeg_encoder",
            skip=not self.jpeg_compress,
            quality=self.jpeg_quality,
        )

        dds = DDSPublisherOp(
            self,
            name="dds_publisher",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=CameraStream,
        )

        sink = NoOp(self)

        self.add_flow(self.source, jpeg, {("output", "input")})
        self.add_flow(jpeg, dds, {("output", "input")})
        self.add_flow(dds, sink, {("output", "input")})
