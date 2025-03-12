import argparse
import os

import holoscan
import numpy as np
import pyrealsense2 as rs
import rti.connextdds as dds
from dds.schemas.camera_info import CameraInfo
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator
from holoscan.core._core import OperatorSpec
from holoscan.operators.holoviz import HolovizOp
from holoscan.resources import UnboundedAllocator


class RealsenseOp(Operator):
    """
    Operator to interface with an Intel RealSense camera.
    Captures RGB and depth frames, optionally publishing them via DDS.
    """

    def __init__(
        self,
        fragment,
        *args,
        domain_id,
        width,
        height,
        topic_rgb,
        topic_depth,
        device_idx,
        framerate,
        show_holoviz,
        **kwargs,
    ):
        """
        Initialize the RealSense operator.

        Parameters:
        - domain_id (int): DDS domain ID.
        - width (int): Width of the camera stream.
        - height (int): Height of the camera stream.
        - topic_rgb (str): DDS topic for RGB frames.
        - topic_depth (str): DDS topic for depth frames.
        - device_idx (int): Camera device index.
        - framerate (int): Frame rate for the camera stream.
        - show_holoviz (bool): Whether to display frames using Holoviz.
        """
        self.domain_id = domain_id
        self.width = width
        self.height = height
        self.topic_rgb = topic_rgb
        self.topic_depth = topic_depth
        self.device_idx = device_idx
        self.framerate = framerate
        self.show_holoviz = show_holoviz
        super().__init__(fragment, *args, **kwargs)

        self.pipeline = rs.pipeline()
        self.rgb_writer = None
        self.depth_writer = None

    def setup(self, spec: OperatorSpec):
        """Define the output ports for the operator."""
        if self.topic_rgb:
            spec.output("color")
        if self.topic_depth and not self.show_holoviz:
            spec.output("depth")

    def start(self):
        """
        Configure and start the RealSense camera pipeline.
        Sets up DDS writers if topics are provided.
        """
        config = rs.config()
        context = rs.context()
        for device in context.query_devices():
            print(f"+++ Available device: {device}")

        if self.device_idx is not None:
            if self.device_idx < len(context.query_devices()):
                config.enable_device(context.query_devices()[self.device_idx].get_info(rs.camera_info.serial_number))
            else:
                print(f"Ignoring input device_idx: {self.device_idx}")

        dp = dds.DomainParticipant(domain_id=self.domain_id)
        if self.topic_rgb:
            print("Enabling color...")
            self.rgb_writer = dds.DataWriter(dp.implicit_publisher, dds.Topic(dp, self.topic_rgb, CameraInfo))
            config.enable_stream(
                rs.stream.color,
                width=self.width,
                height=self.height,
                format=rs.format.rgba8,
                framerate=self.framerate,
            )
        if self.topic_depth:
            print("Enabling depth...")
            config.enable_stream(
                rs.stream.depth,
                width=self.width,
                height=self.height,
                format=rs.format.z16,
                framerate=self.framerate,
            )
            self.depth_writer = dds.DataWriter(dp.implicit_publisher, dds.Topic(dp, self.topic_depth, CameraInfo))
        self.pipeline.start(config)

    def compute(self, op_input, op_output, context):
        """
        Capture frames from the RealSense camera and publish them via DDS.
        """
        frames = self.pipeline.wait_for_frames()
        color = None
        if self.rgb_writer:
            color = np.asanyarray(frames.get_color_frame().get_data())
            self.rgb_writer.write(CameraInfo(data=color.tobytes(), width=self.width, height=self.height))
            op_output.emit({"color": holoscan.as_tensor(color)} if self.show_holoviz else True, "color")
        if self.depth_writer:
            depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) / 1000.0
            self.depth_writer.write(CameraInfo(data=depth.tobytes(), width=self.width, height=self.height))
            if not self.show_holoviz:
                op_output.emit({"depth": holoscan.as_tensor(depth)}, "depth")


class NoOp(Operator):
    """A sink operator that takes input and discards them."""

    def __init__(self, fragment, depth, *args, **kwargs):
        """Initialize the operator."""
        self.depth = depth
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Define input ports for color and optional depth."""
        spec.input("color")
        if self.depth:
            spec.input("depth")

    def compute(self, op_input, op_output, context):
        """Receive and discard input frames."""
        op_input.receive("color")
        if self.depth:
            op_input.receive("depth")


class RealsenseApp(Application):
    """Application to run the RealSense operator and process its output."""

    def __init__(self, domain_id, height, width, topic_rgb, topic_depth, device_idx, framerate, show_holoviz, count):
        self.domain_id = domain_id
        self.height = height
        self.width = width
        self.topic_rgb = topic_rgb
        self.topic_depth = topic_depth
        self.device_idx = device_idx
        self.framerate = framerate
        self.show_holoviz = show_holoviz
        self.count = count
        super().__init__()

    def compose(self):
        """Create and connect application operators."""
        camera = RealsenseOp(
            self,
            CountCondition(self, self.count),
            name="realsense",
            domain_id=self.domain_id,
            height=self.height,
            width=self.width,
            topic_rgb=self.topic_rgb,
            topic_depth=self.topic_depth,
            device_idx=self.device_idx,
            framerate=self.framerate,
            show_holoviz=self.show_holoviz,
        )

        if self.show_holoviz:
            holoviz = HolovizOp(
                self,
                allocator=UnboundedAllocator(self, name="pool"),
                name="holoviz",
                window_title="Realsense Camera",
                width=self.width,
                height=self.height,
            )
            self.add_flow(camera, holoviz, {("color", "receivers")})
        else:
            port_map = {("color", "color"), ("depth", "depth")} if self.topic_depth else {("color", "color")}
            self.add_flow(camera, NoOp(self, True if self.topic_depth else None), port_map)


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="show holoviz")
    parser.add_argument(
        "--count",
        type=int,
        default=-1,
        help="Number of frames to run",
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=int(os.environ.get("OVH_DDS_DOMAIN_ID", 0)),
        help="domain id",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("OVH_HEIGHT", 480)),
        help="height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("OVH_WIDTH", 640)),
        help="width",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=30,
        help="frame rate",
    )
    parser.add_argument(
        "--device_idx",
        type=int,
        default=None,
        help="device index case of multiple cameras",
    )
    parser.add_argument(
        "--topic_rgb",
        type=str,
        default="topic_room_camera_data_rgb",
        help="topic name to produce camera rgb",
    )
    parser.add_argument(
        "--topic_depth",
        type=str,
        default=None,  # "topic_room_camera_data_depth",
        help="topic name to produce camera depth",
    )

    args = parser.parse_args()
    app = RealsenseApp(
        args.domain_id,
        args.height,
        args.width,
        args.topic_rgb,
        args.topic_depth,
        args.device_idx,
        args.framerate,
        args.test,
        args.count,
    )
    app.run()


if __name__ == "__main__":
    main()
