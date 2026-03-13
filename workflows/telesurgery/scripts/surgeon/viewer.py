# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from holohub.operators.dds.subscriber import DDSSubscriberOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_merge import CameraStreamMergeOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_split import CameraStreamSplitOp
from holohub.operators.nvidia_video_codec.utils.merge_side_by_side import MergeSideBySideOp
from holohub.operators.nvjpeg.decoder import NVJpegDecoderOp
from holohub.operators.stats import CameraStreamStats
from holohub.operators.to_viz import CameraStreamToViz
from holoscan.core import Application, MetadataPolicy, Tracker
from holoscan.operators.format_converter import FormatConverterOp
from holoscan.operators.holoviz import HolovizOp
from holoscan.resources import RMMAllocator, UnboundedAllocator
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        width,
        height,
        fullscreen,
        decoder,
        dds_domain_id,
        dds_topic,
        srgb=False,
        is_3d_input=False,
        use_exclusive_display=False,
        vsync=False,
        upsample=False,
    ):
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.decoder = decoder
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.srgb = srgb
        self.is_3d_input = is_3d_input
        self.use_exclusive_display = use_exclusive_display
        self.vsync = vsync
        self.upsample = upsample
        super().__init__()

    def compose(self):
        dds = DDSSubscriberOp(
            self,
            name="dds_subscriber",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=CameraStream,
        )

        split_op = CameraStreamSplitOp(self, name="split_op")
        merge_op = CameraStreamMergeOp(self, name="merge_op", for_encoder=False)
        merge_op.metadata_policy = MetadataPolicy.UPDATE
        merge_side_by_side_op = MergeSideBySideOp(self, name="merge_side_by_side_op")

        if self.decoder == "nvc":
            from holohub.operators.nvidia_video_codec.nv_video_decoder import NvVideoDecoderOp

            decoder_op = NvVideoDecoderOp(
                self,
                name="nvc_decoder",
                cuda_device_ordinal=0,
                allocator=RMMAllocator(self, name="video_decoder_allocator", device_memory_max_size="50MB"),
            )
        else:
            decoder_op = NVJpegDecoderOp(
                self,
                name="nvjpeg_decoder",
                skip=self.decoder != "nvjpeg",
            )

        stats = CameraStreamStats(self, name="stats", interval_ms=1000, stream_lift=self.upsample)
        stream_to_viz = CameraStreamToViz(self)

        viz = HolovizOp(
            self,
            allocator=UnboundedAllocator(self, name="pool"),
            name="holoviz",
            window_title="Camera",
            width=self.width,
            height=self.height,
            framebuffer_srgb=self.srgb,
            fullscreen=self.fullscreen,
            use_exclusive_display=self.use_exclusive_display,
            vsync=self.vsync,
        )

        if self.upsample:
            from holohub.operators.streamlift.streamlift_upsampler import StreamLiftUpSamplerOp

            stream_lift = StreamLiftUpSamplerOp(
                self,
                cuda_device_ordinal=0,
                name="stream_lift_upsampler",
                allocator=UnboundedAllocator(self, name="streamlift_pool"),
            )
            nv12_to_rgb = FormatConverterOp(
                self,
                name="nv12_to_rgb",
                pool=UnboundedAllocator(self, name="pool"),
                in_dtype="nv12",
                out_dtype="rgb888",
            )
            rgb_to_rgba = FormatConverterOp(
                self,
                name="rgb_to_rgba",
                pool=UnboundedAllocator(self, name="pool"),
                in_dtype="rgb888",
                out_dtype="rgba8888",
            )

        if self.decoder == "nvc":
            self.add_flow(dds, split_op, {("output", "input")})
            self.add_flow(split_op, merge_op, {("output", "input")})
            self.add_flow(split_op, decoder_op, {("image", "input")})

            if self.upsample:
                self.add_flow(decoder_op, nv12_to_rgb, {("output", "source_video")})
                self.add_flow(nv12_to_rgb, rgb_to_rgba, {("tensor", "source_video")})
                self.add_flow(rgb_to_rgba, stream_lift, {("tensor", "input")})
                self.add_flow(stream_lift, merge_op, {("output", "image")})
                self.add_flow(merge_op, stats, {("output", "input")})

                if self.is_3d_input:
                    self.add_flow(merge_side_by_side_op, viz, {("output", "receivers")})
                else:
                    self.add_flow(stream_lift, viz, {("output", "receivers")})
            else:
                self.add_flow(decoder_op, merge_op, {("output", "image")})
                self.add_flow(merge_op, stats, {("output", "input")})

                if self.is_3d_input:
                    self.add_flow(decoder_op, merge_side_by_side_op, {("output", "input")})
                    self.add_flow(merge_side_by_side_op, viz, {("output", "receivers")})
                else:
                    self.add_flow(decoder_op, viz, {("output", "receivers")})
        else:
            self.add_flow(dds, decoder_op, {("output", "input")})
            if self.upsample:
                self.add_flow(decoder_op, split_op, {("output", "input")})
                self.add_flow(split_op, merge_op, {("output", "input")})
                if self.decoder == "nvjpeg":
                    self.add_flow(split_op, rgb_to_rgba, {("image", "source_video")})
                    self.add_flow(rgb_to_rgba, stream_lift, {("tensor", "input")})
                else:
                    self.add_flow(split_op, stream_lift, {("image", "input")})

                self.add_flow(stream_lift, merge_op, {("output", "image")})
                self.add_flow(merge_op, stats, {("output", "input")})
                self.add_flow(stream_lift, viz, {("output", "receivers")})
            else:
                self.add_flow(decoder_op, stats, {("output", "input")})
                self.add_flow(decoder_op, stream_to_viz, {("output", "input")})
                self.add_flow(stream_to_viz, viz, {("output", "receivers")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the camera (rcv) application")
    parser.add_argument("--name", type=str, default="robot", help="camera name")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--fullscreen", action="store_true", help="fullscreen mode")
    parser.add_argument("--decoder", type=str, choices=["nvjpeg", "none", "nvc"], default="nvc", help="decoder type")
    parser.add_argument("--domain_id", type=int, default=9, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")
    parser.add_argument("--srgb", action="store_true", help="framebuffer srgb for viz")
    parser.add_argument("--is_3d_input", action="store_true", help="is 3d input")
    parser.add_argument("--use_exclusive_display", action="store_true", help="use exclusive display")
    parser.add_argument("--enable_tracking", action="store_true", help="enable data flow tracking")
    parser.add_argument("--tracking_file", type=str, default="", help="tracking file")
    parser.add_argument("--vsync", action="store_true", help="enable VSync if screen tearing is an issue")
    parser.add_argument("--upsample", action="store_true", help="upsample to 4k input")

    args = parser.parse_args()
    app = App(
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        decoder=args.decoder,
        dds_domain_id=args.domain_id,
        dds_topic=args.topic if args.topic else f"telesurgery/{args.name}_camera/rgb",
        srgb=args.srgb,
        is_3d_input=args.is_3d_input,
        use_exclusive_display=args.use_exclusive_display,
        vsync=args.vsync,
        upsample=args.upsample,
    )

    if args.enable_tracking:
        with Tracker(app, filename=args.tracking_file) as tracker:
            app.run()
            tracker.print()

    else:
        app.run()


if __name__ == "__main__":
    main()
