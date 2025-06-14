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
from holohub.operators.nvjpeg.decoder import NVJpegDecoderOp
from holohub.operators.stats import CameraStreamStats
from holohub.operators.to_viz import CameraStreamToViz
from holoscan.core import Application, MetadataPolicy
from holoscan.operators.holoviz import HolovizOp
from holoscan.resources import RMMAllocator, UnboundedAllocator
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

        if self.decoder == "nvc":
            from holohub.operators.nvidia_video_codec.nv_video_decoder import NvVideoDecoderOp

            decoder_op = NvVideoDecoderOp(
                self,
                name="nvc_decoder",
                cuda_device_ordinal=0,
                allocator=RMMAllocator(self, name="video_decoder_allocator"),
            )
            split_op = CameraStreamSplitOp(self, name="split_op")
            merge_op = CameraStreamMergeOp(self, name="merge_op", for_encoder=False)
            merge_op.metadata_policy = MetadataPolicy.UPDATE
        else:
            decoder_op = NVJpegDecoderOp(
                self,
                name="nvjpeg_decoder",
                skip=self.decoder != "nvjpeg",
            )

        stats = CameraStreamStats(self, name="stats", interval_ms=1000)
        stream_to_viz = CameraStreamToViz(self)
        viz = HolovizOp(
            self,
            allocator=UnboundedAllocator(self, name="pool"),
            name="holoviz",
            window_title="Camera",
            width=self.width,
            height=self.height,
            framebuffer_srgb=self.srgb,
        )

        if self.decoder == "nvc":
            self.add_flow(dds, split_op, {("output", "input")})
            self.add_flow(split_op, merge_op, {("output", "input")})
            self.add_flow(split_op, decoder_op, {("image", "input")})
            self.add_flow(decoder_op, merge_op, {("output", "image")})
            self.add_flow(merge_op, stats, {("output", "input")})
            self.add_flow(decoder_op, viz, {("output", "receivers")})
        else:
            self.add_flow(dds, decoder_op, {("output", "input")})
            self.add_flow(decoder_op, stats, {("output", "input")})
            self.add_flow(decoder_op, stream_to_viz, {("output", "input")})
            self.add_flow(stream_to_viz, viz, {("output", "receivers")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the camera (rcv) application")
    parser.add_argument("--name", type=str, default="robot", help="camera name")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--decoder", type=str, choices=["nvjpeg", "none", "nvc"], default="nvc", help="decoder type")
    parser.add_argument("--domain_id", type=int, default=9, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")
    parser.add_argument("--srgb", type=bool, default=True, help="framebuffer srgb for viz")

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
