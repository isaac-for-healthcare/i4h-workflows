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
import json

from holohub.operators.camera.cv2 import CV2VideoCaptureOp
from holohub.operators.camera.realsense import RealsenseOp
from holohub.operators.dds.publisher import DDSPublisherOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_merge import CameraStreamMergeOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_split import CameraStreamSplitOp
from holohub.operators.nvjpeg.encoder import NVJpegEncoderOp
from holohub.operators.sink import NoOp
from holoscan.core import Application
from holoscan.resources import BlockMemoryPool, MemoryStorageType
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        camera: str,
        camera_name: str,
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
        self.camera_name = camera_name
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
                name=self.camera_name,
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
                name=self.camera_name,
                width=self.width,
                height=self.height,
                device_idx=self.device_idx,
                framerate=self.framerate,
            )
        )

        if self.encoder == "nvc":
            try:
                from holohub.operators.nvidia_video_codec.nv_video_encoder import NvVideoEncoderOp

                encoder_op = NvVideoEncoderOp(
                    self,
                    name="nvc_encoder",
                    cuda_device_ordinal=0,
                    copy_to_host=False,
                    width=self.width,
                    height=self.height,
                    codec=self.encoder_params.get("codec", "H264"),
                    preset=self.encoder_params.get("preset", "P3"),
                    bitrate=self.encoder_params.get("bitrate", 10000000),
                    frame_rate=self.encoder_params.get("frame_rate", 60),
                    rate_control_mode=self.encoder_params.get("rate_control_mode", 1),
                    multi_pass_encoding=self.encoder_params.get("multi_pass_encoding", 0),
                    allocator=BlockMemoryPool(
                        self,
                        name="pool",
                        storage_type=MemoryStorageType.DEVICE,
                        block_size=self.width * self.height * 3 * 4,
                        num_blocks=2,
                    ),
                )
            except Exception as e:
                print(f"Error creating NVC encoder: {e}")
                raise e
            split_op = CameraStreamSplitOp(self, name="split_op")
            merge_op = CameraStreamMergeOp(self, name="merge_op")
        else:
            encoder_op = NVJpegEncoderOp(
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

        if self.encoder == "nvc":
            print("Using NVC encoder with split and merge")
            self.add_flow(source, split_op, {("output", "input")})
            self.add_flow(split_op, encoder_op, {("camera", "input")})
            self.add_flow(split_op, merge_op, {("metadata", "metadata")})
            self.add_flow(encoder_op, merge_op, {("output", "camera")})
            self.add_flow(merge_op, dds, {("output", "input")})
            self.add_flow(dds, sink, {("output", "input")})
        else:
            self.add_flow(source, encoder_op, {("output", "input")})
            self.add_flow(encoder_op, dds, {("output", "input")})
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
    parser.add_argument("--encoder", type=str, choices=["nvjpeg", "nvc", "none"], default="nvjpeg")
    parser.add_argument("--encoder_params", type=str, default=json.dumps({"quality": 90}), help="encoder params")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")

    args = parser.parse_args()
    app = App(
        camera=args.camera,
        camera_name=args.name,
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
