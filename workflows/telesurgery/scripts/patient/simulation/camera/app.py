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

from holohub.operators.camera.sim import IsaacSimToCameraStreamOp
from holohub.operators.dds.publisher import DDSPublisherOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_merge import CameraStreamMergeOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_split import CameraStreamSplitOp
from holohub.operators.nvjpeg.encoder import NVJpegEncoderOp
from holoscan.core import Application
from holoscan.resources import BlockMemoryPool, MemoryStorageType
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        width: int,
        height: int,
        dds_domain_id,
        dds_topic,
        encoder,
        encoder_params,
    ):
        self.width = width
        self.height = height
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.encoder = encoder
        self.encoder_params = encoder_params

        self.source: IsaacSimToCameraStreamOp | None = None

        super().__init__()

    def on_new_frame_rcvd(self, data, frame_num):
        if self.source is not None:
            self.source.on_new_frame_rcvd(data, frame_num)
        else:
            print(f"Discarding an incoming frame: {frame_num}!")

    def compose(self):
        self.source = IsaacSimToCameraStreamOp(
            self,
            name="sim_camera",
            width=self.width,
            height=self.height,
        )

        if self.encoder == "nvc":
            try:
                from holohub.operators.nvidia_video_codec.nv_video_encoder import NvVideoEncoderOp

                encoder_op = NvVideoEncoderOp(
                    self,
                    name="nvc_encoder",
                    cuda_device_ordinal=0,
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
                        storage_type=MemoryStorageType.HOST,
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

        if self.encoder == "nvc":
            self.add_flow(self.source, split_op, {("output", "input")})
            self.add_flow(split_op, merge_op, {("output", "input")})
            self.add_flow(split_op, encoder_op, {("image", "input")})
            self.add_flow(encoder_op, merge_op, {("output", "image")})
            self.add_flow(merge_op, dds, {("output", "input")})
        else:
            self.add_flow(self.source, encoder_op, {("output", "input")})
            self.add_flow(encoder_op, dds, {("output", "input")})
