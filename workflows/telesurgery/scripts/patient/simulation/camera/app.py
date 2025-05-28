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
        encoder,
        encoder_params,
    ):
        self.width = width
        self.height = height
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.encoder = encoder
        self.encoder_params = encoder_params

        self.source: IsaacSimCameraSourceOp | None = None

        super().__init__()

    def on_new_frame_rcvd(self, data, frame_num):
        if self.source is not None:
            self.source.on_new_frame_rcvd(data, frame_num)
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

        self.add_flow(self.source, jpeg, {("output", "input")})
        self.add_flow(jpeg, dds, {("output", "input")})
        self.add_flow(dds, sink, {("output", "input")})
