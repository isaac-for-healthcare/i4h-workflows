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

import os

from holohub.operators.streamlift.streamlift_upsampler import StreamLiftUpSamplerOp
from holohub.operators.streamlift.utils import StreamliftStatsOp
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp, VideoStreamReplayerOp
from holoscan.resources import RMMAllocator, UnboundedAllocator


class StreamLiftUpSamplerApp(Application):
    def __init__(self, data=None):
        super().__init__()

        self.name = "Stream Lift Up Sampler App"
        if data is None:
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")
        self.sample_data_path = data

    def compose(self):
        fps = 30
        video_dir = self.sample_data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

        pool = UnboundedAllocator(self, name="pool")

        source = VideoStreamReplayerOp(
            self,
            PeriodicCondition(self, name="periodic-condition", recess_period=1 / fps),
            name="replayer",
            directory=video_dir,
            allocator=RMMAllocator(
                self,
                name="video_replayer_allocator",
                device_memory_initial_size="160MB",
                device_memory_max_size="320MB",
                host_memory_initial_size="160MB",
                host_memory_max_size="320MB",
            ),
            basename="sample",
            frame_rate=0,
            repeat=False,
            realtime=False,
            count=0,
        )

        format_converter_1 = FormatConverterOp(
            self,
            name="format_converter_1",
            pool=pool,
            in_dtype="rgb888",
            out_dtype="rgba8888",
            out_channel_order=[0, 1, 2, 3],
        )

        format_converter_2 = FormatConverterOp(
            self,
            name="format_converter_2",
            pool=pool,
            in_dtype="rgb888",
            out_dtype="rgba8888",
            out_channel_order=[0, 1, 2, 3],
        )

        remove_alpha = FormatConverterOp(self, name="remove_alpha", pool=pool, in_dtype="rgba8888", out_dtype="rgb888")

        streamlift = StreamLiftUpSamplerOp(
            self,
            cuda_device_ordinal=0,
            name="streamlift_upsampler",
            allocator=UnboundedAllocator(self, name="streamlift_pool"),
        )

        visualizer = HolovizOp(self, name="holoviz", framebuffer_srgb=True)
        stats = StreamliftStatsOp(self, name="streamlift_stats", interval_ms=1000)

        self.add_flow(source, format_converter_1, {("output", "source_video")})
        self.add_flow(format_converter_1, streamlift, {("tensor", "input")})
        self.add_flow(streamlift, remove_alpha, {("output", "source_video")})
        self.add_flow(remove_alpha, format_converter_2, {("tensor", "source_video")})
        self.add_flow(format_converter_2, visualizer, {("tensor", "receivers")})
        self.add_flow(format_converter_2, stats, {("tensor", "input")})

        # self.add_flow(source, visualizer, {("output", "receivers")})


if __name__ == "__main__":
    # ffmpeg -i sample.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | \
    # convert_video_to_gxf_entities.py --width 1920 --height 1080 --channels 3 --framerate 30 --basename sample
    app = StreamLiftUpSamplerApp(data="/workspace/i4h-workflows/third_party/holohub/data/sample2k/")
    app.run()
