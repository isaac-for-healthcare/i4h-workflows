# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Callable

from holoscan.core import Application
from holoscan.conditions import AsynchronousCondition, PeriodicCondition
from holoscan.operators import FormatConverterOp
from holoscan.resources import (
    BlockMemoryPool,
    MemoryStorageType
)
from dds_hid_subscriber import DDSHIDSubscriberOp
from dds_camera_info_publisher import DDSCameraInfoPublisherOp
from tensor_to_video_buffer import TensorToVideoBufferOp

from operators.data_bridge.HidToSimPushOp import HidToSimPushOp

from applications.encoder_imports import VideoEncoderContext, VideoEncoderRequestOp, VideoEncoderResponseOp

class PatientApp(Application):
    """A Holoscan application for transmitting data over RoCE (RDMA over Converged Ethernet).

    This application sets up a data transmission pipeline that can either transmit data
    over a RoCE network interface or display the data locally using Holoviz if no RoCE
    device is available.

    Args:
        ibv_name (str): Name of the InfiniBand verb (IBV) device to use for RoCE transmission.
        ibv_port (int): Port number for the IBV device.
        hololink_ip (str): IP address of the Hololink receiver.
        ibv_qp (int): Queue pair number for the IBV device.
        tx_queue_size (int): Size of the transmission queue.
        buffer_size (int): Size of the buffer for data transmission.
    """

    def __init__(
        self,
        hid_event_callback: Callable,
        width: int,
        height: int,
    ):
        """Initialize the TransmitterApp.

        Args:
            tx_queue_size (int): Size of the transmission queue.
            buffer_size (int): Size of the buffer for data transmission.
        """
        self._hid_event_callback = hid_event_callback
        self._logger = logging.getLogger(__name__)
        self._width = width
        self._height = height
        self._async_data_push = None
        super().__init__()

    def compose(self):
        """Compose the application workflow.

        Sets up the data transmission pipeline by creating and connecting the necessary operators.
        If a RoCE device is available, creates a RoceTransmitterOp for network transmission.
        Otherwise, creates a HolovizOp for local visualization.
        """

        source_block_size = self._width * self._height * 3 * 4
        source_num_blocks = 2

        source_rate_hz = 60  # messages sent per second
        period_source_ns = int(1e9 / source_rate_hz)  # period in nanoseconds

        hid_protocol = str(self.from_config("protocol.hid"))

        if hid_protocol == "dds":
            self.hid_event_source = DDSHIDSubscriberOp(
                self,
                PeriodicCondition(self, recess_period=period_source_ns),
                name="DDS HID Subscriber",
                **self.kwargs("hid"),
            )
        elif hid_protocol == "streamsdk":
            # TODO: Implement StreamSDK HID subscriber
            raise NotImplementedError("StreamSDK HID subscriber is not implemented")
        else:
            raise ValueError(f"Invalid HID protocol: '{hid_protocol}'")

        self._hid_to_sim_push = HidToSimPushOp(
            self,
            name="Hid to Sim Push",
            hid_event_callback=self._hid_event_callback,
        )
        self.add_flow(self.hid_event_source, self._hid_to_sim_push, {("output", "input")})

        video_protocol = str(self.from_config("protocol.video"))
        if video_protocol == "dds":
            from operators.data_bridge.AsyncDataPushOpForDDS import AsyncDataPushForDDS
            self._async_data_push = AsyncDataPushForDDS(
                self,
                name="Async Data Push",
                condition=AsynchronousCondition(self)
            )
            rgba_to_rgb_format_converter = FormatConverterOp(
                self,
                name="rgba_to_rgb_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("rgba_to_rgb_format_converter"),
            )
            rgb_to_yuv420_format_converter = FormatConverterOp(
                self,
                name="rgb_to_yuv420_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("rgb_to_yuv420_format_converter"),
            )
            tensor_to_video_buffer = TensorToVideoBufferOp(
                self, name="tensor_to_video_buffer", **self.kwargs("tensor_to_video_buffer")
            )
            encoder_async_condition = AsynchronousCondition(self, "encoder_async_condition")
            video_encoder_context = VideoEncoderContext(
                self, scheduling_term=encoder_async_condition
            )
            video_encoder_request = VideoEncoderRequestOp(
                self,
                name="video_encoder_request",
                input_width=self._width,
                input_height=self._height,
                videoencoder_context=video_encoder_context,
                **self.kwargs("video_encoder_request"),
            )
            video_encoder_response = VideoEncoderResponseOp(
                self,
                name="video_encoder_response",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                videoencoder_context=video_encoder_context,
                **self.kwargs("video_encoder_response"),
            )

            dds_publisher = DDSCameraInfoPublisherOp(
                self,
                name="DDS Camera Info Publisher",
                **self.kwargs("camera_info_publisher")
            )

            self.add_flow(self._async_data_push, rgba_to_rgb_format_converter, {("image", "source_video")})
            self.add_flow(rgba_to_rgb_format_converter, rgb_to_yuv420_format_converter, {("tensor", "source_video")})
            self.add_flow(rgb_to_yuv420_format_converter, tensor_to_video_buffer, {("tensor", "in_tensor")})
            self.add_flow(tensor_to_video_buffer, video_encoder_request, {("out_video_buffer", "input_frame")})
            self.add_flow(video_encoder_response, dds_publisher, {("output_transmitter", "image")})
            self.add_flow(self._async_data_push, dds_publisher, {("metadata", "metadata")})
        elif video_protocol == "streamsdk":
            # TODO: Implement StreamSDK video publisher
            from operators.data_bridge.AsyncDataPushOpForStreamSDK import AsyncDataPushOpForStreamSDK
            
            raise NotImplementedError("StreamSDK video publisher is not implemented")
        else:
            raise ValueError(f"Invalid video protocol: '{video_protocol}'")


    def push_data(self, data):
        """Push data into the transmission pipeline.

        Args:
            data: The data to be transmitted or displayed.
        """

        if self._async_data_push is not None:
            self._async_data_push.push_data(data)
        else:
            self._logger.warning("AsyncDataPushOp is not initialized")


import os
from argparse import ArgumentParser
from holoscan.gxf import load_extensions

def callback():
    pass

if __name__ == "__main__":
    #initialize logger
    logging.basicConfig(level=logging.INFO)
    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "h264_endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    width, height = 1920, 1080  # Match encoder dimensions
    app = PatientApp(callback, width, height)

    context = app.executor.context_uint64
    exts = [
        "libgxf_videodecoder.so",
        "libgxf_videodecoderio.so",
        "libgxf_videoencoder.so",
        "libgxf_videoencoderio.so",
    ]
    # load_extensions(context, exts)

    # create a thread to call app.push_data() with a random image
    import threading
    import time
    import numpy as np

    def random_image_thread():
        frame_num = 0
        while True:
            print("pushing data")
            # create a random image on GPU memory using cupy
            import cupy as cp
            image = cp.random.randint(0, 255, (height, width, 4), dtype=np.uint8) # Use height, width
            app.push_data({"image": image,
                           "joint_names": ["joint1", "joint2"],
                           "joint_positions": np.random.rand(2),
                           "size": (height, width, 4), # Update size
                           "frame_num": frame_num,
                           "last_hid_event": None,
                           "video_acquisition_timestamp": time.monotonic_ns()})
            frame_num += 1
            time.sleep(1)

    threading.Thread(target=random_image_thread).start()

    app.config(config_file)
    app.run()
