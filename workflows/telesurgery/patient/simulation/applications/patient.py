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

from dds_camera_info_publisher import DDSCameraInfoPublisherOp
from dds_hid_subscriber import DDSHIDSubscriberOp
from holoscan.conditions import AsynchronousCondition, PeriodicCondition
from holoscan.core import Application
from holoscan.resources import UnboundedAllocator
from nv_video_encoder import NvVideoEncoderOp
from operators.data_bridge.HidToSimPushOp import HidToSimPushOp


class PatientApp(Application):
    def __init__(
        self,
        hid_event_callback: Callable,
        width: int,
        height: int,
    ):
        """Initialize the Patient Application.

        Args:
            hid_event_callback: Callback function for handling HID events.
            width: Width of the video stream.
            height: Height of the video stream.
        """
        self._hid_event_callback = hid_event_callback
        self._width = width
        self._height = height
        self._logger = logging.getLogger(__name__)

        self._async_data_push = None
        super().__init__()

    def compose(self):
        """
        Compose the application workflow.
        """

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
            self._compose_dds_video_pipeline()
        elif video_protocol == "streamsdk":
            self._compose_streamsdk_video_pipeline()
        else:
            raise ValueError(f"Invalid video protocol: '{video_protocol}'")

    def _compose_dds_video_pipeline(self):
        from operators.data_bridge.AsyncDataPushOpForDDS import AsyncDataPushForDDS

        self._async_data_push = AsyncDataPushForDDS(self, name="Async Data Push", condition=AsynchronousCondition(self))

        video_encoder = NvVideoEncoderOp(
            self,
            name="Video Encoder",
            width=self._width,
            height=self._height,
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("video_encoder"),
        )

        publisher = DDSCameraInfoPublisherOp(
            self, name="DDS Camera Info Publisher", **self.kwargs("camera_info_publisher")
        )
        self.add_flow(self._async_data_push, video_encoder, {("image", "in")})
        self.add_flow(video_encoder, publisher, {("out", "image")})
        self.add_flow(self._async_data_push, publisher, {("camera_info", "metadata")})

    def _compose_streamsdk_video_pipeline(self):
        # TODO: Implement StreamSDK video publisher

        raise NotImplementedError("StreamSDK video publisher is not implemented")

    def push_data(self, data):
        """Push data into the transmission pipeline.

        Args:
            data: The data to be transmitted or displayed.
        """

        if self._async_data_push is not None:
            self._async_data_push.push_data(data)
        else:
            self._logger.warning("AsyncDataPushOp is not initialized")
