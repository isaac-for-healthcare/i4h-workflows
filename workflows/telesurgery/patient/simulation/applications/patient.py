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

from dds_hid_subscriber import DDSHIDSubscriberOp
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp
from operators.data_bridge.AsyncDataPushOp import AsyncDataPushOp
from operators.data_bridge.HidToSimPushOp import HidToSimPushOp
from operators.dds.CameraInfoPublisherOp import CameraInfoPublisherOp


class PatientApp(Application):
    """The Patient application for the telesurgery simulation.

    This application sets up a Holoscan pipeline that receives HID events from the Surgeon application
    over DDS and pushes them to the Simulation application.
    The simulation application then updates the robot's joint positions and simulates the robot's motion.
    The camera attached to the robot pushes image frames to the AsyncDataPushOp for transmission to the
    Surgeon application and for local display in Holoviz.
    """

    def __init__(
        self,
        tx_queue_size: int,
        buffer_size: int,
        hid_event_callback: Callable,
    ):
        """Initialize the TransmitterApp.

        Args:
            tx_queue_size (int): Size of the transmission queue.
            buffer_size (int): Size of the buffer for data transmission.
            hid_event_callback (Callable): Callback function for handling HID events.
        """
        self._tx_queue_size = tx_queue_size
        self._buffer_size = buffer_size
        self._hid_event_callback = hid_event_callback
        self._logger = logging.getLogger(__name__)

        self._async_data_push = None
        super().__init__()

    def compose(self):
        """Compose the application workflow.
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

        self._async_data_push = AsyncDataPushOp(
            self,
            name="Async Data Push",
        )
        video_protocol = str(self.from_config("protocol.video"))
        if video_protocol == "dds":
            video_source = CameraInfoPublisherOp(
                self, name="DDS Camera Info Publisher", **self.kwargs("camera_info_publisher")
            )
        elif video_protocol == "streamsdk":
            # TODO: Implement StreamSDK video publisher
            raise NotImplementedError("StreamSDK video publisher is not implemented")
        else:
            raise ValueError(f"Invalid video protocol: '{video_protocol}'")

        self.add_flow(self._async_data_push, video_source, {("camera_info", "input")})

        holoviz = HolovizOp(
            self,
            name="Holoviz",
            tensors=[
                HolovizOp.InputSpec("", HolovizOp.InputType.COLOR),
            ],
        )

        self.add_flow(self._async_data_push, holoviz, {("image", "receivers")})

    def push_data(self, data):
        """Push the camera data to the AsyncDataPushOp.

        Args:
            data: The camera data to be transmitted or displayed.
        """

        if self._async_data_push is not None:
            self._async_data_push.push_data(data)
        else:
            self._logger.warning("AsyncDataPushOp is not initialized")
