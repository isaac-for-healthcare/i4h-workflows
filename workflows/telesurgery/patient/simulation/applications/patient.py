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
from holoscan.operators import HolovizOp
from dds_hid_subscriber import DDSHIDSubscriberOp
from dds_camera_info_publisher import DDSCameraInfoPublisherOp

from operators.data_bridge.HidToSimPushOp import HidToSimPushOp

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
        tx_queue_size: int,
        buffer_size: int,
        hid_event_callback: Callable,
    ):
        """Initialize the TransmitterApp.

        Args:
            ibv_name (str): Name of the InfiniBand verb (IBV) device.
            ibv_port (int): Port number for the IBV device.
            hololink_ip (str): IP address of the Hololink receiver.
            ibv_qp (int): Queue pair number for the IBV device.
            tx_queue_size (int): Size of the transmission queue.
            buffer_size (int): Size of the buffer for data transmission.
        """
        self._tx_queue_size = tx_queue_size
        self._buffer_size = buffer_size
        self._hid_event_callback = hid_event_callback
        self._logger = logging.getLogger(__name__)

        self._async_data_push = None
        super().__init__()

    def compose(self):
        """Compose the application workflow.

        Sets up the data transmission pipeline by creating and connecting the necessary operators.
        If a RoCE device is available, creates a RoceTransmitterOp for network transmission.
        Otherwise, creates a HolovizOp for local visualization.
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
            from operators.data_bridge.AsyncDataPushOpForDDS import AsyncDataPushForDDS
            self._async_data_push = AsyncDataPushForDDS(
                self,
                name="Async Data Push",
                condition=AsynchronousCondition(self)
            )
            video_source = DDSCameraInfoPublisherOp(
                self,
                name="DDS Camera Info Publisher",
                **self.kwargs("camera_info_publisher")
            )
            self.add_flow(self._async_data_push, video_source, {("camera_info", "input")})
        elif video_protocol == "streamsdk":
            # TODO: Implement StreamSDK video publisher
            from operators.data_bridge.AsyncDataPushOpForStreamSDK import AsyncDataPushOpForStreamSDK
            
            raise NotImplementedError("StreamSDK video publisher is not implemented")
        else:
            raise ValueError(f"Invalid video protocol: '{video_protocol}'")


        # holoviz = HolovizOp(
        #     self,
        #     name="Holoviz",
        #     tensors=[
        #         HolovizOp.InputSpec("", HolovizOp.InputType.COLOR),
        #     ],
        # )

        # self.add_flow(self._async_data_push, holoviz, {("image", "receivers")})

    def push_data(self, data):
        """Push data into the transmission pipeline.

        Args:
            data: The data to be transmitted or displayed.
        """

        if self._async_data_push is not None:
            self._async_data_push.push_data(data)
        else:
            self._logger.warning("AsyncDataPushOp is not initialized")
