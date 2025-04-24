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

import os

from holoscan.core import Application
from operators.AsyncDataPushOp import AsyncDataPushOp
from holoscan.operators import HolovizOp

# import hololink


class TransmitterApp(Application):
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
        ibv_name: str,
        ibv_port: int,
        hololink_ip: str,
        ibv_qp: int,
        tx_queue_size: int,
        buffer_size: int,
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
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._hololink_ip = hololink_ip
        self._ibv_qp = ibv_qp
        self._tx_queue_size = tx_queue_size
        self._buffer_size = buffer_size
        super().__init__()

    def compose(self):
        """Compose the application workflow.

        Sets up the data transmission pipeline by creating and connecting the necessary operators.
        If a RoCE device is available, creates a RoceTransmitterOp for network transmission.
        Otherwise, creates a HolovizOp for local visualization.
        """
        self._async_data_push = AsyncDataPushOp(
            self,
            name="Async Data Push",
        )

        transmitter = HolovizOp(
                self,
                name="Holoviz",
                tensors=[
                    HolovizOp.InputSpec("", HolovizOp.InputType.COLOR),
                ],
            )

        self.add_flow(self._async_data_push, transmitter, {("out", "receivers")})

    def push_data(self, data):
        """Push data into the transmission pipeline.

        Args:
            data: The data to be transmitted or displayed.
        """
        self._async_data_push.push_data(data)
