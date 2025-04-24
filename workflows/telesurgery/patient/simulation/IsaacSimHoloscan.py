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
import argparse
import logging

# todo: importing holoscan before running IsaacSim results in this error caused by a
# missing symbol in `libyaml-cpp.so.0.7``
#  2025-03-24 11:44:16 [8,628ms] [Error] [omni.ext._impl.custom_importer] Failed to import
#  python module omni.isaac.motion_generation. Error:
#  /isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs/liblula_util.so:
#  undefined symbol: _ZN4YAML8LoadFileERKSs.
# Reason is that libyaml packed with Omniverse (and probably all of Omniverse) is compiled
# with _GLIBCXX_USE_CXX11_ABI=0 and Holoscan is compiled with _GLIBCXX_USE_CXX11_ABI=1.
import holoscan
from applications.TransmitterApp import TransmitterApp

from Simulation import Simulation


def main():
    """
    Main entry point for the IsaacSim Holoscan simulation.

    This function sets up and runs the simulation environment with the following components:
    1. IsaacSim simulation environment
    2. Holoscan transmitter application
    3. Holoscan receiver application

    The function parses command line arguments to configure:
    - Logging level
    - Headless mode
    - IBV device names and ports
    - Hololink IP addresses
    - Queue parameters
    - Image dimensions

    The simulation runs until completion, and the function waits for both transmitter
    and receiver applications to finish before exiting.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=False,
        help="Run IsaacSim in headless mode",
    )
    parser.add_argument(
        "--ibv-name-tx",
        type=str,
        default="",
        help="IBV device to use for transmitter",
    )
    parser.add_argument(
        "--ibv-name-rx",
        type=str,
        default="",
        help="IBV device to use for receiver",
    )
    parser.add_argument(
        "--ibv-port-tx",
        type=int,
        default=1,
        help="Port number of IBV device for transmitter",
    )
    parser.add_argument(
        "--ibv-port-rx",
        type=int,
        default=1,
        help="Port number of IBV device for receiver",
    )
    parser.add_argument(
        "--hololink-ip-tx",
        type=str,
        default="192.168.0.2",
        help="IP address of Hololink board for transmitter",
    )
    parser.add_argument(
        "--hololink-ip-rx",
        type=str,
        default="192.168.0.3",
        help="IP address of Hololink board for receiver",
    )
    parser.add_argument(
        "--ibv-qp",
        type=int,
        default=2,
        help="QP number for the IBV stream",
    )
    parser.add_argument(
        "--queue-size-tx",
        type=int,
        default=2,
        help="Transmitter queue size",
    )
    args = parser.parse_args()

    image_size = (1080, 1920, 4)

    # start the simulation
    simulation = Simulation(args.headless, image_size)

    # set up logging
    # hololink.logging_level(args.log_level)
    if args.log_level == logging.DEBUG:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.DEBUG)
    elif args.log_level == logging.INFO:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.INFO)
    elif args.log_level == logging.WARNING:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.WARN)
    elif args.log_level == logging.ERROR:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.ERROR)
    elif args.log_level == logging.CRITICAL:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.CRITICAL)
    elif args.log_level == logging.NOTSET:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.OFF)

    # Set up the Holoscan transmitter application
    transmitter_app = TransmitterApp(
        ibv_name=args.ibv_name_tx,
        ibv_port=args.ibv_port_tx,
        hololink_ip=args.hololink_ip_tx,
        ibv_qp=args.ibv_qp,
        tx_queue_size=args.queue_size_tx,
        buffer_size=image_size[0] * image_size[1] * image_size[2],
    )
    # Run the transmitter application
    transmitter_future = transmitter_app.run_async()

    def transmitter_done_callback(future):
        if future.exception():
            print(f"TransmitterApp failed with exception: {future.exception()}")
            os._exit(1)

    transmitter_future.add_done_callback(transmitter_done_callback)

    # # Set up the Holoscan receiver application
    # receiver_app = ReceiverApp(
    #     ibv_name=args.ibv_name_rx,
    #     ibv_port=args.ibv_port_rx,
    #     hololink_ip=args.hololink_ip_rx,
    #     buffer_size=image_size[0] * image_size[1] * image_size[2],
    #     data_ready_callback=simulation.data_ready_callback,
    # )
    # # Run the receiver application
    # receiver_future = receiver_app.run_async()

    # def receiver_done_callback(future):
    #     if future.exception():
    #         print(f"ReceiverApp failed with exception: {future.exception()}")
    #         os._exit(1)

    # receiver_future.add_done_callback(receiver_done_callback)

    # Run the simulation, this will return if the simulation is finished
    simulation.run(transmitter_app.push_data)

    # Wait for the transmitter and receiver applications to finish
    transmitter_future.result()
    # receiver_future.result()


if __name__ == "__main__":
    main()
