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

import argparse
import logging
import os

# todo: importing holoscan before running IsaacSim results in this error caused by a
# missing symbol in `libyaml-cpp.so.0.7``
#  2025-03-24 11:44:16 [8,628ms] [Error] [omni.ext._impl.custom_importer] Failed to import
#  python module omni.isaac.motion_generation. Error:
#  /isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs/liblula_util.so:
#  undefined symbol: _ZN4YAML8LoadFileERKSs.
# Reason is that libyaml packed with Omniverse (and probably all of Omniverse) is compiled
# with _GLIBCXX_USE_CXX11_ABI=0 and Holoscan is compiled with _GLIBCXX_USE_CXX11_ABI=1.
import holoscan
from applications.patient import PatientApp
from applications.simulation import Simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "patient.yaml"),
        help="Path to the patient configuration file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Width of the image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Height of the image",
    )
    parser.add_argument(
        "--camera-frequency",
        type=int,
        default=60,
        help="Frequency of the camera",
    )
    args = parser.parse_args()

    image_size = (args.height, args.width, 4)

    # start the simulation
    simulation = Simulation(args.headless, image_size, args.camera_frequency)

    # set up logging
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
    patient_app = PatientApp(
        hid_event_callback=simulation.hid_event_callback,
        width=args.width,
        height=args.height,
    )

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")

    logger.info(f"Using config file: {args.config}")

    patient_app.config(args.config)
    # Run the transmitter application
    patient_future = patient_app.run_async()

    def patient_done_callback(future):
        if future.exception():
            logger.error(f"PatientApp failed with exception: {future.exception()}")
            os._exit(1)

    patient_future.add_done_callback(patient_done_callback)

    # Run the simulation, this will return if the simulation is finished
    try:
        logger.info("Starting simulation run...")
        simulation.run(patient_app.push_data)
        logger.info("Simulation run finished normally.")
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Stopping simulation and Holoscan app...")
        simulation.stop()
    except Exception as e:
        logger.error(f"Simulation failed with exception: {e}", exc_info=True)
        simulation.stop()
        os._exit(1)
    finally:
        simulation.stop()
        logger.info("Waiting for PatientApp to finish...")
        try:
            patient_future.result(timeout=5)
            logger.info("PatientApp finished.")
        except TimeoutError:
            logger.warning("PatientApp did not finish within the timeout.")
        except Exception as e:
            logger.error(f"Error waiting for PatientApp: {e}", exc_info=True)

    logger.info("Exiting main application.")


if __name__ == "__main__":
    main()
