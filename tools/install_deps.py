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

import argparse
import os
import subprocess
import sys

WORKFLOWS = [
    "robotic_ultrasound",
    "robotic_ultrasound/cosmos_transfer",
    "robotic_surgery",
]

def install_dependencies(workflow_name: str = "robotic_ultrasound"):
    """Install project dependencies from requirements.txt"""
    if workflow_name not in WORKFLOWS:
        raise ValueError(f"Invalid workflow name: {workflow_name}")

    try:
        # Install test dependencies
        apt_cmd = [
            "apt-get",
            "install",
            "-y",
            "xvfb",
            "x11-utils",  # needed to run tests that need display
            "unzip",  # handle zip files
            "libusb-1.0-0",  # needed for realsense camera test
            "libegl1",
            "libxcb-icccm4",
            "libxkbcommon-x11-0",  # needed for clarius tests
        ]
        # check if the user is root
        if os.geteuid() != 0:
            apt_cmd.insert(0, "sudo")
        subprocess.check_call(apt_cmd)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage", "parameterized"])

        # Install workflow dependencies
        dir = os.path.dirname(os.path.abspath(__file__))
        if workflow_name == "robotic_ultrasound":
            subprocess.check_call(["./env_setup_robot_us.sh"], cwd=dir)
        elif workflow_name == "robotic_ultrasound/cosmos_transfer":
            subprocess.check_call(["./env_setup_cosmos_transfer1.sh"], cwd=dir)
        elif workflow_name == "robotic_surgery":
            subprocess.check_call(["./env_setup_robot_surgery.sh"], cwd=dir)

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install project dependencies")
    parser.add_argument("--workflow", type=str, default="robotic_ultrasound", help="Workflow name")
    args = parser.parse_args()
    install_dependencies(args.workflow)
