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

import pyrealsense2 as rs


def count_devices():
    # Create a context object to get access to connected devices
    context = rs.context()
    devices = context.query_devices()
    return len(devices)


def list_devices():
    # Create a context object to get access to connected devices
    context = rs.context()
    devices = context.query_devices()

    # Enumerate and list all connected devices
    if len(devices) == 0:
        print("No RealSense devices connected.")
    else:
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")


if __name__ == "__main__":
    list_devices()
