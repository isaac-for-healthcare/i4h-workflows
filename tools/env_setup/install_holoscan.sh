#!/bin/bash

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

set -e

# Get the parent directory of the current script
# Assuming this script is in tools/env_setup/
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"

# Allow setting the python in PYTHON_EXECUTABLE
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

HOLOSCAN_DIR=${1:-$PROJECT_ROOT/workflows/robotic_ultrasound/scripts/holoscan_apps/}

# ---- Install Holoscan ----
$PYTHON_EXECUTABLE -m pip install holoscan==2.9.0
echo "Holoscan installed successfully!"

# ---- Install Holoscan Apps ----
echo "Building Holoscan Apps..."
pushd $HOLOSCAN_DIR

# clean previous downloads and builds
rm -rf build
rm -rf clarius_solum/include
rm -rf clarius_solum/lib
rm -rf clarius_cast/include
rm -rf clarius_cast/lib
cmake -B build -S . && cmake --build build

popd
echo "Holoscan Apps build completed!"
