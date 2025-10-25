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
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"

ULTRASOUND_RAYTRACING_DIR=${1:-$PROJECT_ROOT/third_party/i4h-sensor-simulation-internal}


# Ensure the cuda compiler nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo "Error: nvcc (CUDA compiler) could not be found. Please install CUDA Toolkit and ensure nvcc is in your PATH."
    exit 1
else
    echo "nvcc found: $(which nvcc)"
    nvcc --version
fi


echo "Cloning i4h-sensor-simulation repo..."

if [ -d "$ULTRASOUND_RAYTRACING_DIR" ]; then
    echo "Ultrasound-raytracing repo already exists at $ULTRASOUND_RAYTRACING_DIR. Skipping clone."
    exit 1
else
    git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git "$ULTRASOUND_RAYTRACING_DIR"
    pushd "$ULTRASOUND_RAYTRACING_DIR"
    git checkout v0.4.0
    popd
fi


cd $ULTRASOUND_RAYTRACING_DIR/ultrasound-raytracing

export CMAKE_BUILD_PARALLEL_LEVEL=1
export CXXFLAGS="${CXXFLAGS} -Wno-error=array-bounds"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80" pip install -e .

# copy raysim to workflows/robotic_ultrasound/scripts
cp -r $ULTRASOUND_RAYTRACING_DIR/ultrasound-raytracing/raysim $PROJECT_ROOT/workflows/robotic_ultrasound/scripts

echo "Raysim installed successfully."
