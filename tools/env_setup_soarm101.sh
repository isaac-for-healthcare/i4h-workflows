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


# --- Setup Steps ---
# Get the parent directory of the current script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

# Check if running in a conda environment
check_conda_env

# Check if NVIDIA GPU is available
check_nvidia_gpu

# Check if the third_party directory exists
ensure_fresh_third_party_dir


# ---- Install build tools (Common) ----
echo "Installing build tools..."
if [ "$EUID" -ne 0 ]; then
    sudo apt-get install -y cmake
    sudo apt-get update
    sudo apt-get install -y git build-essential pybind11-dev libxcb-cursor0
else
    apt-get install -y cmake
    apt-get update
    apt-get install -y git build-essential pybind11-dev libxcb-cursor0
fi


# ---- Install necessary dependencies (Common) ----
echo "Installing necessary dependencies..."
pip install rti.connext==7.3.0 pyrealsense2==2.55.1.6486 toml==0.10.2 dearpygui==2.0.0 \
    setuptools==75.8.0 pydantic==2.10.6 matplotlib scipy\
    --extra-index-url https://pypi.nvidia.com


# ---- Install IsaacSim and IsaacLab (Common) ----
# Check if IsaacLab is already cloned
echo "Installing IsaacSim and IsaacLab..."
bash $PROJECT_ROOT/tools/env_setup/install_isaac.sh


echo "=========================================="
echo "Environment setup script finished."
echo "=========================================="
