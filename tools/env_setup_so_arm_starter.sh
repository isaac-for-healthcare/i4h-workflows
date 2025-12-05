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
    sudo apt-get update
    sudo apt-get install -y cmake git build-essential pybind11-dev libxcb-cursor0
else
    apt-get update
    apt-get install -y cmake git build-essential pybind11-dev libxcb-cursor0
fi

# ---- Install necessary dependencies (Common) ----
echo "Installing necessary dependencies..."
pip install rti.connext==7.3.0 pyrealsense2==2.55.1.6486 toml==0.10.2 dearpygui==2.0.0 \
    setuptools==75.8.0 pydantic==2.10.6 matplotlib scipy

# ---- Install IsaacSim and IsaacLab (Common) ----
# Check if IsaacLab is already cloned
echo "Installing IsaacSim and IsaacLab..."
# Set versions based on architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected aarch64 architecture. This is for DGX Spark. Use IsaacSim 5.1 and IsaacLab 2.3."
    bash $PROJECT_ROOT/tools/env_setup/install_isaacsim5.1_isaaclab2.3.sh
else
    echo "Detected $ARCH architecture. Use IsaacSim 5.0 and IsaacLab 2.2"
    bash $PROJECT_ROOT/tools/env_setup/install_isaacsim5.0_isaaclab2.2.sh
fi

# ---- Install leisaac (Common) ----
echo "Installing leisaac..."
LEISAAC_DIR=${1:-$PROJECT_ROOT/third_party/leisaac}
LEISAAC_VERSION="v0.2.0"
git clone --branch $LEISAAC_VERSION --depth 1 https://github.com/LightwheelAI/leisaac.git $LEISAAC_DIR
pushd $LEISAAC_DIR
pip install -e source/leisaac
popd

# ---- Install lerobot (Common) ----
echo "Installing lerobot..."
LEROBOT_DIR=${1:-$PROJECT_ROOT/third_party/lerobot}
git clone https://github.com/huggingface/lerobot.git $LEROBOT_DIR
pushd $LEROBOT_DIR
git checkout 483be9aac217c2d8ef16982490f22b2ad091ab46
pip install -e ".[feetech]"
popd

# ---- Install gr00tn1.5 (Common) ----
echo "Installing gr00t n1.5..."
bash $PROJECT_ROOT/tools/env_setup/install_gr00tn15.sh

# ---- install so_arm_starter_ext ----
bash $PROJECT_ROOT/tools/env_setup/install_so_arm_starter_extensions.sh

# ---- Install tensorrt ----
bash $PROJECT_ROOT/tools/env_setup/install_tensorrt.sh

# ---- Install Holoscan ----
bash $PROJECT_ROOT/tools/env_setup/install_holoscan_3.5.0.sh


echo "=========================================="
echo "Environment setup script finished."
echo "=========================================="
