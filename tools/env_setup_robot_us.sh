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

# --- Configuration ---
INSTALL_WITH_POLICY="pi0" # Default value

# --- Helper Functions ---
usage() {
    echo "Usage: $0 --policy [pi0|gr00tn1|none]"
    echo "  pi0:   Install base dependencies + PI0 policy dependencies (openpi)."
    echo "  gr00tn1: Install base dependencies + GR00T N1 policy dependencies (Isaac-GR00T)."
    echo "  none:  Install only base dependencies (IsaacSim, IsaacLab, Holoscan, etc.)."
    exit 1
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --policy)
        INSTALL_WITH_POLICY="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        usage
        ;;
    esac
done

# Validate policy argument
if [[ "$INSTALL_WITH_POLICY" != "pi0" && "$INSTALL_WITH_POLICY" != "gr00tn1" && "$INSTALL_WITH_POLICY" != "none" ]]; then
    echo "Error: Invalid policy specified."
    usage
fi

echo "Selected policy setup: $INSTALL_WITH_POLICY"


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
    sudo apt-get install -y git cmake build-essential pybind11-dev libxcb-cursor0
else
    apt-get install -y git cmake build-essential pybind11-dev libxcb-cursor0
fi


# ---- Install necessary dependencies (Common) ----
echo "Installing necessary dependencies..."
pip install rti.connext==7.3.0 pyrealsense2==2.55.1.6486 toml==0.10.2 dearpygui==2.0.0 \
    setuptools==75.8.0 pydantic==2.10.6 \
    --extra-index-url https://pypi.nvidia.com


# ---- Install IsaacSim and IsaacLab (Common) ----
# Check if IsaacLab is already cloned
echo "Installing IsaacSim and IsaacLab..."
bash $PROJECT_ROOT/tools/env_setup/install_isaac.sh


# ---- Install Robotic Ultrasound Extensions and Dependencies ----
echo "Installing Robotic Ultrasound Extensions and Dependencies..."
bash "$PROJECT_ROOT/tools/env_setup/install_robotic_us_ext.sh"


# ---- Install lerobot (Common) ----
echo "Installing lerobot..."
git clone https://github.com/huggingface/lerobot.git $PROJECT_ROOT/third_party/lerobot
pushd $PROJECT_ROOT/third_party/lerobot
git checkout 6674e368249472c91382eb54bb8501c94c7f0c56

# Update pyav dependency in pyproject.toml
sed -i 's/pyav/av/' pyproject.toml

pip install -e .
popd


# ---- Install PI0 Policy Dependencies (Conditional) ----
if [[ "$INSTALL_WITH_POLICY" == "pi0" ]]; then
    echo "Installing PI0 Policy Dependencies..."
    bash "$PROJECT_ROOT/tools/env_setup/install_pi0.sh"
fi


# ---- Install GR00T N1 Policy Dependencies (Conditional) ----
if [[ "$INSTALL_WITH_POLICY" == "gr00tn1" ]]; then
    echo "Installing GR00T N1 Policy Dependencies (delegating to script)..."
    bash "$PROJECT_ROOT/tools/env_setup/install_gr00tn1.sh"
fi


# for holoscan and cosmos transfer1, we need to install the following conda packages:
conda install -c conda-forge ninja libgl ffmpeg pybind11 gcc=12.4.0 gxx=12.4.0 libstdcxx-ng=12.4.0 -y


# ---- Install Holoscan (Common) ----
echo "Installing Holoscan..."
bash "$PROJECT_ROOT/tools/env_setup/install_holoscan.sh"

# ---- Install Cosmos (Common) ----
echo "Installing Cosmos..."
bash "$PROJECT_ROOT/tools/env_setup/install_cosmos_transfer1.sh"

echo "=========================================="
echo "Environment setup script finished."
echo "=========================================="
