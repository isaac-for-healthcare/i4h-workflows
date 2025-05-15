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

# Check if running in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No active conda environment detected"
    echo "Please activate a conda environment before running this script"
    exit 1
fi
echo "Using conda environment: $CONDA_DEFAULT_ENV"

# Check if NVIDIA GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found or driver not installed"
    exit 1
fi

# Check if the third_party directory exists
if [ -d "$PROJECT_ROOT/third_party" ]; then
    echo "Error: third_party directory already exists"
    echo "Please remove the third_party directory before running this script"
    exit 1
else
    mkdir $PROJECT_ROOT/third_party
    echo "Created directory: $PROJECT_ROOT/third_party"
fi


# ---- Install build tools (Common) ----
echo "Installing build tools..."
if [ "$EUID" -ne 0 ]; then
    sudo apt-get install -y git cmake build-essential pybind11-dev libxcb-cursor0
else
    apt-get install -y git cmake build-essential pybind11-dev libxcb-cursor0
fi


# ---- Install IsaacSim and necessary dependencies (Common) ----
echo "Installing IsaacSim and base dependencies..."
pip install 'isaacsim[all,extscache]==4.5.0' \
    rti.connext==7.3.0 pyrealsense2==2.55.1.6486 toml==0.10.2 dearpygui==2.0.0 \
    git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.1.0 \
    setuptools==75.8.0 pydantic==2.10.6 \
    --extra-index-url https://pypi.nvidia.com


# ---- Install IsaacLab (Common) ----
# Check if IsaacLab is already cloned
echo "Installing IsaacLab..."
echo "Cloning IsaacLab repository into $PROJECT_ROOT/third_party/IsaacLab..."
git clone -b v2.0.2 git@github.com:isaac-sim/IsaacLab.git $PROJECT_ROOT/third_party/IsaacLab
pushd $PROJECT_ROOT/third_party/IsaacLab
yes Yes | ./isaaclab.sh --install
popd


# ---- Install robotic ultrasound extension (Common) ----
echo "Installing robotic ultrasound extension..."
pushd $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/simulation
# Ensure the target directory exists before installing
if [ -d "exts/robotic_us_ext" ]; then
    pip install -e exts/robotic_us_ext
else
    echo "Error: robotic_us_ext directory not found in $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/simulation/exts/"
    exit 1
fi
popd

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
    echo "------------------------------------------"
    echo "Installing PI0 Policy Dependencies..."
    echo "------------------------------------------"

    echo "Cloning OpenPI repository..."
    git clone git@github.com:Physical-Intelligence/openpi.git $PROJECT_ROOT/third_party/openpi
    pushd $PROJECT_ROOT/third_party/openpi
    git checkout 581e07d73af36d336cef1ec9d7172553b2332193

    # Update python version in pyproject.toml
    pyproject_path="$PROJECT_ROOT/third_party/openpi/pyproject.toml"
    echo "Patching OpenPI pyproject.toml..."
    sed -i.bak \
        -e 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' \
        -e 's/"s3fs>=2024.9.0"/"s3fs==2024.9.0"/' \
        "$pyproject_path"

    # Apply temporary workaround for openpi/src/openpi/shared/download.py
    file_path="$PROJECT_ROOT/third_party/openpi/src/openpi/shared/download.py"
    echo "Patching OpenPI download.py..."
    # Comment out specific import lines
    sed -i.bak \
        -e 's/^import boto3\.s3\.transfer as s3_transfer/# import boto3.s3.transfer as s3_transfer/' \
        -e 's/^import s3transfer\.futures as s3_transfer_futures/# import s3transfer.futures as s3_transfer_futures/' \
        -e 's/^from types_boto3_s3\.service_resource import ObjectSummary/# from types_boto3_s3.service_resource import ObjectSummary/' \
        "$file_path"
    # Remove the type hint
    sed -i.bak -e 's/)[[:space:]]*-> s3_transfer\.TransferManager[[:space:]]*:/):/' "$file_path"
    # Modify the datetime line
    sed -i.bak -e 's/datetime\.UTC/datetime.timezone.utc/' "$file_path"

    # Modify the type hints in training/utils.py to use Any instead of optax types
    utils_path="$PROJECT_ROOT/third_party/openpi/src/openpi/training/utils.py"
    echo "Patching OpenPI utils.py..."
    sed -i.bak \
        -e 's/opt_state: optax\.OptState/opt_state: Any/' \
        "$utils_path"

    # Remove the backup files
    rm "$pyproject_path.bak" "$file_path.bak" "$utils_path.bak"

    # Add training script to openpi module
    echo "Copying OpenPI utility scripts..."
    if [ ! -f src/openpi/train.py ]; then
        cp scripts/train.py src/openpi/train.py
    fi
    if [ ! -f src/openpi/compute_norm_stats.py ]; then
        cp scripts/compute_norm_stats.py src/openpi/compute_norm_stats.py
    fi

    popd # Back to PROJECT_ROOT

    echo "Installing OpenPI Client..."
    pip install -e $PROJECT_ROOT/third_party/openpi/packages/openpi-client/
    echo "Installing OpenPI Core..."
    pip install -e $PROJECT_ROOT/third_party/openpi/

    # Revert the "import changes of "$file_path after installation to prevent errors
    echo "Reverting temporary patches in OpenPI download.py..."
    file_path_revert="$PROJECT_ROOT/third_party/openpi/src/openpi/shared/download.py"
    sed -i \
        -e 's/^# import boto3\.s3\.transfer as s3_transfer/import boto3.s3.transfer as s3_transfer/' \
        -e 's/^# import s3transfer\.futures as s3_transfer_futures/import s3transfer.futures as s3_transfer_futures/' \
        -e 's/^# from types_boto3_s3\.service_resource import ObjectSummary/from types_boto3_s3.service_resource import ObjectSummary/' \
        "$file_path_revert"
    echo "PI0 Dependencies installed."
fi


# ---- Install GR00T N1 Policy Dependencies (Conditional) ----
if [[ "$INSTALL_WITH_POLICY" == "gr00tn1" ]]; then
    echo "Installing GR00T N1 Policy Dependencies..."
    git clone https://github.com/NVIDIA/Isaac-GR00T $PROJECT_ROOT/third_party/Isaac-GR00T
    pushd $PROJECT_ROOT/third_party/Isaac-GR00T
    sed -i 's/pyav/av/' pyproject.toml
    pip install -e .
    popd
    pip install --no-build-isolation flash-attn==2.7.1.post4
    echo "GR00T N1 Policy Dependencies installed."
fi


# ---- Install Holoscan (Common) ----
echo "------------------------------------------"
echo "Installing Holoscan..."
echo "------------------------------------------"
conda install -c conda-forge gcc=13.3.0 -y
pip install holoscan==2.9.0

HOLOSCAN_DIR=$PROJECT_ROOT/workflows/robotic_ultrasound/scripts/holoscan_apps/

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

# ---- Install dependencies for cosmos transfer (Common) ----
echo "------------------------------------------"
echo "Installing cosmos transfer dependencies..."
echo "------------------------------------------"
conda install -c conda-forge ninja libgl ffmpeg gcc=12.4.0 gxx=12.4.0 -y
git clone git@github.com:nvidia-cosmos/cosmos-transfer1.git $PROJECT_ROOT/third_party/cosmos-transfer1
pushd $PROJECT_ROOT/third_party/cosmos-transfer1
git checkout bf54a70a8c44d615620728c493ee26b4376ccfd6
git submodule update --init --recursive
pip install -r requirements.txt
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]==1.12.0
popd


# ---- Install libstdcxx-ng for raysim (Common) ----
echo "------------------------------------------"
echo "Installing libstdcxx-ng..."
echo "------------------------------------------"
conda install -c conda-forge libstdcxx-ng=13.2.0 -y

echo "=========================================="
echo "Environment setup script finished."
echo "Selected policy dependencies ($INSTALL_WITH_POLICY) should be installed along with base components."
echo "=========================================="