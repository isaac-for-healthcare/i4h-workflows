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

# Check if the third_party directory exists, if yes, then exit
if [ -d "$PROJECT_ROOT/third_party" ]; then
    echo "Error: third_party directory already exists"
    echo "Please remove the third_party directory before running this script"
    exit 1
fi

# ---- Install build tools ----
echo "Installing build tools..."
if [ "$EUID" -ne 0 ]; then
    sudo apt-get install -y cmake build-essential pybind11-dev libxcb-cursor0
else
    apt-get install -y cmake build-essential pybind11-dev libxcb-cursor0
fi


# ---- Install IsaacSim and necessary dependencies ----
echo "Installing IsaacSim..."
pip install 'isaacsim[all,extscache]==4.5.0' \
    rti.connext==7.3.0 pyrealsense2==2.55.1.6486 toml==0.10.2 dearpygui==2.0.0 \
    git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@mz/isaacsim45 \
    setuptools==75.8.0 pydantic==2.10.6 \
    --extra-index-url https://pypi.nvidia.com


# ---- Install IsaacLab ----
echo "Installing IsaacLab..."
# CLONING REPOSITORIES INTO PROJECT_ROOT/third_party
echo "Cloning repositories into $PROJECT_ROOT/third_party..."
mkdir $PROJECT_ROOT/third_party
git clone -b v2.0.2 git@github.com:isaac-sim/IsaacLab.git $PROJECT_ROOT/third_party/IsaacLab
pushd $PROJECT_ROOT/third_party/IsaacLab
yes Yes | ./isaaclab.sh --install
popd


# ---- Install robotic ultrasound extension ----
echo "Installing robotic ultrasound extension..."
pushd $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/simulation
pip install -e exts/robotic_us_ext
popd


# ---- Install OpenPI with IsaacSim 4.2 ----
echo "Installing OpenPI..."
# Clone the openpi repository
git clone git@github.com:Physical-Intelligence/openpi.git $PROJECT_ROOT/third_party/openpi
pushd $PROJECT_ROOT/third_party/openpi
git checkout 581e07d73af36d336cef1ec9d7172553b2332193

# Update python version in pyproject.toml
pyproject_path="$PROJECT_ROOT/third_party/openpi/pyproject.toml"
sed -i.bak \
    -e 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' \
    -e 's/"s3fs>=2024.9.0"/"s3fs==2024.9.0"/' \
    "$pyproject_path"

# Apply temporary workaround for openpi/src/openpi/shared/download.py
file_path="$PROJECT_ROOT/third_party/openpi/src/openpi/shared/download.py"

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
sed -i.bak \
    -e 's/opt_state: optax\.OptState/opt_state: Any/' \
    "$utils_path"

# Remove the backup files
rm "$pyproject_path.bak"
rm "$file_path.bak"
rm "$utils_path.bak"

# Add training script to openpi module
if [ ! -f $PROJECT_ROOT/third_party/openpi/src/openpi/train.py ]; then
    cp $PROJECT_ROOT/third_party/openpi/scripts/train.py $PROJECT_ROOT/third_party/openpi/src/openpi/train.py
fi

# Add norm stats generator script to openpi module
if [ ! -f $PROJECT_ROOT/third_party/openpi/src/openpi/compute_norm_stats.py ]; then
    cp $PROJECT_ROOT/third_party/openpi/scripts/compute_norm_stats.py $PROJECT_ROOT/third_party/openpi/src/openpi/compute_norm_stats.py
fi

# Install the dependencies
pip install git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56
pip install -e $PROJECT_ROOT/third_party/openpi/packages/openpi-client/
pip install -e $PROJECT_ROOT/third_party/openpi/

# Revert the "import changes of "$file_path after installation to prevent errors
sed -i \
    -e 's/^# import boto3\.s3\.transfer as s3_transfer/import boto3.s3.transfer as s3_transfer/' \
    -e 's/^# import s3transfer\.futures as s3_transfer_futures/import s3transfer.futures as s3_transfer_futures/' \
    -e 's/^# from types_boto3_s3\.service_resource import ObjectSummary/from types_boto3_s3.service_resource import ObjectSummary/' \
    "$file_path"

popd

# ---- Install Holoscan ----
# Install Holoscan
echo "Installing Holoscan..."
conda install -c conda-forge gcc=13.3.0 -y
pip install holoscan==2.9.0

echo "Dependencies installed successfully!"

HOLOSCAN_DIR=$PROJECT_ROOT/workflows/robotic_ultrasound/scripts/holoscan_apps/

echo "Building Holoscan Apps"

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

# ---- Install libstdcxx-ng for raysim ----
echo "Installing libstdcxx-ng..."
conda install -c conda-forge libstdcxx-ng=13.2.0 -y

echo "Dependencies installed successfully!"
