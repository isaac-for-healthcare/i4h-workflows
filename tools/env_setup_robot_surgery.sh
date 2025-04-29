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


# ---- Clone IsaacLab ----
echo "Cloning IsaacLab..."
# CLONING REPOSITORIES INTO PROJECT_ROOT/third_party
echo "Cloning repositories into $PROJECT_ROOT/third_party..."
mkdir $PROJECT_ROOT/third_party
git clone -b v2.0.2 git@github.com:isaac-sim/IsaacLab.git $PROJECT_ROOT/third_party/IsaacLab
pushd $PROJECT_ROOT/third_party/IsaacLab


# ---- Install IsaacSim and necessary dependencies ----
echo "Installing IsaacSim..."
pip install 'isaacsim[all,extscache]==4.5.0' \
    git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@main \
    --extra-index-url https://pypi.nvidia.com


echo "Installing IsaacLab ..."
yes Yes | ./isaaclab.sh --install
popd


# ---- Install robotic.surgery.assets and robotic.surgery.tasks ----
echo "Installing robotic.surgery.assets and robotic.surgery.tasks..."
pip install -e $PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.assets
pip install -e $PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks


# ---- Install libstdcxx-ng for mesa ----
echo "Installing libstdcxx-ng..."
conda install -c conda-forge libstdcxx-ng=13.2.0 -y

echo "Dependencies installed successfully!"
