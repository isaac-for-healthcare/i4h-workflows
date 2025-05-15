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

# Source utility functions
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

# Check if running in a conda environment
check_conda_env

# Check if NVIDIA GPU is available
check_nvidia_gpu

# Check if the third_party directory exists, if yes, then exit
ensure_fresh_third_party_dir

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
    git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.1.0 \
    --extra-index-url https://pypi.nvidia.com

echo "Installing IsaacLab ..."
yes Yes | ./isaaclab.sh --install
popd

# ---- Install libstdcxx-ng for mesa ----
echo "Installing libstdcxx-ng..."
conda install -c conda-forge libstdcxx-ng=13.2.0 -y

echo "IsaacSim and dependencies installed successfully!"
