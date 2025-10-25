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
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

# Check if running in a conda environment
check_conda_env

# Check if NVIDIA GPU is available
check_nvidia_gpu

# Check if the third_party directory exists, if yes, then exit
ensure_fresh_third_party_dir

# Install tools and dependencies
pip install setuptools==75.8.0 toml==0.10.2

# Run the installation scripts
echo "Installing IsaacSim and dependencies..."
bash $PROJECT_ROOT/tools/env_setup/install_isaacsim5.0_isaaclab2.3.sh

echo "Installing extensions..."
bash $PROJECT_ROOT/tools/env_setup/install_robotic_surgery_extensions.sh

echo "All dependencies installed successfully!"
