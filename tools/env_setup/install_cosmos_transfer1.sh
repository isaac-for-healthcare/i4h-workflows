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

# Source utility functions
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

check_project_root

conda install -c conda-forge ninja libgl ffmpeg gcc=12.4.0 gxx=12.4.0 -y

COSMOS_TRANSFER_DIR="$PROJECT_ROOT/third_party/cosmos-transfer1"

if [ -d "$COSMOS_TRANSFER_DIR" ]; then
    echo "Cosmos Transfer directory already exists at $COSMOS_TRANSFER_DIR. Skipping clone."
else
    echo "Cloning Cosmos Transfer repository into $COSMOS_TRANSFER_DIR..."
    git clone git@github.com:nvidia-cosmos/cosmos-transfer1.git "$COSMOS_TRANSFER_DIR"
fi

pushd "$COSMOS_TRANSFER_DIR"
git checkout bf54a70a8c44d615620728c493ee26b4376ccfd6
git submodule update --init --recursive
pip install -r requirements.txt
conda install cuda -c nvidia -y
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]==1.12.0
pip install tensorstore==0.1.74

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
popd

echo "Cosmos Transfer Dependencies Installation Finished"
