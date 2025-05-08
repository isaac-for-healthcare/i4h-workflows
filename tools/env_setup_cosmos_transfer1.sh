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

# Check if NVIDIA GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found or driver not installed"
    exit 1
fi

# Check if the third_party directory exists, if not, create it
if [ ! -d "$PROJECT_ROOT/third_party" ]; then
    echo "Creating third_party directory..."
    mkdir -p $PROJECT_ROOT/third_party
else
    echo "third_party directory already exists"
fi

# ---- Clone cosmos-transfer1 ----
echo "Cloning cosmos-transfer1..."
# CLONING REPOSITORIES INTO PROJECT_ROOT/third_party
echo "Cloning repositories into $PROJECT_ROOT/third_party..."
git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git $PROJECT_ROOT/third_party/cosmos-transfer1
pushd $PROJECT_ROOT/third_party/cosmos-transfer1
git checkout bf54a70a8c44d615620728c493ee26b4376ccfd6
git submodule update --init --recursive

# Create the cosmos-transfer1 conda environment.
conda env create --file cosmos-transfer1.yaml -y

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-transfer1

# Check if running in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No active conda environment detected"
    echo "Please activate a conda environment before running this script"
    exit 1
fi
echo "Using conda environment: $CONDA_DEFAULT_ENV"

# Install the dependencies.
pip install -r requirements.txt
# Install cuDNN.
conda install conda-forge::cudnn=9.8.0.87 -y
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
pip install h5py

# Test the environment.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py