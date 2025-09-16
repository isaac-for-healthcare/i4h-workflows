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

# Assuming bash_utils.sh is in $PROJECT_ROOT/tools/env_setup/bash_utils.sh
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

check_project_root

echo "--- Installing GR00T N1.5 Policy Dependencies ---"

GR00T_DIR="$PROJECT_ROOT/third_party/Isaac-GR00T"

if [ -d "$GR00T_DIR" ]; then
    echo "Isaac-GR00T directory already exists at $GR00T_DIR. Using existing clone."
else
    echo "Cloning Isaac-GR00T repository into $GR00T_DIR..."
    # Ensure parent third_party dir exists
    mkdir -p "$PROJECT_ROOT/third_party"
    git clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_DIR"
fi

pushd "$GR00T_DIR"

# checkout to used commit
git checkout 17a77ebf646cf13460cdbc8f49f9ec7d0d63bcb1

pip install -e .[base]
popd


# Install flash-attn with optimized wheel download
echo "Installing flash-attn..."

FLASH_ATTN_VERSION="2.7.4.post1"
# Detect Python version
PLATFORM=$(uname -m)

if [ "$PLATFORM" == "x86_64" ]; then
    PYTHON_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    echo "Detected Python version: $PYTHON_VERSION"

    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.6")
    TORCH_MAJOR_MINOR=$(echo "$TORCH_VERSION" | cut -d'.' -f1,2)
    echo "Detected PyTorch version: $TORCH_VERSION (using $TORCH_MAJOR_MINOR for wheel)"

    # Define flash-attn version and wheel URL
    WHEEL_FILE=flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_MAJOR_MINOR}cxx11abiFALSE-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
    if wget https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL_FILE}; then
        echo "Successfully downloaded pre-built wheel. Installing..."
        pip install ${WHEEL_FILE}
        rm -f ${WHEEL_FILE}
    else
        echo "Failed to download pre-built wheel. Falling back to pip install..."
        pip install --no-build-isolation flash-attn==${FLASH_ATTN_VERSION}
    fi
else
    pip install --no-build-isolation flash-attn==${FLASH_ATTN_VERSION}
fi

echo "GR00T N1.5 Policy Dependencies installed."

# resolve hdf5 to lerobot conflicts
pip install av==14.4.0
