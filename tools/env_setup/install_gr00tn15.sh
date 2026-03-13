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
DEFAULT_GR00T_COMMIT="17a77ebf646cf13460cdbc8f49f9ec7d0d63bcb1"

# Optional args
# Usage:
#   ./tools/env_setup/install_gr00tn15.sh [--commit <git-ref>] [-p|--policy-patch]
# If not provided, uses DEFAULT_GR00T_COMMIT.
GR00T_COMMIT="$DEFAULT_GR00T_COMMIT"
APPLY_POLICY_PATCH=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit|--ref)
            GR00T_COMMIT="${2:-}"
            if [[ -z "$GR00T_COMMIT" ]]; then
                echo "ERROR: $1 requires a value (commit hash / tag / branch)."
                exit 1
            fi
            shift 2
            ;;
        -p|--policy-patch|--apply-policy-patch)
            APPLY_POLICY_PATCH=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--commit <git-ref>] [-p|--policy-patch]"
            echo ""
            echo "Options:"
            echo "  --commit, --ref   Git commit hash / tag / branch for Isaac-GR00T (default: $DEFAULT_GR00T_COMMIT)"
            echo "  -p, --policy-patch  Apply local GR00T policy patch (eagle padding and replace dropout with identity.)"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

if [ -d "$GR00T_DIR" ]; then
    echo "Isaac-GR00T directory already exists at $GR00T_DIR. Using existing clone."
else
    echo "Cloning Isaac-GR00T repository into $GR00T_DIR..."
    # Ensure parent third_party dir exists
    mkdir -p "$PROJECT_ROOT/third_party"
    git clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_DIR"
fi

pushd "$GR00T_DIR"

# checkout to desired commit/tag/branch
echo "Checking out Isaac-GR00T ref: $GR00T_COMMIT"
git fetch --all --tags --prune
git checkout "$GR00T_COMMIT"

# Apply optional patch(es)
if [[ "$APPLY_POLICY_PATCH" == "true" ]]; then
    PATCH_DIR="$PROJECT_ROOT/tools/env_setup/patches"
    echo "Applying GR00T policy patch (eagle input padding and replace dropout with identity)..."
    git apply --check "$PATCH_DIR/gr00t_policy_padding_dropout.patch" 2>/dev/null && \
        git apply "$PATCH_DIR/gr00t_policy_padding_dropout.patch" || \
        echo "Patch already applied or not applicable, skipping..."
else
    echo "Skipping GR00T policy patch (pass --apply-policy-patch to enable)."
fi

pip install -e .[base]
pip3 install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
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
pip install --upgrade av==16.1.0
pip install --upgrade 'pyarrow>=17.0.0'
