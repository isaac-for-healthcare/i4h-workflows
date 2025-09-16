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

# Ensure the parent third_party directory exists
mkdir -p "$PROJECT_ROOT/third_party"

# ---- Install IsaacSim ----
echo "Installing IsaacSim..."
pip install 'isaacsim[all,extscache]==4.5.0' \
    git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.3.0 \
    --extra-index-url https://pypi.nvidia.com

ISAACLAB_DIR="$PROJECT_ROOT/third_party/IsaacLab"

if [ -d "$ISAACLAB_DIR" ]; then
    echo "IsaacLab directory already exists at $ISAACLAB_DIR. Skipping clone. Will use existing."
else
    echo "Cloning IsaacLab repository into $ISAACLAB_DIR..."
    git clone -b v2.1.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

# ---- Apply events_random_texture.patch patch to IsaacLab ----
# https://github.com/isaac-sim/IsaacLab/pull/2476
EVENTS_PATCH_SRC_FILE="$PROJECT_ROOT/tools/env_setup/patches/events_random_texture.patch"

if [ -f "$EVENTS_PATCH_SRC_FILE" ]; then
    echo "Applying events_random_texture patch ($EVENTS_PATCH_SRC_FILE) to IsaacLab at $ISAACLAB_DIR..."
    # -p1 strips the first component (e.g., a/ or b/) from paths in the patch file.
    (cd "$ISAACLAB_DIR" && patch -p1 < "$EVENTS_PATCH_SRC_FILE")
    if [ $? -eq 0 ]; then
        echo "Events random_texture patch applied successfully."
    else
        echo "Error applying events random_texture patch. Please check for conflicts or errors above." >&2
    fi
else
    echo "Error: Events random_texture patch file ($EVENTS_PATCH_SRC_FILE) not found. Skipping this patch." >&2
fi

# ---- Apply semantic tags patch to IsaacLab ----
# https://github.com/isaac-sim/IsaacLab/pull/2410
FROM_FILES_PATCH_SRC_FILE="$PROJECT_ROOT/tools/env_setup/patches/from_files_semantic_tags.patch"

if [ -f "$FROM_FILES_PATCH_SRC_FILE" ]; then
    echo "Applying semantic tags patch ($FROM_FILES_PATCH_SRC_FILE) to IsaacLab at $ISAACLAB_DIR..."
    (cd "$ISAACLAB_DIR" && patch -p1 < "$FROM_FILES_PATCH_SRC_FILE")
    if [ $? -eq 0 ]; then
        echo "Semantic tags patch applied successfully."
    else
        echo "Error applying semantic tags patch. Please check for conflicts or errors above." >&2
    fi
else
    echo "Error: Semantic tags patch file ($FROM_FILES_PATCH_SRC_FILE) not found. Skipping this patch." >&2
fi

pushd "$ISAACLAB_DIR"
echo "Installing IsaacLab ..."
yes Yes | ./isaaclab.sh --install
popd

# ---- Apply dependency package version patch ----
pip install 'warp-lang==1.7.2'

echo "IsaacSim and dependencies installed successfully!"
