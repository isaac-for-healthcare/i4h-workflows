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

# --- Utility Functions ---

check_conda_env() {
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo "Error: No active conda environment detected"
        echo ""
        echo "To resolve this, you can either:"
        echo "1. Install miniconda if you don't have it:"
        echo "   - Download from: https://docs.conda.io/en/latest/miniconda.html"
        echo "   - Or run: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
        echo ""
        echo "2. If conda is already installed, initialize it:"
        echo "   - Run: conda init"
        echo ""
        echo "3. Create and activate a conda environment:"
        echo "   - Run: conda create -n i4h python=3.10"
        echo "   - Then: conda activate i4h"
        echo ""
        exit 1
    fi
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
}

check_nvidia_gpu() {
    # skip check if building docker image
    if [ "$BUILD_DOCKER_IMAGE" = "true" ]; then
        return
    fi

    if ! nvidia-smi &> /dev/null; then
        echo "Error: NVIDIA GPU not found or driver not installed"
        exit 1
    fi
    echo "NVIDIA GPU detected."
}

check_project_root() {
    if [ -z "$PROJECT_ROOT" ]; then
        echo "Error: PROJECT_ROOT is not set. This script should be sourced by a parent script that defines PROJECT_ROOT."
        exit 1
    fi
    echo "PROJECT_ROOT is set to: $PROJECT_ROOT"
}

ensure_fresh_third_party_dir() {
    check_project_root # Ensure PROJECT_ROOT is available
    THIRDPARTY_DIR="$PROJECT_ROOT/third_party"

    if [ -d "$THIRDPARTY_DIR" ]; then
        echo "Error: third_party directory already exists at $THIRDPARTY_DIR"
        echo "Please remove the third_party directory before running this script with policies that require a fresh setup (e.g., pi0, gr00tn1)."
        exit 1
    else
        mkdir "$THIRDPARTY_DIR"
        echo "Created directory: $THIRDPARTY_DIR"
    fi
}
