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

# Allow setting the python in PYTHON_EXECUTABLE
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

EXTS_DIR=${1:-$PROJECT_ROOT/workflows/robotic_ultrasound/scripts/simulation}

echo "--- Installing Robotic Ultrasound Extensions and Dependencies ---"

# ---- Install robotic ultrasound extension ----
echo "Installing actual robotic ultrasound extension..."
pushd "$EXTS_DIR"
# Ensure the target directory exists before installing
if [ -d "exts/robotic_us_ext" ]; then
    $PYTHON_EXECUTABLE -m pip install -e exts/robotic_us_ext
else
    echo "Error: robotic_us_ext directory not found in ${EXTS_DIR}/exts/"
    exit 1 # Exit if not found
fi
popd

echo "--- Robotic Ultrasound Extensions and Dependencies Installation Finished ---"
