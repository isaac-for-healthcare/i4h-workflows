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

# Can take EXTS_DIR from input, default to $PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts
EXTS_DIR=${1:-$PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts}

# Allow setting the python in PYTHON_EXECUTABLE
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

# ---- Install robotic.surgery.assets and robotic.surgery.tasks ----
echo "Installing robotic.surgery.assets and robotic.surgery.tasks..."
$PYTHON_EXECUTABLE -m pip install --no-build-isolation -e $EXTS_DIR/robotic.surgery.assets
$PYTHON_EXECUTABLE -m pip install --no-build-isolation -e $EXTS_DIR/robotic.surgery.tasks

echo "Extensions installed successfully!"
