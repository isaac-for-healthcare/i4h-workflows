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

# Allow setting the python in PYTHON_EXECUTABLE
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

LEROBOT_DIR=${1:-$PROJECT_ROOT/third_party/lerobot}

echo "Installing lerobot..."
git clone https://github.com/huggingface/lerobot.git $LEROBOT_DIR
pushd $LEROBOT_DIR
git checkout 6674e368249472c91382eb54bb8501c94c7f0c56

# Update pyav dependency in pyproject.toml
sed -i 's/pyav/av/' pyproject.toml

$PYTHON_EXECUTABLE -m pip install -e .
$PYTHON_EXECUTABLE -m pip install "datasets<4.0.0"
popd

echo "Lerobot installed successfully!"
