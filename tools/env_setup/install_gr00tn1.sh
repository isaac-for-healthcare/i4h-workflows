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
check_conda_env

echo "--- Installing GR00T N1 Policy Dependencies ---"

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

# Update pyav dependency in pyproject.toml if not already done
if grep -q "pyav" pyproject.toml; then
    echo "Updating pyav to av in Isaac-GR00T's pyproject.toml..."
    sed -i 's/pyav/av/' pyproject.toml
else
    echo "pyav already updated to av or not found in Isaac-GR00T's pyproject.toml."
fi

pip install -e .
popd

pip install --no-build-isolation flash-attn==2.7.1.post4

echo "GR00T N1 Policy Dependencies installed."
