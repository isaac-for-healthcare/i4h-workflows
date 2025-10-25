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

# Assuming this script is in tools/env_setup/
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"

# Allow setting the python in PYTHON_EXECUTABLE
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

echo "Cloning OpenPI repository..."
OPENPI_DIR=${1:-$PROJECT_ROOT/third_party/openpi}

if [ -d "$OPENPI_DIR" ]; then
    echo "OpenPI directory already exists at $OPENPI_DIR. Skipping clone."
    exit 1
else
    git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi

pushd "$OPENPI_DIR"
git checkout 581e07d73af36d336cef1ec9d7172553b2332193


# Modify the type hints in training/utils.py to use Any instead of optax types
utils_path="$OPENPI_DIR/src/openpi/training/utils.py"
echo "Patching OpenPI utils.py..."
sed -i.bak \
    -e 's/opt_state: optax\.OptState/opt_state: Any/' \
    "$utils_path"

# Remove the backup files
rm "$utils_path.bak"

# Add training script to openpi module
echo "Copying OpenPI utility scripts..."
if [ ! -f src/openpi/train.py ]; then
    cp scripts/train.py src/openpi/train.py
fi
if [ ! -f src/openpi/compute_norm_stats.py ]; then
    cp scripts/compute_norm_stats.py src/openpi/compute_norm_stats.py
fi

popd # Back to PROJECT_ROOT

echo "Installing OpenPI Client..."
$PYTHON_EXECUTABLE -m pip install -e $OPENPI_DIR/packages/openpi-client/
echo "Installing OpenPI Core..."
$PYTHON_EXECUTABLE -m pip install -e $OPENPI_DIR/


echo "PI0 Dependencies installed."
