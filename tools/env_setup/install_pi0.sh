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

# Source utility functions
# Assuming bash_utils.sh is in $PROJECT_ROOT/tools/env_setup/bash_utils.sh
source "$PROJECT_ROOT/tools/env_setup/bash_utils.sh"

# Perform necessary checks
check_project_root
check_conda_env # Ensure conda environment is still active

echo "--- Installing PI0 Policy Dependencies ---" # Combined echo

OPENPI_DIR="$PROJECT_ROOT/third_party/openpi"

if [ -d "$OPENPI_DIR" ]; then
    echo "OpenPI directory already exists at $OPENPI_DIR. Using existing clone."
    # Optionally, you might want to add a git pull or reset --hard to a specific commit here
    # For now, we assume if it exists, its state is managed or acceptable for re-running patches/install.
else
    echo "Cloning OpenPI repository into $OPENPI_DIR..."
    # Ensure parent third_party dir exists
    mkdir -p "$PROJECT_ROOT/third_party"
    git clone git@github.com:Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi

pushd "$OPENPI_DIR"
# Checkout the specific commit regardless of whether it was just cloned or already existed
echo "Ensuring OpenPI is on commit 581e07d..."
git fetch origin --tags # Ensure commit is available
git checkout 581e07d73af36d336cef1ec9d7172553b2332193

# Update python version in pyproject.toml
pyproject_path="$OPENPI_DIR/pyproject.toml"
echo "Patching OpenPI pyproject.toml ($pyproject_path)..."
sed -i.bak \
    -e 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' \
    -e 's/"s3fs>=2024.9.0"/"s3fs==2024.9.0"/' \
    "$pyproject_path"

# Apply temporary workaround for openpi/src/openpi/shared/download.py
file_path="$OPENPI_DIR/src/openpi/shared/download.py"
echo "Patching OpenPI download.py ($file_path)..."
# Comment out specific import lines
sed -i.bak \
    -e 's/^import boto3\.s3\.transfer as s3_transfer/# import boto3.s3.transfer as s3_transfer/' \
    -e 's/^import s3transfer\.futures as s3_transfer_futures/# import s3transfer.futures as s3_transfer_futures/' \
    -e 's/^from types_boto3_s3\.service_resource import ObjectSummary/# from types_boto3_s3.service_resource import ObjectSummary/' \
    "$file_path"
# Remove the type hint
sed -i.bak -e 's/)[[:space:]]*-> s3_transfer\.TransferManager[[:space:]]*:/):/' "$file_path"
# Modify the datetime line
sed -i.bak -e 's/datetime\.UTC/datetime.timezone.utc/' "$file_path"

# Modify the type hints in training/utils.py to use Any instead of optax types
utils_path="$OPENPI_DIR/src/openpi/training/utils.py"
echo "Patching OpenPI utils.py ($utils_path)..."
sed -i.bak \
    -e 's/opt_state: optax\.OptState/opt_state: Any/' \
    "$utils_path"

# Remove the backup files if they exist
rm -f "$pyproject_path.bak" "$file_path.bak" "$utils_path.bak"

# Add training script to openpi module
echo "Copying OpenPI utility scripts..."
if [ ! -f src/openpi/train.py ]; then
    cp scripts/train.py src/openpi/train.py
fi
if [ ! -f src/openpi/compute_norm_stats.py ]; then
    cp scripts/compute_norm_stats.py src/openpi/compute_norm_stats.py
fi

popd

echo "Installing OpenPI Client..."
pip install -e "$OPENPI_DIR/packages/openpi-client/"
echo "Installing OpenPI Core..."
pip install -e "$OPENPI_DIR/"

# Revert the "import changes of "$file_path after installation to prevent errors
echo "Reverting temporary patches in OpenPI download.py ($file_path)..."
sed -i \
    -e 's/^# import boto3\.s3\.transfer as s3_transfer/import boto3.s3.transfer as s3_transfer/' \
    -e 's/^# import s3transfer\.futures as s3_transfer_futures/import s3transfer.futures as s3_transfer_futures/' \
    -e 's/^# from types_boto3_s3\.service_resource import ObjectSummary/from types_boto3_s3.service_resource import ObjectSummary/' \
    "$file_path"
echo "PI0 Dependencies installed."
