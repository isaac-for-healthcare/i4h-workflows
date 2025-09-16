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

# Update python version in pyproject.toml
pyproject_path="$OPENPI_DIR/pyproject.toml"
echo "Patching OpenPI pyproject.toml..."
sed -i.bak \
    -e 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' \
    -e 's/"s3fs>=2024.9.0"/"s3fs==2024.9.0"/' \
    "$pyproject_path"

# Apply temporary workaround for openpi/src/openpi/shared/download.py
file_path="$OPENPI_DIR/src/openpi/shared/download.py"
echo "Patching OpenPI download.py..."
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
echo "Patching OpenPI utils.py..."
sed -i.bak \
    -e 's/opt_state: optax\.OptState/opt_state: Any/' \
    "$utils_path"

# Remove the backup files
rm "$pyproject_path.bak" "$file_path.bak" "$utils_path.bak"

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

# Revert the "import changes of "$file_path after installation to prevent errors
echo "Reverting temporary patches in OpenPI download.py..."
file_path_revert="$OPENPI_DIR/src/openpi/shared/download.py"
sed -i \
    -e 's/^# import boto3\.s3\.transfer as s3_transfer/import boto3.s3.transfer as s3_transfer/' \
    -e 's/^# import s3transfer\.futures as s3_transfer_futures/import s3transfer.futures as s3_transfer_futures/' \
    -e 's/^# from types_boto3_s3\.service_resource import ObjectSummary/from types_boto3_s3.service_resource import ObjectSummary/' \
    "$file_path_revert"

if $PYTHON_EXECUTABLE -m pip show isaacsim >/dev/null 2>&1; then
    # Version pinning to allow pi0 policy only work with isaacsim 4.5.0
    # Version 5.0.0 should not need to pin these dependencies
    $PYTHON_EXECUTABLE -m pip install \
        "aiobotocore==2.23.0" \
        "botocore==1.38.27" \
        "boto3==1.38.27" \
        "botocore-stubs==1.38.46" \
        "types-boto3==1.38.27" \
        "types-boto3-s3==1.38.44" \
        "s3transfer==0.13.0" \
        "types-s3transfer==0.13.0" \
        "pydantic==2.9.2"

    echo "isaacsim found. Adding DEFAULT_CHECKSUM_ALGORITHM='crc32' to botocore httpchecksum.py..."

    # Get the isaacsim installation path
    ISAACSIM_LOCATION=$($PYTHON_EXECUTABLE -m pip show isaacsim | grep "Location:" | cut -d' ' -f2)
    ISAACSIM_DIR="$ISAACSIM_LOCATION/isaacsim"

    # Find the botocore httpchecksum.py file
    HTTPCHECKSUM_FILE=$(find "$ISAACSIM_DIR" -name "httpchecksum.py" -path "*/botocore/*" 2>/dev/null | head -1)

    if [ -n "$HTTPCHECKSUM_FILE" ] && [ -f "$HTTPCHECKSUM_FILE" ]; then
        echo "Found botocore httpchecksum.py at: $HTTPCHECKSUM_FILE"

        # Check if DEFAULT_CHECKSUM_ALGORITHM is already defined
        if ! grep -q "DEFAULT_CHECKSUM_ALGORITHM" "$HTTPCHECKSUM_FILE"; then
            echo "Adding DEFAULT_CHECKSUM_ALGORITHM='crc32' to httpchecksum.py..."
            # Append the line to the end of the file
            echo "DEFAULT_CHECKSUM_ALGORITHM='crc32'" >> "$HTTPCHECKSUM_FILE"
            echo "Successfully added DEFAULT_CHECKSUM_ALGORITHM='crc32'"
        else
            echo "DEFAULT_CHECKSUM_ALGORITHM already exists in httpchecksum.py"
        fi
    else
        echo "Warning: Could not find botocore httpchecksum.py file in isaacsim installation"
    fi
else
    echo "isaacsim not found. Skipping DEFAULT_CHECKSUM_ALGORITHM setup."
fi

echo "PI0 Dependencies installed."
