#!/bin/bash

# Clone the openpi repository
git clone git@github.com:Physical-Intelligence/openpi.git
cd openpi
git checkout 581e07d73af36d336cef1ec9d7172553b2332193
cd ..
# Install toml to process pyproject.toml
pip install toml

# Extract dependencies from pyproject.toml and create requirements.txt and add LeRobot dependency
pyproject_path="openpi/pyproject.toml"
requirements_path="openpi/requirements.txt"
python install_scripts/produce_openpi_requirements.py "$pyproject_path" "$requirements_path" --add-lerobot

# Update pyproject.toml to use python >= 3.10
sed -i.bak -e 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' "$pyproject_path"

# Apply temporary workaround for openpi/src/openpi/shared/download.py
file_path="openpi/src/openpi/shared/download.py"

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

# Remove the backup files
rm "$pyproject_path.bak"
rm "$file_path.bak"
