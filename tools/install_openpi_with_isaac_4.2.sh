#!/bin/bash

# Clone the openpi repository
git clone git@github.com:Physical-Intelligence/openpi.git
cd openpi
git checkout 581e07d73af36d336cef1ec9d7172553b2332193
cd ..

# Update python version in pyproject.toml
pyproject_path="openpi/pyproject.toml"
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

# Install the dependencies
pip install -e git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56#egg=lerobot
pip install -e openpi/packages/openpi-client/
pip install -e openpi/.
