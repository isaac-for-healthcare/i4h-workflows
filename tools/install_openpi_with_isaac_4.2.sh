#!/bin/bash

# Clone the openpi repository
git clone git@github.com:Physical-Intelligence/openpi.git
cd openpi
git checkout 581e07d73af36d336cef1ec9d7172553b2332193
cd ..

# Update python version in pyproject.toml
pyproject_path="openpi/pyproject.toml"
sed -i.bak \
    -e 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' \
    -e 's/"s3fs>=2024.9.0"/"s3fs==2024.9.0"/' \
    "$pyproject_path"

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

# Modify the type hints in training/utils.py to use Any instead of optax types
utils_path="openpi/src/openpi/training/utils.py"
sed -i.bak \
    -e 's/opt_state: optax\.OptState/opt_state: Any/' \
    "$utils_path"

# Remove the backup files
rm "$pyproject_path.bak"
rm "$file_path.bak"
rm "$utils_path.bak"

# Add training script to openpi module
if [ ! -f openpi/src/openpi/train.py ]; then
    cp openpi/scripts/train.py openpi/src/openpi/train.py
fi

# Add norm stats generator script to openpi module
if [ ! -f openpi/src/openpi/compute_norm_stats.py ]; then
    cp openpi/scripts/compute_norm_stats.py openpi/src/openpi/compute_norm_stats.py
fi

# Install the dependencies
pip install git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56
pip install -e openpi/packages/openpi-client/
pip install -e openpi/.

# Revert the "import changes of "$file_path after installation to prevent errors
sed -i \
    -e 's/^# import boto3\.s3\.transfer as s3_transfer/import boto3.s3.transfer as s3_transfer/' \
    -e 's/^# import s3transfer\.futures as s3_transfer_futures/import s3transfer.futures as s3_transfer_futures/' \
    -e 's/^# from types_boto3_s3\.service_resource import ObjectSummary/from types_boto3_s3.service_resource import ObjectSummary/' \
    "$file_path"
