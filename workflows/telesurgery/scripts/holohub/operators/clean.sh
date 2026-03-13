#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e; cd "$(dirname "$0")"

echo "Cleaning build artifacts..."

# Remove build logs
rm -f build.log

# Remove CMake build directories
find . -name ".build_py*" -type d -exec rm -rf {} + 2>/dev/null || true
find . -maxdepth 1 -name "build*" ! -name "build.sh" -exec rm -rf {} +

# Remove streamlift build artifacts
rm -rf streamlift/lib
rm -rf streamlift/streamlift_downsampler
rm -rf streamlift/streamlift_upsampler
rm -rf streamlift/__pycache__

# Remove holohub operator libs
rm -rf camera/aja_source/lib
rm -f camera/aja_source/_aja_source*.so
rm -rf camera/aja_source/__pycache__

# Remove nvidia_video_codec generated directories
rm -rf nvidia_video_codec/lib
rm -rf nvidia_video_codec/nv_video_encoder
rm -rf nvidia_video_codec/nv_video_decoder
rm -rf nvidia_video_codec/__pycache__

echo "✓ Cleaned"
