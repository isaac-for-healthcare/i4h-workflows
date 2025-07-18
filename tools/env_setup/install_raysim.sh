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

echo "Installing Raysim..."

# Get the parent directory of the current script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"


if [ -d "$PROJECT_ROOT/workflows/robotic_ultrasound/scripts/raysim" ]; then
    echo "Raysim already installed. Skipping installation."
    exit 0
fi

echo "Downloading Raysim..."

wget https://github.com/isaac-for-healthcare/i4h-asset-catalog/releases/download/v0.2.0/raysim-py310-linux-v0.2.0.zip \
    -O $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/raysim.zip

echo "Unzipping Raysim..."

unzip $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/raysim.zip -d $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/raysim

rm $PROJECT_ROOT/workflows/robotic_ultrasound/scripts/raysim.zip

echo "Raysim installed successfully."
