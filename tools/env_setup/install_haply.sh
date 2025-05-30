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

VERSION=3.3.2
PLATFORM=amd64
OS_VERSION=$(lsb_release -rs)

# ---- Install Haply Inverse Service
if command -v "haply-inverse-service" &> /dev/null; then
    echo "Haply Inverse Service already installed, skipping."
    exit 0
fi

if [[ "$(uname -m)" != x86_64 ]]; then
    echo "Skipping for non-x86_64 arch."
    exit 1
fi

# Download the deb package
if [[ "$OS_VERSION" == "24.04" ]]; then
    echo "Detected Ubuntu 24.04"
    wget -O haply-inverse-service.deb https://cdn.haply.co/r/38736259/haply-inverse-service_3.3.2_amd64.deb
elif [[ "$OS_VERSION" == "22.04" ]]; then
    echo "Detected Ubuntu 22.04"
    wget -O haply-inverse-service.deb https://cdn.haply.co/r/38736259/haply-inverse-service_3.3.2_2204_amd64.deb
else
    echo "Unsupported version of Ubuntu: $OS_VERSION"
    exit 1
fi


# Install debian
sudo apt-get install -y ./haply-inverse-service.deb
rm haply-inverse-service.deb

echo "Haply Inverse Service installed!"
