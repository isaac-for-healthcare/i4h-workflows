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

HOLOSCAN_I4H_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../holoscan_i4h && pwd)"

(
  cd $HOLOSCAN_I4H_ROOT
  cmake . -B build && cmake --build build && cmake --install build

  # Check if install folder was created successfully
  if [ -d "install" ]; then
    echo "Successfully built Clarius libs!"
  else
    echo "ERROR: Install folder not found - build may have failed"
    exit 1
  fi
)
