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

# Common utility functions for telesurgery workflow

run_quiet() {
    local cmd="$1"
    local output

    # Capture output and exit code
    output=$(eval "$cmd" 2>&1)
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Command failed: $cmd"
        echo "$output"
        return $exit_code
    fi
    return 0
}

get_host_gpu() {
    if ! command -v nvidia-smi >/dev/null; then
        echo "Could not find any GPU drivers on host. Defaulting build to target dGPU/CPU stack." >&2
        echo -n "dgpu";
    elif [[ ! $(nvidia-smi --query-gpu=name --format=csv,noheader) ]] || \
         [[ $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) =~ "Orin (nvgpu)" ]]; then
        echo -n "igpu";
    else
        echo -n "dgpu";
    fi
}
