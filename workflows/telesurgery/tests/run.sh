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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/" >/dev/null 2>&1 && pwd)"

# Convert relative paths to absolute paths
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../scripts" && pwd)"

PYTHONPATH="$SCRIPTS_DIR:$SCRIPT_DIR:$PYTHONPATH"
export PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"
echo "Running tests..."
# Run the test
python -m unittest discover -s $SCRIPT_DIR $@
