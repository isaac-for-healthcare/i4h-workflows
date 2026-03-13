# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions for the so_arm_starter workflow."""

import os


def resolve_recording_path(recording_path: str) -> str:
    """Resolve HDF5 teleoperation recording path according to standard I4H / HoloHub structure."""
    if os.path.isabs(recording_path):
        return recording_path
    parent_output_dir = os.getenv("HOLOHUB_DATA_PATH", "") or os.path.abspath("./data")
    default_output_dir = os.path.join(parent_output_dir, "so_arm_starter", "recordings")
    return os.path.normpath(os.path.join(default_output_dir, recording_path))
