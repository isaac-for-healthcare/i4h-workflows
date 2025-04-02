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

from pathlib import Path


def get_workflow_root() -> Path:
    """Get the root directory of the workflow."""
    return Path(__file__).parent.parent


def get_scripts_dir() -> Path:
    """Get the scripts directory of the workflow."""
    return get_workflow_root() / "scripts"


def get_default_dds_qos_profile() -> Path:
    """Get the default DDS QoS profile."""
    return str(get_scripts_dir() / "dds" / "qos_profiles.xml")


if __name__ == "__main__":
    print(get_default_dds_qos_profile())
