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

import os
from dataclasses import dataclass

from i4h_asset_helper import get_i4h_local_asset_path


@dataclass
class Enums:
    """Enums for the assets in the robotic ultrasound workflow."""

    basic = "Test/basic.usda"
    panda = "Robots/Franka/Collected_panda_assembly/panda_assembly.usda"
    phantom = "Props/ABDPhantom/phantom.usda"
    table_with_cover = "Props/VentionTableWithBlackCover/table_with_cover.usd"
    policy_ckpt = "Policies/LiverScan"
    organs = "Props/ABDPhantom/Organs"


class Assets(Enums):
    """
    Assets for the robotic ultrasound workflow customized for the user download directory.

    This class is a singleton that will download the assets to the user's local asset directory.
    It has the following attributes:
    - basic: the path to the basic.usda asset
    - panda: the path to the panda_assebly.usda asset
    - phantom: the path to the phantom.usda asset
    - table_with_cover: the path to the table_with cover_.usd asset

    """

    def __init__(self, download_dir: str | None = None):
        """
        Initialize the assets

        Args:
            download_dir: The directory to download the assets to
        """
        if download_dir is None:
            download_dir = get_i4h_local_asset_path()
        self._download_dir = download_dir
        self._update_paths()

    def _update_paths(self):
        for attr in dir(Enums):
            if not attr.startswith("_"):
                value = getattr(Enums, attr)
                setattr(self, attr, os.path.join(self._download_dir, value))

    @property
    def download_dir(self):
        """Get the download directory."""
        return self._download_dir

    @download_dir.setter
    def download_dir(self, value):
        """Set the download directory and update the paths."""
        self._download_dir = value
        self._update_paths()


# singleton object for the assets
robotic_ultrasound_assets = Assets()
