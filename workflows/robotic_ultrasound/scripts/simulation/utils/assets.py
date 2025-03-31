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

import logging
import os
from dataclasses import dataclass

from i4h_asset_helper import get_i4h_local_asset_path


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

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
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, download_dir: str | None = None):
        """
        Initialize the assets

        Args:
            download_dir: The directory to download the assets to
        """
        if self._initialized:
            logger.warning("Assets already initialized. Please use the set_download_dir method to change the download directory.")
            return
            
        if download_dir is None:
            download_dir = get_i4h_local_asset_path()
        self._download_dir = download_dir
        self._update_paths()
        self._initialized = True

    def _update_paths(self):
        for attr in dir(Enums):
            if not attr.startswith("_"):
                value = getattr(Enums, attr)
                setattr(self, attr, os.path.join(self._download_dir, value))

    def set_download_dir(self, value: str):
        """
        Explicitly set the download directory and update the paths.
        
        Args:
            value: New download directory path
        """
        self._download_dir = value
        self._update_paths()

    @property
    def download_dir(self):
        """Get the download directory."""
        return self._download_dir



# singleton object for the assets
robotic_ultrasound_assets = Assets()
