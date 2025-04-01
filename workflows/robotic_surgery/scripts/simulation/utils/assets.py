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
    """Enums for the assets in the robotic surgery workflow."""

    dVRK_ECM = "Robots/dVRK/ECM/ecm.usd"
    dVRK_PSM = "Robots/dVRK/PSM/psm.usd"
    STAR = "Robots/STAR/star.usd"
    Board = "Props/Board/board.usd"
    Block = "Props/PegBlock/block.usd"
    Needle = "Props/SutureNeedle/needle_sdf.usd"
    SuturePad = "Props/SuturePad/suture_pad.usd"
    Table = "Props/Table/table.usd"


class Assets(Enums):
    """
    Assets manager for the robotic surgery workflow.
    
    This singleton class manages asset paths for the simulation, automatically resolving them
    relative to a configurable download directory.

    Attributes:
        dVRK_ECM (str): Path to the dVRK ECM asset
        dVRK_PSM (str): Path to the dVRK PSM asset
        STAR (str): Path to the STAR asset
        Board (str): Path to the Board asset
        Block (str): Path to the Block asset
        Needle (str): Path to the Needle asset
        SuturePad (str): Path to the SuturePad asset
        Table (str): Path to the Table asset
        download_dir (str): Base directory containing all assets

    Important:
        The download directory must be set before any simulation extensions that depend on
        these assets are imported. This is because asset paths are resolved during
        initialization of those extensions.

    Example:
        >>> from simulation.utils.assets import robotic_surgery_assets as assets
        >>> assets.download_dir = "/path/to/assets"
        >>> # Only import dependent modules after setting download_dir
        >>> from robotic.surgery.assets.psm import PSM_HIGH_PD_CFG
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
            logger.warning(
                (
                    f"Assets {self.__class__.__name__} was previously initialized. "
                    "Re-initializing will not change the path of the assets that have already been loaded."
                )
            )
            return

        if download_dir is None:
            download_dir = get_i4h_local_asset_path()
        self._download_dir = download_dir
        self._update_paths()
        self._initialized = True

    def _update_paths(self):
        """Update all the paths."""
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
        if not os.path.isdir(value):
            logger.warning(f"Download directory {value} does not exist.")
        self._update_paths()
    

# singleton object for the assets
robotic_surgery_assets = Assets()
