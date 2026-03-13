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

from i4h_asset_helper import BaseI4HAssets

ASSET_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d/"

BASIC_USD = ASSET_PATH + "Test/basic.usda"
PANDA_USD = ASSET_PATH + "Robots/Franka/Collected_panda_assembly/panda_assembly.usda"
PHANTOM_USD = ASSET_PATH + "Props/ABDPhantom/phantom.usda"
TABLE_WITH_COVER_USD = ASSET_PATH + "Props/VentionTable/BlackCover/table_with_cover.usd"


class Assets(BaseI4HAssets):
    """Assets manager for folder-based assets that require download."""

    organs = "Props/ABDPhantom/Organs"


# singleton used for folder assets (e.g. organs) that go through BaseI4HAssets download logic
robotic_ultrasound_assets = Assets()
