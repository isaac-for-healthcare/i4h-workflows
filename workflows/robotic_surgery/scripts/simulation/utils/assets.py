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


class Assets(BaseI4HAssets):
    """Assets manager for the robotic surgery workflow."""

    dVRK_ECM = "Robots/dVRK/ECM/ecm.usd"
    dVRK_PSM = "Robots/dVRK/PSM/psm.usd"
    STAR = "Robots/STAR/star.usd"
    Board = "Props/Board/board.usd"
    Block = "Props/PegBlock/block.usd"
    Needle = "Props/SutureNeedle/needle.usd"
    Needle_SDF = "Props/SutureNeedle/needle_sdf.usd"
    SuturePad = "Props/SuturePad/suture_pad.usd"
    Table = "Props/Table/table.usd"
    Organs = "Props/Organs/organs.usd"


# singleton object for the assets
robotic_surgery_assets = Assets()
