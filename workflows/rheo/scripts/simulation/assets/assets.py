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

ASSET_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d/"

# Scenes
MAIN_BACKGROUND_USD = ASSET_PATH + "Props/Rheo/main_new_light.usd"
TROCAR_ASSEMBLY_SCENE_USD = ASSET_PATH + "Props/LightWheel/scene03.usd"

# Robots
UNITREE_G1_29DOF_BASE_FIX_USD = (
    ASSET_PATH + "Robots/UnitreeG1/g1_29dof_with_dex3_base_fix/g1_29dof_with_dex3_base_fix.usd"
)
UNITREE_G1_29DOF_USD = ASSET_PATH + "Robots/UnitreeG1/g1_29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"

# Props
SCISSORS_USD = ASSET_PATH + "Props/SurgicalInstruments/SurgicalScissors.usd"

# Props Powered by LightWheel
# Please be noted that these assets are under Attribution-NonCommercial 4.0 International License.
# Check the license details by replacing the usd file base name with "LICENSE.txt"

PLATE_USD = ASSET_PATH + "Props/LightWheel/Assets/Plate001/plate001.usd"
TROCAR_USD = ASSET_PATH + "Props/LightWheel/Assets/Trocar002/Trocar002.usd"
TROCAR_XFORM_WO_USD = ASSET_PATH + "Props/LightWheel/Assets/Trocar002/Trocar002-xform-wo.usd"
NEEDLE_USD = ASSET_PATH + "Props/LightWheel/Assets/PneumoperitoneumNeedle001/PneumoperitoneumNeedle001.usd"
PUNCTURE_DEVICE_USD = ASSET_PATH + "Props/LightWheel/Assets/PunctureDevice002/PunctureDevice002.usd"
PUNCTURE_DEVICE_XFORM_USD = (
    ASSET_PATH
    + "Props/LightWheel/Assets/DisposableLaparoscopicPunctureDevice001/DisposableLaparoscopicPunctureDevice005-xform.usd"  # noqa: E501
)
TUBE_USD = ASSET_PATH + "Props/LightWheel/Assets/DrainageTube002/DrainageTube003.usd"

TWEEZERS_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTweezers/AngledTweezers001.usd"
TRAY_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTray006/SurgicalTray006.usd"
TRAY_NO_LID_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTrayNoLid006/SurgicalTrayNoLid006.usd"
TRAY_TROCAR_ASSEMBLY_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTray001/SurgicalTray001.usd"
CART_USD = ASSET_PATH + "Props/LightWheel/Assets/Cart003/Cart003.usd"
