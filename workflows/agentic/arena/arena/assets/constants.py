# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD path constants shared across the agentic Arena envs."""

ASSET_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d/"

# Backgrounds
MAIN_BACKGROUND_USD = ASSET_PATH + "Props/Rheo/main_new_light.usd"
TROCAR_ASSEMBLY_SCENE_USD = ASSET_PATH + "Props/LightWheel/scene03.usd"

# Robots
STAR_USD = ASSET_PATH + "Robots/STAR/star.usd"
DVRK_ECM_USD = ASSET_PATH + "Robots/dVRK/ECM/ecm.usd"
DVRK_PSM_USD = ASSET_PATH + "Robots/dVRK/PSM/psm.usd"
UNITREE_G1_29DOF_BASE_FIX_USD = (
    ASSET_PATH + "Robots/UnitreeG1/g1_29dof_with_dex3_base_fix/g1_29dof_with_dex3_base_fix.usd"
)
UNITREE_G1_29DOF_USD = ASSET_PATH + "Robots/UnitreeG1/g1_29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"

# Locomanip task props
TRAY_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTray006/SurgicalTray006.usd"
CART_USD = ASSET_PATH + "Props/LightWheel/Assets/Cart003/Cart003.usd"

# Assemble-trocar task props
TROCAR_XFORM_WO_USD = ASSET_PATH + "Props/LightWheel/Assets/Trocar002/Trocar002-xform-wo.usd"
PUNCTURE_DEVICE_XFORM_USD = (
    ASSET_PATH
    + "Props/LightWheel/Assets/DisposableLaparoscopicPunctureDevice001/DisposableLaparoscopicPunctureDevice005-xform.usd"
)
TRAY_TROCAR_ASSEMBLY_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTray001/SurgicalTray001.usd"

# Ultrasound liver-scan task props
PANDA_USD = ASSET_PATH + "Robots/Franka/Collected_panda_assembly/panda_assembly.usda"
PHANTOM_USD = ASSET_PATH + "Props/ABDPhantom/phantom.usda"
TABLE_WITH_COVER_USD = ASSET_PATH + "Props/VentionTable/BlackCover/table_with_cover.usd"

# Scissor pick-and-place task props
SOARM101_USD = ASSET_PATH + "Robots/SO-ARM/SO-ARMDualCamera.usd"
BOARD_USD = ASSET_PATH + "Props/Board/board.usd"
BLOCK_USD = ASSET_PATH + "Props/PegBlock/block.usd"
NEEDLE_USD = ASSET_PATH + "Props/SutureNeedle/needle.usd"
NEEDLE_SDF_USD = ASSET_PATH + "Props/SutureNeedle/needle_sdf.usd"
ORGANS_USD = ASSET_PATH + "Props/Organs/organs.usd"
SCISSOR_TABLE_USD = ASSET_PATH + "Props/Table/table.usd"
SCISSORS_USD = ASSET_PATH + "Props/SurgicalInstruments/SurgicalScissors.usd"
SCISSOR_TRAY_USD = ASSET_PATH + "Props/SurgicalInstruments/SurgicalTray.usd"
SUTURE_PAD_USD = ASSET_PATH + "Props/SuturePad/suture_pad.usd"
SURGICAL_TWEEZERS_USD = ASSET_PATH + "Props/LightWheel/Assets/SurgicalTweezers/AngledTweezers001.usd"
TABLE_USD = ASSET_PATH + "Props/Table/table.usd"
