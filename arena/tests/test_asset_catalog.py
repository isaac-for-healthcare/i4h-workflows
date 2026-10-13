# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow assets load from the published repaired catalog."""

import importlib.util
from pathlib import Path

CONSTANTS = Path(__file__).parents[1] / "i4h_arena" / "assets" / "constants.py"


def test_workflow_uses_published_repaired_catalog():
    spec = importlib.util.spec_from_file_location("asset_constants", CONSTANTS)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    assert constants.ASSET_PATH.endswith("/IsaacHealthcare/v0.9.0/")
    assert constants.ORGANS_USD == constants.ASSET_PATH + "Props/Organs/organs.usd"
    assert constants.DVRK_PSM_USD.startswith(constants.ASSET_PATH)
