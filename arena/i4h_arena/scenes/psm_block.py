# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM lifting a peg-transfer block off Props/Table."""

from __future__ import annotations

from i4h_arena.scenes._surgical import SurgicalLiftScene


class PsmBlockScene(SurgicalLiftScene):
    name = "psm_block"
    asset_mode = "lift_block"
    task_description = "Lift the peg-transfer block with the dVRK PSM."
    organs = False
