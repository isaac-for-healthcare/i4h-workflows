# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM lifting a suture needle (SDF collision mesh) off Props/Table."""

from __future__ import annotations

from i4h_arena.scenes._surgical import SurgicalLiftScene


class PsmNeedleScene(SurgicalLiftScene):
    name = "psm_needle"
    asset_mode = "lift_needle"
    task_description = "Lift the suture needle with the dVRK PSM."
    organs = False
