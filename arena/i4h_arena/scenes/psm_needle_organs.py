# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM lifting a suture needle from an organ bed."""

from __future__ import annotations

from i4h_arena.scenes._surgical import SurgicalLiftScene


class PsmNeedleOrgansScene(SurgicalLiftScene):
    name = "psm_needle_organs"
    asset_mode = "lift_needle_organs"
    task_description = "Lift the suture needle in an organ scene with the dVRK PSM."
    organs = True
