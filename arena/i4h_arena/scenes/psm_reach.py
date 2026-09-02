# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM reaching sampled targets on Props/Table."""

from __future__ import annotations

from i4h_arena.scenes._surgical import SurgicalReachScene


class PsmReachScene(SurgicalReachScene):
    name = "psm_reach"
    asset_mode = "reach_psm"
    reach_mode = "psm"
