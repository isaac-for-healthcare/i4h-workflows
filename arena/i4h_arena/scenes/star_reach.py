# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STAR arm reaching sampled targets.

Mounts SeattleLabTable rather than Props/Table, so it is not a drop-in mode
of psm_reach even though the workflows read almost identically."""

from __future__ import annotations

from i4h_arena.scenes._surgical import SurgicalReachScene


class StarReachScene(SurgicalReachScene):
    name = "star_reach"
    asset_mode = "reach_star"
    reach_mode = "star"
