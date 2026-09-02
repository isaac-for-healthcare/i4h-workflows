# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two dVRK PSM arms reaching sampled targets on Props/Table."""

from __future__ import annotations

from i4h_arena.adapters.actuation import RobotSlice
from i4h_arena.scenes._surgical import SurgicalReachScene


class DualPsmReachScene(SurgicalReachScene):
    name = "dual_psm_reach"
    asset_mode = "reach_dual_psm"
    reach_mode = "dual_psm"

    def robot_assets(self) -> dict[str, str]:
        return {"psm1": "robot_1", "psm2": "robot_2"}

    def tcp_sensors(self) -> dict[str, str]:
        return {"psm1": "ee_frame", "psm2": "ee_2_frame"}

    def robot_slices(self, env):
        width = int(env.action_space.shape[-1])
        half = width // 2
        return (
            RobotSlice("psm1", 0, half),
            RobotSlice("psm2", half, width),
        )
