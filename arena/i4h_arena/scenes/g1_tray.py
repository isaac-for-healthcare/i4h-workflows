# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 in the Rheo room with a surgical tray and a destination cart."""

from __future__ import annotations

from typing import Any

from i4h_arena.scenes._locomanip import LocomanipScene


class G1TrayScene(LocomanipScene):
    name = "g1_tray"
    pick_up_pose = ((-1.15, -1.6, -0.08), (0.0, 0.0, 0.707, 0.707))
    robot_pose = ((-0.5, -1.62, 0.0), (0.0, 0.0, 1.0, 0.0))

    def envcfg(self, pick_up_object: Any, destination_cart: Any, background: Any) -> Any:
        from i4h_arena.envcfg.g1_tray import G1TrayEnvCfg  # noqa: PLC0415

        return G1TrayEnvCfg(
            pick_up_object,
            destination_cart,
            background,
            episode_length_s=self.episode_length_s(),
        )
