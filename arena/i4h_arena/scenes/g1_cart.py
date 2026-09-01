# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 in the Rheo room with a cart to push.

Same room and robot as g1_tray; the cart is the prop here rather than the
destination, so it is placed at grasping height instead of on the floor.
"""

from __future__ import annotations

from typing import Any

from i4h_arena.scenes._locomanip import LocomanipScene


class G1CartScene(LocomanipScene):
    name = "g1_cart"
    pick_up_pose = ((0.35, -1.65, 0.10), (0.0, 0.0, 0.707, 0.707))
    robot_pose = ((-0.4, -1.62, 0.0), (0.0, 0.0, 0.0, 1.0))

    def envcfg(self, pick_up_object: Any, destination_cart: Any, background: Any) -> Any:
        from i4h_arena.envcfg.g1_cart import G1CartEnvCfg  # noqa: PLC0415

        return G1CartEnvCfg(
            pick_up_object,
            destination_cart,
            background,
            episode_length_s=self.episode_length_s(),
        )
