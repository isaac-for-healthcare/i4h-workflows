# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assets used by the G1 loco-manipulation envs (tray pick-and-place, push cart).

Importing this module registers the ``pre_op`` background and the
``surgical_tray`` / ``cart`` objects against
:class:`isaaclab_arena.assets.AssetRegistry`. The locomanip envs look them
up by string name (see :meth:`get_env`).
"""

from __future__ import annotations

from arena.assets._base import _ArticulatedUsdObject
from arena.assets.constants import CART_USD, MAIN_BACKGROUND_USD, TRAY_USD
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class PreOpBackground(LibraryBackground):
    """Pre-operative room used as the locomanip task backdrop."""

    name = "pre_op"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = MAIN_BACKGROUND_USD
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.5

    def __init__(self):
        super().__init__()


@register_asset
class SurgicalTray(_ArticulatedUsdObject):
    """Tray with lid — the locomanip tray-pick-and-place target object."""

    name = "surgical_tray"
    tags = ["object"]
    prim_name = "surgical_tray"
    usd_path = TRAY_USD
    semantic_class = "box"

    def __init__(self, *args, mass: float | None = 0.1, **kwargs):
        super().__init__(*args, mass=mass, **kwargs)


@register_asset
class Cart(_ArticulatedUsdObject):
    """Mobile cart — destination for tray-pick-and-place, target for push-cart."""

    name = "cart"
    tags = ["object"]
    prim_name = "cart"
    usd_path = CART_USD
    semantic_class = "cart"
