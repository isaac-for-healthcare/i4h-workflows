# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assets used by the G1 assemble-trocar env.

Importing this module registers the ``trocar_assembly_scene`` background and
the ``trocar_1`` / ``trocar_2`` / ``tray`` objects against
:class:`isaaclab_arena.assets.AssetRegistry`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from arena.assets._base import _ArticulatedUsdObject, _RigidUsdObject
from arena.assets.constants import (
    PUNCTURE_DEVICE_XFORM_USD,
    TRAY_TROCAR_ASSEMBLY_USD,
    TROCAR_ASSEMBLY_SCENE_USD,
    TROCAR_XFORM_WO_USD,
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class TrocarAssemblyBackground(Background):
    """LightWheel trocar assembly room — backdrop for the trocar task."""

    name = "trocar_assembly_scene"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = TROCAR_ASSEMBLY_SCENE_USD
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = 0.5

    def __init__(self):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=self.usd_path,
            initial_pose=self.initial_pose,
            object_min_z=self.object_min_z,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Scene",
            spawn=UsdFileCfg(usd_path=self.usd_path),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


@register_asset
class TrocarAssemblyTrocar(_RigidUsdObject):
    """First trocar piece to be assembled (the cannula)."""

    name = "trocar_1"
    tags = ["object"]
    prim_name = "trocar_1"
    usd_path = TROCAR_XFORM_WO_USD

    def _generate_rigid_cfg(self):
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.001,
                    rest_offset=-0.001,
                ),
            ),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


@register_asset
class TrocarAssemblyPunctureDevice(_RigidUsdObject):
    """Second trocar piece to be assembled (the puncture device)."""

    name = "trocar_2"
    tags = ["object"]
    prim_name = "trocar_2"
    usd_path = PUNCTURE_DEVICE_XFORM_USD

    def _generate_rigid_cfg(self):
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                ),
            ),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


@register_asset
class TrocarAssemblyTray(_ArticulatedUsdObject):
    """Holding tray that contains both trocar pieces at episode start."""

    name = "tray"
    tags = ["object"]
    prim_name = "surgical_tray"
    usd_path = TRAY_TROCAR_ASSEMBLY_USD

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path),
            actuators={},
        )
        return self._add_initial_pose_to_cfg(object_cfg)
