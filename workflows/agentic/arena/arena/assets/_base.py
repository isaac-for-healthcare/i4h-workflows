# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared object base classes used by the per-task asset modules.

These three abstract bases (rigid, articulated, plain USD) cover every
@register_asset entry we ship today. New tasks should reuse them rather
than re-deriving from :class:`isaaclab_arena.assets.Object` directly.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose


class _BaseUsdObject(Object):
    name: str
    usd_path: str
    prim_name: str
    scale = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            prim_path=prim_path or f"{{ENV_REGEX_NS}}/{self.prim_name}",
            object_type=ObjectType.BASE,
            usd_path=self.usd_path,
            scale=self.scale,
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


class _ArticulatedUsdObject(Object):
    name: str
    usd_path: str
    prim_name: str
    scale = (1.0, 1.0, 1.0)
    semantic_class: str | None = None

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        kinematic_enabled: bool = False,
        disable_gravity: bool = False,
        mass: float | None = None,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
        **kwargs,
    ):
        self.kinematic_enabled = kinematic_enabled
        self.disable_gravity = disable_gravity
        self.mass = mass
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        super().__init__(
            name=self.name,
            prim_path=prim_path or f"{{ENV_REGEX_NS}}/{self.prim_name}",
            object_type=ObjectType.ARTICULATION,
            usd_path=self.usd_path,
            scale=self.scale,
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        semantic_tags = [("class", self.semantic_class)] if self.semantic_class is not None else None
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=self.kinematic_enabled,
                    disable_gravity=self.disable_gravity,
                    linear_damping=self.linear_damping,
                    angular_damping=self.angular_damping,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass) if self.mass is not None else None,
                activate_contact_sensors=False,
                semantic_tags=semantic_tags,
            ),
            actuators={},
        )
        return self._add_initial_pose_to_cfg(object_cfg)


class _RigidUsdObject(Object):
    name: str
    usd_path: str
    prim_name: str
    scale = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            prim_path=prim_path or f"{{ENV_REGEX_NS}}/{self.prim_name}",
            object_type=ObjectType.RIGID,
            usd_path=self.usd_path,
            scale=self.scale,
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_rigid_cfg(self):
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.001,
                    rest_offset=-0.001,
                ),
            ),
        )
        return self._add_initial_pose_to_cfg(object_cfg)
