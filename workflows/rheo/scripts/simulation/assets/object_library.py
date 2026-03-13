# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow-specific object library entries."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from simulation.assets.assets import (
    CART_USD,
    NEEDLE_USD,
    PLATE_USD,
    PUNCTURE_DEVICE_USD,
    PUNCTURE_DEVICE_XFORM_USD,
    SCISSORS_USD,
    TRAY_NO_LID_USD,
    TRAY_USD,
    TROCAR_USD,
    TUBE_USD,
    TWEEZERS_USD,
)


@register_asset
class SurgicalPlate(LibraryObject):
    """A surgical plate from the Pre-Operative scene."""

    name = "surgical_plate"
    tags = ["object"]
    usd_path = PLATE_USD
    default_prim_path = "{ENV_REGEX_NS}/surgical_plate"
    scale = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Trocar(Object):
    """A trocar from the Pre-Operative scene (visual/static object)."""

    name = "trocar"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="trocar",
            prim_path=prim_path or "{ENV_REGEX_NS}/trocar",
            object_type=ObjectType.BASE,
            usd_path=TROCAR_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class Laparoscope(Object):
    """A laparoscope from the Pre-Operative scene (pure visual object)."""

    name = "laparoscope"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="laparoscope",
            prim_path=prim_path or "{ENV_REGEX_NS}/laparoscope",
            object_type=ObjectType.BASE,
            usd_path=PUNCTURE_DEVICE_XFORM_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class Needle(Object):
    """A needle from the Pre-Operative scene (pure visual object)."""

    name = "needle"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="needle",
            prim_path=prim_path or "{ENV_REGEX_NS}/needle",
            object_type=ObjectType.BASE,
            usd_path=NEEDLE_USD,
            scale=(1.0, 1.0, 0.8),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class oPunctureDevice(Object):
    """A puncture device from the Pre-Operative scene."""

    name = "puncture_device"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="puncture_device",
            prim_path=prim_path or "{ENV_REGEX_NS}/puncture_device",
            object_type=ObjectType.BASE,
            usd_path=PUNCTURE_DEVICE_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class Tube(Object):
    """A tube from the Pre-Operative scene."""

    name = "tube"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="tube",
            prim_path=prim_path or "{ENV_REGEX_NS}/tube",
            object_type=ObjectType.BASE,  # Pure visual, no physics to avoid deformable conflicts
            usd_path=TUBE_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class Scissors(Object):
    """A scissors from the Pre-Operative scene."""

    name = "scissors"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="scissors",
            prim_path=prim_path or "{ENV_REGEX_NS}/scissors",
            object_type=ObjectType.BASE,
            usd_path=SCISSORS_USD,
            scale=(0.006, 0.006, 0.006),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class Tweezers(Object):
    """A tweezers from the Pre-Operative scene."""

    name = "tweezers"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name="tweezers",
            prim_path=prim_path or "{ENV_REGEX_NS}/tweezers",
            object_type=ObjectType.BASE,
            usd_path=TWEEZERS_USD,
            scale=(0.8, 0.8, 0.8),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_base_cfg(self):
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class SurgicalTray(Object):
    """Sterile container with articulated components."""

    name = "surgical_tray"
    tags = ["object"]

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        kinematic_enabled: bool = False,
        disable_gravity: bool = False,
        mass: float = 0.1,
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
            name="surgical_tray",
            prim_path=prim_path or "{ENV_REGEX_NS}/surgical_tray",
            object_type=ObjectType.ARTICULATION,
            usd_path=TRAY_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=self.kinematic_enabled,
            disable_gravity=self.disable_gravity,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        )

        mass_props = sim_utils.MassPropertiesCfg(mass=self.mass) if self.mass is not None else None

        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                articulation_props=articulation_props,
                rigid_props=rigid_props,
                mass_props=mass_props,
                activate_contact_sensors=False,
                semantic_tags=[("class", "box")],  # Add semantic tag for segmentation
            ),
            actuators={},
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class SurgicalTrayNoLid(Object):
    """A Surgical Tray from the Pre-Operative scene without a lid - multi-part articulation."""

    name = "surgical_tray_no_lid"
    tags = ["object"]

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
            name="surgical_tray_no_lid",
            prim_path=prim_path or "{ENV_REGEX_NS}/surgical_tray_no_lid",
            object_type=ObjectType.ARTICULATION,
            usd_path=TRAY_NO_LID_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=self.kinematic_enabled,
            disable_gravity=self.disable_gravity,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        )

        mass_props = sim_utils.MassPropertiesCfg(mass=self.mass) if self.mass is not None else None

        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                articulation_props=articulation_props,
                rigid_props=rigid_props,
                mass_props=mass_props,
                activate_contact_sensors=False,
            ),
            actuators={},
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg


@register_asset
class Cart(Object):
    """Surgical cart with multiple rigid bodies."""

    name = "cart"
    tags = ["object"]

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
            name="cart",
            prim_path=prim_path or "{ENV_REGEX_NS}/cart",
            object_type=ObjectType.ARTICULATION,
            usd_path=CART_USD,
            scale=(1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=self.kinematic_enabled,
            disable_gravity=self.disable_gravity,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        )

        mass_props = sim_utils.MassPropertiesCfg(mass=self.mass) if self.mass is not None else None

        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                articulation_props=articulation_props,
                rigid_props=rigid_props,
                mass_props=mass_props,
                activate_contact_sensors=False,
                semantic_tags=[("class", "cart")],  # Add semantic tag for segmentation
            ),
            actuators={},
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg
