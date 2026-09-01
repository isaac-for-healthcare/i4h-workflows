# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assets for the ultrasound-probe target-reaching scene."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from i4h_arena.assets.config_asset import ConfigAsset
from i4h_arena.assets.panda_phantom import make_assets as make_ultrasound_assets


def make_assets() -> list[ConfigAsset]:
    """Reuse the maintained ultrasound cell and add a randomized marker."""
    scene_assets = make_ultrasound_assets()
    for asset in scene_assets:
        name, cfg = asset.get_object_cfg()
        if name in {"table", "organs"}:
            # This is a reach objective, not a contact-dynamics objective. Keep
            # the authored work surface and phantom fixed across every reset.
            cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            )
            cfg.spawn.mass_props = None
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target",
        init_state=RigidObjectCfg.InitialStateCfg(
            # Measured from the phantom mesh, whose upper torso surface is
            # centered near (0.60, -0.075, 0.191) in world coordinates.
            # The 8 mm marker sits 1 mm above that surface.
            pos=(0.60, -0.075, 0.196),
            # Probe-down orientation measured from the maintained ultrasound
            # setup pose (wxyz), so the marker normal matches the actual TCP.
            rot=(0.00625011, 0.99994457, 0.00289107, 0.00796396),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.2),
                emissive_color=(0.02, 0.25, 0.04),
            ),
        ),
    )
    return [*scene_assets, ConfigAsset("target", target)]
