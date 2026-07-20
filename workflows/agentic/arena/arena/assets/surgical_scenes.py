# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene assets for surgical robot environments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import isaaclab.sim as sim_utils
from arena.assets.constants import BLOCK_USD, NEEDLE_SDF_USD, NEEDLE_USD, ORGANS_USD, TABLE_USD
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_arena.assets.asset import Asset

REACH_TARGET_POS = (0.02, 0.0, 0.055)
STAR_TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"


@configclass
class SurgicalSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.95)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD, copy_from_source=True, visible=True),
    )
    star_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
        spawn=sim_utils.UsdFileCfg(usd_path=STAR_TABLE_USD, copy_from_source=True, visible=True),
    )
    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReachTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=REACH_TARGET_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
                emissive_color=(0.8, 0.0, 0.0),
            ),
        ),
    )
    needle_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=NEEDLE_SDF_USD,
            scale=(0.4, 0.4, 0.4),
            copy_from_source=True,
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    organs_needle_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=NEEDLE_USD,
            scale=(0.4, 0.4, 0.4),
            copy_from_source=True,
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    block_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BLOCK_USD,
            scale=(0.011, 0.011, 0.011),
            copy_from_source=True,
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    organs = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Organs",
        # Matches robotic_surgery's NeedleLiftOREnvCfg: the USD is authored as
        # a full OR scene and must be scaled down into the PSM workspace.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.25, -0.14, -0.85),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=ORGANS_USD,
            scale=(0.01, 0.01, 0.01),
            copy_from_source=True,
            visible=True,
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    or_light = AssetBaseCfg(
        prim_path="/World/light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.1), rot=(0.7071068, 0.0, -0.7071068, 0.0)),
        spawn=sim_utils.DiskLightCfg(radius=0.2, intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


class ConfigAsset(Asset):
    """Arena asset wrapper around an existing IsaacLab manager config."""

    def __init__(self, name: str, cfg: Any, tags: list[str] | None = None):
        super().__init__(name=name, tags=tags or ["scene"])
        self._cfg = cfg

    def get_object_cfg(self) -> dict[str, Any]:
        return {self.name: self._cfg}


def make_surgical_scene_assets(mode: str) -> list[ConfigAsset]:
    """Return scene assets for one surgical scenario."""
    source = SurgicalSceneCfg(env_spacing=2.5)
    assets: list[ConfigAsset] = []

    def add(scene_key: str, cfg_name: str | None = None) -> None:
        assets.append(ConfigAsset(scene_key, deepcopy(getattr(source, cfg_name or scene_key))))

    add("ground")
    if mode == "reach_star":
        add("table", "star_table")
    elif mode != "lift_needle_organs":
        add("table")

    if mode == "lift_needle_organs":
        add("light", "or_light")
    else:
        add("light", "dome_light")

    if mode == "lift_needle":
        add("object", "needle_object")
    elif mode == "lift_needle_organs":
        add("organs")
        add("object", "organs_needle_object")
    elif mode == "lift_block":
        add("object", "block_object")

    return assets
