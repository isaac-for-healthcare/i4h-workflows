# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene assets for surgical robot environments."""

from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
from pxr import Usd, UsdPhysics

from i4h_arena.assets.config_asset import ConfigAsset
from i4h_arena.assets.constants import BLOCK_USD, NEEDLE_SDF_USD, ORGANS_USD, TABLE_USD

REACH_TARGET_POS = (0.02, 0.0, 0.055)
STAR_TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"


def _spawn_static_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD below a non-physics frame and disable authored rigid bodies."""
    root = sim_utils.create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
    )
    asset_root = sim_utils.spawn_from_usd(
        f"{prim_path}/Asset",
        cfg,
        **kwargs,
    )
    for prim in Usd.PrimRange(asset_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
    return root


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
        spawn=sim_utils.UsdFileCfg(
            func=_spawn_static_usd,
            usd_path=TABLE_USD,
            copy_from_source=True,
            visible=True,
        ),
    )
    star_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.0, 0.0, 0.70711, 0.70711)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=NEEDLE_SDF_USD,
            scale=(0.4, 0.4, 0.4),
            copy_from_source=True,
            visible=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            # The source OR scene selected needle.usd, whose collision does not
            # produce a grasp in the current IsaacLab/PhysX stack. Use the
            # catalog's collision-ready variant, as the table needle task does.
            usd_path=NEEDLE_SDF_USD,
            scale=(0.4, 0.4, 0.4),
            copy_from_source=True,
            visible=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                # The organ USD has no reliable support collision under the
                # needle, so hold its authored placement until the PSM grasps it.
                disable_gravity=True,
            ),
        ),
    )
    block_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BLOCK_USD,
            scale=(0.011, 0.011, 0.011),
            copy_from_source=True,
            visible=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(
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
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.1), rot=(0.0, -0.7071068, 0.0, 0.7071068)),
        spawn=sim_utils.DiskLightCfg(radius=0.2, intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def make_assets(mode: str) -> list[ConfigAsset]:
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
