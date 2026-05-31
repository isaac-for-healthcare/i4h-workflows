# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scissor pick-and-place scene assets.

Defines the InteractiveSceneCfg consumed by the scissor env (table, scissors,
tray, lights, and the SO-ARM 101 robot/cameras imported from
:mod:`arena.embodiments.so_arm`).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import isaaclab.sim as sim_utils
from arena.assets.constants import SCISSOR_TABLE_USD, SCISSOR_TRAY_USD, SCISSORS_USD
from arena.embodiments.so_arm import SOARM_ROOM_CAMERA_CFG, SOARM_WRIST_CAMERA_CFG
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset


@configclass
class ScissorPickAndPlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # NOTE: The robot is not part of this SceneCfg — SoArm101Embodiment owns
    # its own RobotSceneCfg and is merged in by `embodiment.get_scene_cfg()`.
    # `asset_names` below does not include "robot". To move / rotate the SO-ARM,
    # call `embodiment.set_initial_pose(Pose(...))` in the env class, NOT a
    # `SOARM101_CFG.replace(init_state=…)` line here (that would be silently
    # discarded).
    wrist = SOARM_WRIST_CAMERA_CFG
    room = SOARM_ROOM_CAMERA_CFG
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCISSOR_TABLE_USD,
            copy_from_source=True,
            visible=True,
            scale=(0.7, 0.7, 0.52),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.001),
        ),
    )
    scissors = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalScissors",
        # Tabletop world z ≈ 0.238 with this table scale; spawn just above so
        # the tool is visible from reset and doesn't bounce out of the table
        # collision on the first physics step.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, -0.02, 0.253), rot=(0.707, 0, 0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCISSORS_USD,
            scale=(0.006, 0.0065, 0.012),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.001),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        ),
    )
    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalTray",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, 0.25, 0.26), rot=(0.7071, 0.0, 0.0, 0.7071)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCISSOR_TRAY_USD,
            scale=(0.7, 0.7, 0.18),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.8, roughness=0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8)),
    )
    directional_light = AssetBaseCfg(
        prim_path="/World/DirectionalLight",
        spawn=sim_utils.DistantLightCfg(intensity=800.0, color=(0.95, 0.95, 0.9), angle=45.0),
    )


class ConfigAsset(Asset):
    """Arena asset wrapper around an existing IsaacLab manager config."""

    def __init__(self, name: str, cfg: Any, tags: list[str] | None = None):
        super().__init__(name=name, tags=tags or ["scene"])
        self._cfg = cfg

    def get_object_cfg(self) -> dict[str, Any]:
        return {self.name: self._cfg}


def make_scissor_pick_and_place_scene_assets() -> list[ConfigAsset]:
    """Return the scissor scene's non-robot assets as Arena ``ConfigAsset`` wrappers."""
    source = ScissorPickAndPlaceSceneCfg(env_spacing=4.0)
    asset_names = (
        "ground",
        "table",
        "scissors",
        "tray",
        "dome_light",
        "directional_light",
    )
    return [ConfigAsset(name, deepcopy(getattr(source, name))) for name in asset_names]
