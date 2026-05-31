# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene assets for the ultrasound liver-scan env.

Defines ground, dome light, table, abdominal phantom, and the goal /
mesh-to-organ / organ-to-EE transforms — everything the env scene needs
that lives outside the embodiment (which carries the robot + ee_frame +
ee-to-us transform + on-board cameras).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from arena.assets.constants import PHANTOM_USD, TABLE_WITH_COVER_USD
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_arena.assets.asset import Asset
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats

_FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
_FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
_FRAME_MARKER_TINY_CFG = FRAME_MARKER_CFG.copy()
_FRAME_MARKER_TINY_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)


class _ConfigAsset(Asset):
    """Arena asset wrapper around an existing IsaacLab manager config."""

    def __init__(self, name: str, cfg, tags: list[str] | None = None) -> None:
        super().__init__(name=name, tags=tags or ["scene"])
        self._cfg = cfg

    def get_object_cfg(self) -> dict:
        return {self.name: self._cfg}


def make_ultrasound_scene_assets() -> list[_ConfigAsset]:
    """Build the ultrasound scene's non-robot assets as Arena ``Asset`` wrappers."""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.84]),
        spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.4804, 0.02017, -0.84415],
            rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, -90.0]), degrees=True),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_WITH_COVER_USD, semantic_tags=[("class", "table")]),
    )
    organs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.6, 0.0, 0.09],
            rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 180.0]), degrees=True),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PHANTOM_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            semantic_tags=[("class", "organ")],
        ),
    )
    goal_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        debug_vis=False,
        visualizer_cfg=_FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/goal_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/organs",
                name="goal_frame",
                offset=OffsetCfg(pos=(0.0, -0.25, 0.75), rot=(0, 1, 0, 0)),
            ),
        ],
    )
    mesh_to_organ_transform = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        debug_vis=False,
        visualizer_cfg=_FRAME_MARKER_TINY_CFG.replace(prim_path="/Visuals/mesh_to_organ_transform"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/organs",
                name="mesh_to_organ_transform",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0], rot=(0.7071, 0.7071, 0, 0)),
            ),
        ],
    )
    organ_to_ee_transform = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        debug_vis=False,
        visualizer_cfg=_FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/organ_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/TCP",
                name="organ_to_ee_transform",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )

    return [
        _ConfigAsset("ground", ground),
        _ConfigAsset("dome_light", dome_light),
        _ConfigAsset("table", table),
        _ConfigAsset("organs", organs),
        _ConfigAsset("goal_frame", goal_frame),
        _ConfigAsset("mesh_to_organ_transform", mesh_to_organ_transform),
        _ConfigAsset("organ_to_ee_transform", organ_to_ee_transform),
    ]
