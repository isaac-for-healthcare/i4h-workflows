# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional full patient and acoustic sensor in the maintained ultrasound cell."""

import hashlib
import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from i4h_arena.assets.config_asset import ConfigAsset
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_ultrasound import patient_layout
from i4h_arena.sensors.ultrasound import UltrasoundSensorCfg


def replace_phantom(assets, manifest):
    twin = PatientTwin.load(manifest)
    layout = patient_layout(twin)
    # Wrapper preserves the referenced patient's authored transforms when the
    # simulator sets the rigid root pose. The source bundle stays unchanged.
    source = twin.artifacts["anatomy_usd"]
    key = hashlib.sha256(
        (str(source) + str(source.stat().st_mtime_ns) + str(layout.world_from_patient_m)).encode()
    ).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"i4h-ultrasound-{key}.usda"
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Patient")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    model = UsdGeom.Xform.Define(stage, "/Patient/Model")
    model.GetPrim().GetReferences().AddReference(str(source))
    model.MakeMatrixXform().Set(Gf.Matrix4d(layout.world_from_patient_m.T.tolist()))
    stage.GetRootLayer().Export(str(path))
    for asset in assets:
        name, cfg = asset.get_object_cfg()
        if name == "organs":
            cfg.init_state.pos = (0.0, 0.0, 0.0)
            cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
            cfg.spawn = sim_utils.UsdFileCfg(
                usd_path=str(path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                semantic_tags=[("class", "organ")],
            )
        elif name == "table":
            cfg.init_state.pos = layout.table_position_m
            cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
            cfg.spawn = sim_utils.CuboidCfg(
                size=layout.table_size_m,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.24, 0.28)),
            )
        elif name == "goal_frame":
            target, rotation = layout.tcp_target()
            from scipy.spatial.transform import Rotation

            cfg.target_frames[0].offset.pos = tuple(map(float, target))
            cfg.target_frames[0].offset.rot = tuple(map(float, Rotation.from_matrix(rotation).as_quat()))
    # A simple fixed pedestal supports the tabletop down to the existing floor.
    from isaaclab.assets import AssetBaseCfg

    assets.append(
        ConfigAsset(
            "patient_table_pedestal",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/PatientTablePedestal",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(*layout.table_position_m[:2], -0.445)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.35, 0.55, 0.77), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.65, 0.7))
                ),
            ),
        )
    )
    assets.append(
        ConfigAsset(
            "robot_pedestal",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/RobotPedestal",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.42)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.30, 0.30, 0.84), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.35, 0.4))
                ),
            ),
        )
    )
    assets.append(
        ConfigAsset(
            "ultrasound",
            UltrasoundSensorCfg(
                prim_path="{ENV_REGEX_NS}/Ultrasound", patient_twin_manifest=str(twin.source), update_period=0.1
            ),
        )
    )
    return assets
