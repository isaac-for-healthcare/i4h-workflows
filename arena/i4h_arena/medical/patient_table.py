# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Support a scan-registered SOMA patient without moving the CT coordinate frame."""

from __future__ import annotations

import numpy as np


def fit_table_to_soma(scene, twin) -> bool:
    """Adjust the existing table to the full-body exterior, if present.

    The CT, catheter, C-arm and patient transforms stay untouched. Legacy CT-only
    bundles keep their existing table. Geometry is measured in world meters.
    """
    from pxr import Usd, UsdGeom

    path = twin.artifacts.get("anatomy_usd")
    if path is None:
        return False
    stage = Usd.Stage.Open(str(path))
    skin = stage.GetPrimAtPath("/HumanBody/Exterior/SOMA")
    if not skin:
        return False
    points = np.asarray(UsdGeom.Mesh(skin).GetPointsAttr().Get(), dtype=float)
    local = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(skin)).T
    points = points @ local[:3, :3].T + local[:3, 3]
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    world = twin.world_from_patient_m
    points = points @ world[:3, :3].T + world[:3, 3]
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("SOMA exterior has no finite bounds")
    lower, upper = points.min(0), points.max(0)
    top, frame, base, foot = (
        getattr(scene, name)
        for name in ("patient_table", "patient_table_frame", "patient_table_base", "patient_table_foot")
    )
    center = (lower[:2] + upper[:2]) / 2
    shift = center - np.asarray(top.init_state.pos[:2])
    top_z = lower[2] - 0.002 - top.spawn.size[2] / 2
    dz = top_z - top.init_state.pos[2]
    # Keep a five-centimeter margin around the full patient silhouette.
    top.spawn.size = (*np.maximum(top.spawn.size[:2], upper[:2] - lower[:2] + 0.10), top.spawn.size[2])
    top.init_state.pos = (*center, top_z)
    frame.spawn.size = (top.spawn.size[0] - 0.29, top.spawn.size[1] - 0.12, frame.spawn.size[2])
    frame.init_state.pos = (*center, frame.init_state.pos[2] + dz)
    if base.spawn.size[2] + dz <= 0:
        raise ValueError("Patient is too low to support on the table pedestal")
    base.spawn.size = (*base.spawn.size[:2], base.spawn.size[2] + dz)
    base.init_state.pos = (*(np.asarray(base.init_state.pos[:2]) + shift), base.init_state.pos[2] + dz / 2)
    foot.init_state.pos = (*(np.asarray(foot.init_state.pos[:2]) + shift), foot.init_state.pos[2])
    return True
