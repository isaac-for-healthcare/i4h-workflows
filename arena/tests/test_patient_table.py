# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.patient_table import fit_table_to_soma


def test_soma_table_preserves_ct_frame_and_supports_patient(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "patient.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Mesh.Define(stage, "/HumanBody/Exterior/SOMA").CreatePointsAttr([(-0.30, -0.10, -1.1), (0.30, 0.20, 0.60)])
    stage.GetRootLayer().Save()
    world = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.85], [0, 0, 0, 1]], dtype=float)
    twin = SimpleNamespace(artifacts={"anatomy_usd": tmp_path / "patient.usda"}, world_from_patient_m=world)

    def asset(size, pos):
        return SimpleNamespace(spawn=SimpleNamespace(size=size), init_state=SimpleNamespace(pos=pos))

    scene = SimpleNamespace(
        patient_table=asset((1.55, 0.56, 0.06), (0, 0, 0.74)),
        patient_table_frame=asset((1.26, 0.44, 0.1), (0, 0, 0.67)),
        patient_table_base=asset((0.34, 0.3, 0.64), (0.42, 0, 0.39)),
        patient_table_foot=asset((0.82, 0.42, 0.07), (0.42, 0, 0.055)),
    )
    original = world.copy()
    assert fit_table_to_soma(scene, twin)
    np.testing.assert_array_equal(world, original)
    top = scene.patient_table
    assert top.init_state.pos[2] + top.spawn.size[2] / 2 == pytest.approx(0.648)
    np.testing.assert_allclose(top.init_state.pos[:2], [-0.25, 0], atol=1e-7)
    np.testing.assert_allclose(top.spawn.size[:2], [1.8, 0.7], atol=1e-7)
    base = scene.patient_table_base
    assert base.init_state.pos[2] - base.spawn.size[2] / 2 == pytest.approx(0.07)
    # Old CT-only bundles keep the previous table behavior.
    stage.RemovePrim("/HumanBody/Exterior/SOMA")
    stage.GetRootLayer().Save()
    assert not fit_table_to_soma(scene, twin)
