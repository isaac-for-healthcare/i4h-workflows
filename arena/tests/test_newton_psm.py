# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM lift workflows under the Newton preset (NVBug 6928490).

The PSM embodiment and scene modules need Kit to import, so their wiring is
checked from source; the Newton settings and the mesh handling are plain
modules and are exercised directly.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Sdf, Usd

from i4h_arena import newton_meshes
from i4h_arena.embodiments import _psm_newton

ARENA = Path(__file__).parents[1] / "i4h_arena"


def _psm_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=True)),
        actuators={"psm": SimpleNamespace(armature=0.0), "psm_tool": SimpleNamespace(stiffness=500.0)},
    )


def test_the_arm_is_weightless_through_mujoco_gravity_compensation() -> None:
    """Newton ignores physxRigidBody:disableGravity; the insertion joint sagged 7-8 mm."""
    cfg = _psm_cfg()

    _psm_newton.configure_psm_articulation(cfg)

    assert cfg.spawn.rigid_props.gravcomp == 1.0
    assert cfg.spawn.rigid_props.disable_gravity is True


def test_the_jaw_drive_is_critically_damped_with_armature_on_the_light_links() -> None:
    """Without armature and damping the jaws swung past their limits from the first step."""
    cfg = _psm_cfg()

    _psm_newton.configure_psm_articulation(cfg)

    jaw = cfg.actuators["psm_tool"]
    critical = 2.0 * math.sqrt(jaw.stiffness * jaw.armature)
    assert jaw.damping == pytest.approx(critical, rel=0.1)
    assert jaw.armature > 0.0
    assert cfg.actuators["psm"].armature == _psm_newton.ARM_ARMATURE


@pytest.mark.parametrize("disabled", [True, False])
def test_only_objects_physx_holds_weightless_get_gravity_compensation(disabled: bool) -> None:
    original = SimpleNamespace(disable_gravity=disabled)
    asset = SimpleNamespace(spawn=SimpleNamespace(rigid_props=original))

    _psm_newton.compensate_gravity_if_disabled(asset)

    assert (asset.spawn.rigid_props is not original) is disabled


def _source(path: Path) -> str:
    return path.read_text()


def test_the_psm_applies_newton_settings_and_the_ik_offset_only_when_asked() -> None:
    source = _source(ARENA / "embodiments" / "psm.py")
    assert "if newton:\n            configure_psm_articulation(self.scene_config.robot)" in source
    assert "configure_psm_articulation(self.scene_config.robot_1)" in source
    assert "configure_psm_articulation(self.scene_config.robot_2)" in source
    assert "body_offset=(" in source and "if ik_body_offset is not None" in source


def test_surgical_scenes_select_newton_settings_from_the_preset() -> None:
    text = _source(ARENA / "scenes" / "_surgical.py")
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "SurgicalScene")
    newton = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_newton")
    assert 'getattr(self.args, "presets", None) == "newton"' in ast.get_source_segment(text, newton)
    embodiment = ast.get_source_segment(
        text, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_embodiment")
    )
    assert 'shared["ik_body_offset"] = (0.0, 0.0, _psm_newton.GRASP_DEPTH_M)' in embodiment
    assert 'shared["gripper_close"] = _psm_newton.NEEDLE_GRIPPER_CLOSE' in embodiment


# -- newton_meshes -------------------------------------------------------------
MESH, BOX = 8, 7  # newton.GeoType values
COLLIDE = 2  # newton.ShapeFlags.COLLIDE_SHAPES


def _builder() -> SimpleNamespace:
    return SimpleNamespace(
        body_label=["/World/envs/env_0/Robot/psm_tool_gripper1_link", "/World/envs/env_0/Object"],
        shape_type=[MESH, MESH, MESH, BOX, MESH],
        shape_flags=[COLLIDE, COLLIDE, COLLIDE, COLLIDE, 0],
        shape_body=[0, 1, -1, 1, 1],
    )


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(newton_meshes, "_SUFFIXES", set())
    monkeypatch.setattr(newton_meshes, "_keep_static", False)


def test_a_registered_body_is_decomposed_and_every_other_collision_mesh_hulled(registry) -> None:
    newton_meshes._SUFFIXES.add("/Object")

    decompose, hull = newton_meshes._split(_builder())

    assert decompose == [1]
    assert hull == [0, 2]  # the box and the non-colliding mesh are left alone


def test_static_meshes_can_stay_triangle_meshes(registry) -> None:
    newton_meshes._SUFFIXES.add("/Object")
    newton_meshes._keep_static = True

    assert newton_meshes._split(_builder()) == ([1], [0])


def test_only_the_cloners_blanket_hull_request_is_split(registry) -> None:
    calls: list[tuple] = []

    def original(builder, method, shape_indices, **kwargs):
        calls.append((method, shape_indices, kwargs))
        return set(shape_indices or ())

    approximate = newton_meshes._wrap(original)
    newton_meshes._SUFFIXES.add("/Object")

    approximate(_builder(), "convex_hull", keep_visual_shapes=True)
    approximate(_builder(), "convex_hull", [0])

    assert calls[0] == ("convex_hull", [0, 2], {"keep_visual_shapes": True})
    assert calls[1] == ("coacd", [1], {"keep_visual_shapes": True, "threshold": newton_meshes.COACD_THRESHOLD})
    assert calls[2] == ("convex_hull", [0], {})


def test_payloads_without_a_file_format_plugin_are_found(tmp_path: Path) -> None:
    layer = Sdf.Layer.CreateNew(str(tmp_path / "organs.usda"))
    stage = Usd.Stage.Open(layer)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    stage.DefinePrim("/World/Operating_table", "Xform").GetPayloads().AddPayload("./models/table.obj")
    stage.DefinePrim("/World/Liver", "Xform").GetPayloads().AddPayload("./models/liver.usd")
    layer.Save()

    assert newton_meshes.unreadable_payload_prims(str(tmp_path / "organs.usda")) == ["Operating_table"]


def test_destinations_cover_every_environment_a_regex_spawn_reaches() -> None:
    stage = Usd.Stage.CreateInMemory()
    for env in range(2):
        stage.DefinePrim(f"/World/envs/env_{env}", "Xform")

    assert newton_meshes._destinations(stage, "/World/envs/env_.*/Organs") == [
        "/World/envs/env_0/Organs",
        "/World/envs/env_1/Organs",
    ]
    assert newton_meshes._destinations(stage, "/World/Organs") == ["/World/Organs"]
