# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "live_scene_edit.py"
    spec = importlib.util.spec_from_file_location("live_scene_edit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_usd_uses_transform_wrapper_and_visible_update() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-usd",
            "--prim-path",
            "/World/envs/env_0/Table",
            "--usd-path",
            "https://example.test/table.usd",
            "--position",
            "1,2,3",
            "--scale",
            "0.5,0.5,0.5",
        ]
    )

    code = args.code_builder(args)

    assert 'wrapper = stage.DefinePrim(path, "Xform")' in code
    assert 'asset = stage.DefinePrim(path + "/Asset", "Xform")' in code
    assert "asset.GetReferences().AddReference" in code
    assert "set_selected_prim_paths" in code
    assert "app_utils.update_app" in code
    assert "bounds_min=" in code


def test_one_invocation_parses_one_operation() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "set-transform",
            "--prim-path",
            "/World/envs/env_0/Robot",
            "--position=-4.64,0,0.8",
        ]
    )

    assert args.command == "set-transform"
    assert args.position == (-4.64, 0.0, 0.8)
    assert args.rotation is None
    assert args.scale is None


def test_red_cube_and_relative_scale_are_separate_operations() -> None:
    module = _load_script()
    cube_args = module.build_parser().parse_args(
        [
            "add-cube",
            "--prim-path",
            "/World/envs/env_0/RedCube",
            "--position",
            "0,0,0.3",
            "--size",
            "0.1",
        ]
    )
    scale_args = module.build_parser().parse_args(
        [
            "scale-by",
            "--prim-path",
            "/World/envs/env_0/RedCube",
            "--factor",
            "2",
        ]
    )

    cube_code = cube_args.code_builder(cube_args)
    scale_code = scale_args.code_builder(scale_args)

    assert "UsdGeom.Cube.Define" in cube_code
    assert '"diffuseColor"' in cube_code
    assert "UsdPhysics.CollisionAPI.Apply" in cube_code
    assert "current * 2.0" in scale_code
    assert cube_args.command == "add-cube"
    assert scale_args.command == "scale-by"


def test_rigid_cube_applies_live_rigid_body_and_mass() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-cube",
            "--prim-path",
            "/World/envs/env_0/Block",
            "--physics",
            "rigid",
            "--mass",
            "0.25",
        ]
    )

    code = args.code_builder(args)

    assert "UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())" in code
    assert "mass_api.CreateMassAttr(0.25)" in code
    assert '\\"physics\\":\\"rigid\\"' in code


def test_cube_supports_rectangular_dimensions() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-cube",
            "--prim-path",
            "/World/envs/env_0/TrainingPad",
            "--size-xyz",
            "0.45,0.35,0.04",
        ]
    )

    code = args.code_builder(args)

    assert "xform.SetScale(Gf.Vec3f(*[0.45,0.35,0.04]))" in code


def test_known_asset_uses_catalog_scale_and_bounds_guard() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-known-asset",
            "--asset",
            "surgical_table",
            "--prim-path",
            "/World/envs/env_0/Table",
        ]
    )

    code = args.code_builder(args)

    assert "Props/Table/table.usd" in code
    assert "xform.SetScale(Gf.Vec3f(*[0.7,0.7,0.52]))" in code
    assert "size_ratio=" in code
    assert "more than 20%" in code
    assert 'SetCustomDataByKey("i4h_authoring"' in code
    assert '\\"kind\\":\\"known_asset\\"' in code
    assert '\\"preset\\":\\"surgical_table\\"' in code


def test_known_asset_bounds_guard_accounts_for_authored_rotation() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-known-asset",
            "--asset",
            "g1",
            "--prim-path",
            "/World/envs/env_0/Robot",
            "--rotation",
            "0,0,90",
        ]
    )

    code = args.code_builder(args)
    expected = module._expected_rotated_size(
        (0.495829, 0.371414, 1.322845),
        catalog_rotation=(0.0, 0.0, 0.0),
        authored_rotation=(0.0, 0.0, 90.0),
    )

    assert expected == pytest.approx((0.371414, 0.495829, 1.322845))
    assert f"expected = {module._literal(expected)}" in code


def test_known_rigid_asset_applies_catalog_mass_live() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-known-asset",
            "--asset",
            "scissors",
            "--prim-path",
            "/World/envs/env_0/Scissors",
        ]
    )

    code = args.code_builder(args)

    assert "UsdPhysics.RigidBodyAPI.Apply(wrapper)" in code
    assert "mass_api.CreateMassAttr(0.15)" in code


def test_names_are_derived_generically_without_asset_or_camera_special_cases() -> None:
    module = _load_script()
    scissors = module.build_parser().parse_args(
        [
            "add-known-asset",
            "--asset",
            "scissors",
            "--prim-path",
            "/World/envs/env_0/SurgicalScissors",
        ]
    )
    camera = module.build_parser().parse_args(
        [
            "camera-from-view",
            "--prim-path",
            "/World/envs/env_0/RoomCamera",
        ]
    )

    assert module._logical_name(scissors) == "surgical_scissors"
    assert module._logical_name(camera) == "room_camera"


def test_g1_known_asset_adds_its_standard_head_camera() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-known-asset",
            "--asset",
            "g1",
            "--prim-path",
            "/World/envs/env_0/Robot",
        ]
    )

    code = args.code_builder(args)

    assert "camera_parent_path = path + \"/\" + 'Asset/head_link'" in code
    assert "camera_path = camera_parent_path + \"/\" + 'RobotHeadCam'" in code
    assert "camera.GetFocalLengthAttr().Set(15.0)" in code
    assert "attached_camera=head:" in code


def test_registered_robot_rejects_a_noncanonical_runtime_path() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "add-known-asset",
            "--asset",
            "g1",
            "--prim-path",
            "/World/envs/env_0/G1",
        ]
    )

    with pytest.raises(ValueError, match="must use runtime path.*Robot"):
        args.code_builder(args)


def test_capture_camera_uses_synchronous_viewport_file_capture() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "capture-camera",
            "--prim-path",
            "/World/envs/env_0/RoomCamera",
            "--output-path",
            "/tmp/room.png",
        ]
    )

    code = args.code_builder(args)

    assert "FileCapture" in code
    assert "viewport.schedule_capture" in code
    assert "capture_viewport_screenshot_async" not in code
    assert "camera capture did not create a non-empty file" in code


def test_camera_from_view_copies_optics_unless_explicitly_overridden() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "camera-from-view",
            "--prim-path",
            "/World/envs/env_0/RoomCamera",
        ]
    )

    code = args.code_builder(args)

    compile(code, "camera_from_view_remote.py", "exec")
    assert "source_camera.GetFocalLengthAttr().Get()" in code
    assert "source_camera.GetClippingRangeAttr().Get()" in code
    assert "focal_length = None" in code
    assert "Gf.Vec2f(0.1, 100000.0)" not in code


def test_export_scene_serializes_tagged_live_edits() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "export-scene",
            "--workflow",
            "my_workflow",
            "--output-path",
            "/tmp/my_workflow.json",
            "--baseline",
            "/tmp/previous_live_scene.json",
        ]
    )

    code = args.code_builder(args)

    compile(code, "export_scene_remote.py", "exec")
    assert 'GetCustomDataByKey("i4h_authoring")' in code
    assert "GetLocalTransformation" in code
    assert '"schema_version": 1' in code
    assert "\"workflow\": 'my_workflow'" in code
    assert "seed_by_path" in code
    assert "/tmp/previous_live_scene.json" in code
    assert 'baseline["environment_root"]' in code
    assert 'payload["world"] = baseline["world"]' in code
    assert "os.replace(temporary, output_path)" in code


def test_first_export_requires_explicit_environment_root(tmp_path: Path) -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "export-scene",
            "--workflow",
            "no_baseline",
            "--output-path",
            str(tmp_path / "scene.json"),
        ]
    )

    try:
        args.code_builder(args)
    except ValueError as exc:
        assert "requires --root-path" in str(exc)
    else:
        raise AssertionError("expected root-free first export to be rejected")
