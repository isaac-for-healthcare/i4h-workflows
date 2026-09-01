#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply one observable edit to an open ``run.sh --live`` Isaac Sim session.

The one-command/one-edit interface is intentional: agents and developers should
send compound scene requests as an ordered sequence so each change is visible
in the open viewport before the next change starts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT / "arena"))

from i4h_arena.assets.authoring_catalog import AUTHORING_ASSETS, authoring_asset


def _rotation_matrix_xyz(rotation_deg: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    """Return the matrix used by an XYZ Euler transform."""
    x, y, z = (math.radians(value) for value in rotation_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def _expected_rotated_size(
    size: tuple[float, float, float],
    *,
    catalog_rotation: tuple[float, float, float],
    authored_rotation: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a catalog AABB when authoring overrides the preset orientation."""
    if authored_rotation == catalog_rotation:
        return size
    catalog = _rotation_matrix_xyz(catalog_rotation)
    authored = _rotation_matrix_xyz(authored_rotation)
    # Relative rotation = authored * inverse(catalog); rotation inverse is transpose.
    relative = tuple(
        tuple(sum(authored[row][k] * catalog[col][k] for k in range(3)) for col in range(3)) for row in range(3)
    )
    return tuple(sum(abs(relative[row][col]) * size[col] for col in range(3)) for row in range(3))


def _remote_client() -> Path:
    matches = sorted(
        (WORKFLOWS_ROOT / "third_party").glob("IsaacSim-*/skills/isaac-sim-remote/scripts/isaacsim_send.py")
    )
    if len(matches) != 1:
        raise RuntimeError(
            "expected one pinned Isaac Sim remote client; run setup.sh if the "
            f"third-party checkout is missing (found {len(matches)})"
        )
    return matches[0]


def _literal(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _snake_name(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value).strip("_")
    if not value or not value[0].isalpha():
        raise ValueError(f"cannot derive a source identifier from {value!r}")
    return value


def _logical_name(args: argparse.Namespace) -> str:
    if getattr(args, "name", None):
        return args.name
    return _snake_name(args.prim_path.rstrip("/").rsplit("/", 1)[-1])


def _authoring_marker(args: argparse.Namespace, *, kind: str, **values: object) -> str:
    marker = {"kind": kind, "name": _logical_name(args)}
    marker.update(values)
    return json.dumps(marker, separators=(",", ":"), sort_keys=True)


def _send(code: str, timeout: float) -> None:
    subprocess.run(
        [
            sys.executable,
            str(_remote_client()),
            "--timeout",
            str(timeout),
            code,
        ],
        cwd=WORKFLOWS_ROOT,
        check=True,
    )


def _visible_tail(path: str, *, steps: int) -> str:
    return f"""
omni.usd.get_context().get_selection().set_selected_prim_paths([{path!r}], True)
app_utils.update_app(steps={steps})
print("live_edit_complete prim={path}")
"""


def _add_usd_code(args: argparse.Namespace, *, finish: bool = True) -> str:
    marker = getattr(args, "authoring_marker", None)
    if marker is None:
        marker = _authoring_marker(
            args,
            kind="usd",
            usd_path=args.usd_path,
            physics=args.physics,
            mass_kg=args.mass,
        )
    expected_size = getattr(args, "expected_size", None)
    size_validation = ""
    if expected_size is not None:
        size_validation = f"""
size = bounds.GetSize()
expected = {_literal(expected_size)}
ratios = [float(size[i]) / expected[i] for i in range(3)]
print("expected_size=" + str(tuple(expected)))
print("size_ratio=" + str(tuple(ratios)))
if any(ratio < 0.8 or ratio > 1.2 for ratio in ratios):
    raise RuntimeError(
        "authored bounds differ from the known-asset preset by more than 20%: "
        + str(tuple(ratios))
    )
"""
    physics_code = ""
    if args.physics == "rigid":
        mass_code = ""
        if args.mass is not None:
            mass_code = f"""
mass_api = UsdPhysics.MassAPI.Apply(wrapper)
mass_api.CreateMassAttr({args.mass})
"""
        physics_code = f"""
UsdPhysics.RigidBodyAPI.Apply(wrapper)
{mass_code}
"""
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
stage.RemovePrim(path)
wrapper = stage.DefinePrim(path, "Xform")
wrapper.SetCustomDataByKey("i4h_authoring", {_literal(marker)})
asset = stage.DefinePrim(path + "/Asset", "Xform")
asset.GetReferences().AddReference({args.usd_path!r})
xform = UsdGeom.XformCommonAPI(wrapper)
xform.SetTranslate(Gf.Vec3d(*{_literal(args.position)}))
xform.SetRotate(
    Gf.Vec3f(*{_literal(args.rotation)}),
    UsdGeom.XformCommonAPI.RotationOrderXYZ,
)
xform.SetScale(Gf.Vec3f(*{_literal(args.scale)}))
{physics_code}
app_utils.update_app(steps={args.steps})
bounds = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    useExtentsHint=True,
).ComputeWorldBound(wrapper).ComputeAlignedRange()
print("bounds_min=" + str(tuple(bounds.GetMin())))
print("bounds_max=" + str(tuple(bounds.GetMax())))
{size_validation}
""" + (_visible_tail(args.prim_path, steps=args.steps) if finish else "")


def _attached_cameras_code(args: argparse.Namespace, preset) -> str:
    statements: list[str] = []
    for attached in preset.attached_cameras:
        width, height = attached.resolution
        qx, qy, qz, qw = attached.rotation_opengl_xyzw
        statements.append(
            f"""
camera_parent_path = path + "/" + {attached.relative_parent_path!r}
camera_parent = stage.GetPrimAtPath(camera_parent_path)
if not camera_parent or not camera_parent.IsValid():
    raise RuntimeError("missing attached-camera parent: " + camera_parent_path)
camera_path = camera_parent_path + "/" + {attached.prim_name!r}
stage.RemovePrim(camera_path)
camera = UsdGeom.Camera.Define(stage, camera_path)
camera_xform = UsdGeom.Xformable(camera.GetPrim())
camera_xform.ClearXformOpOrder()
camera_xform.AddTranslateOp().Set(Gf.Vec3d(*{_literal(attached.position_m)}))
camera_xform.AddOrientOp().Set(
    Gf.Quatf({qw}, Gf.Vec3f({qx}, {qy}, {qz}))
)
camera.GetFocalLengthAttr().Set({attached.focal_length})
camera.GetFocusDistanceAttr().Set({attached.focus_distance})
camera.GetHorizontalApertureAttr().Set({attached.horizontal_aperture})
camera.GetVerticalApertureAttr().Set({attached.horizontal_aperture * height / width})
camera.GetClippingRangeAttr().Set(Gf.Vec2f(*{_literal(attached.clipping_range_m)}))
print("attached_camera={attached.alias}:" + camera_path)
"""
        )
    return "".join(statements)


def _add_known_asset_code(args: argparse.Namespace) -> str:
    preset = authoring_asset(args.asset)
    if preset.embodiment is not None:
        expected_path = f"/World/envs/env_0/{preset.embodiment.runtime_prim_path}"
        if args.prim_path.rstrip("/") != expected_path:
            raise ValueError(
                f"robot preset {args.asset!r} must use runtime path {expected_path!r}; " f"got {args.prim_path!r}"
            )
    args.authoring_marker = _authoring_marker(
        args,
        kind="known_asset",
        preset=args.asset,
    )
    args.usd_path = preset.usd_path
    args.scale = preset.scale
    args.rotation = args.rotation if args.rotation is not None else preset.rotation_deg
    args.expected_size = _expected_rotated_size(
        preset.canonical_size_m,
        catalog_rotation=preset.rotation_deg,
        authored_rotation=args.rotation,
    )
    args.physics = preset.physics
    args.mass = preset.mass_kg
    asset_code = _add_usd_code(args, finish=not preset.attached_cameras)
    if not preset.attached_cameras:
        return asset_code
    return asset_code + _attached_cameras_code(args, preset) + _visible_tail(args.prim_path, steps=args.steps)


def _transform_code(args: argparse.Namespace) -> str:
    assignments: list[str] = []
    if args.position is not None:
        assignments.append(f"xform.SetTranslate(Gf.Vec3d(*{_literal(args.position)}))")
    if args.rotation is not None:
        assignments.append(
            f"xform.SetRotate(Gf.Vec3f(*{_literal(args.rotation)}), UsdGeom.XformCommonAPI.RotationOrderXYZ)"
        )
    if args.scale is not None:
        assignments.append(f"xform.SetScale(Gf.Vec3f(*{_literal(args.scale)}))")
    if not assignments:
        raise ValueError("set-transform requires --position, --rotation, or --scale")
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from pxr import Gf, UsdGeom
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
prim = stage.GetPrimAtPath(path)
if not prim or not prim.IsValid():
    raise RuntimeError("missing prim: " + path)
xform = UsdGeom.XformCommonAPI(prim)
{chr(10).join(assignments)}
""" + _visible_tail(args.prim_path, steps=args.steps)


def _scale_by_code(args: argparse.Namespace) -> str:
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from pxr import Usd, UsdGeom
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
prim = stage.GetPrimAtPath(path)
if not prim or not prim.IsValid():
    raise RuntimeError("missing prim: " + path)
xform = UsdGeom.XformCommonAPI(prim)
current = xform.GetXformVectors(Usd.TimeCode.Default())[2]
xform.SetScale(current * {args.factor})
print("scale=" + str(tuple(current * {args.factor})))
""" + _visible_tail(args.prim_path, steps=args.steps)


def _add_cube_code(args: argparse.Namespace) -> str:
    size = args.size_xyz if args.size_xyz is not None else (args.size, args.size, args.size)
    marker = _authoring_marker(
        args,
        kind="cube",
        color=args.color,
        physics=args.physics,
        mass_kg=args.mass,
    )
    rigid_code = ""
    if args.physics == "rigid":
        mass_code = ""
        if args.mass is not None:
            mass_code = f"""
mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
mass_api.CreateMassAttr({args.mass})
"""
        rigid_code = f"""
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
{mass_code}
"""
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
stage.RemovePrim(path)
cube = UsdGeom.Cube.Define(stage, path)
cube.GetPrim().SetCustomDataByKey("i4h_authoring", {_literal(marker)})
cube.GetSizeAttr().Set(1.0)
xform = UsdGeom.XformCommonAPI(cube.GetPrim())
xform.SetTranslate(Gf.Vec3d(*{_literal(args.position)}))
xform.SetScale(Gf.Vec3f(*{_literal(size)}))
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
{rigid_code}
material = UsdShade.Material.Define(stage, path + "/Looks/Material")
shader = UsdShade.Shader.Define(stage, path + "/Looks/Material/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(*{_literal(args.color)})
)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
""" + _visible_tail(args.prim_path, steps=args.steps)


def _set_view_code(args: argparse.Namespace) -> str:
    return f"""
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.rendering_manager import ViewportManager
from omni.kit.viewport.utility import get_active_viewport
viewport = get_active_viewport()
viewport.camera_path = "/OmniverseKit_Persp"
ViewportManager.set_camera_view(
    "/OmniverseKit_Persp",
    eye={_literal(args.eye)},
    target={_literal(args.target)},
)
app_utils.update_app(steps={args.steps})
print("live_edit_complete view=/OmniverseKit_Persp")
"""


def _camera_from_view_code(args: argparse.Namespace) -> str:
    marker = _authoring_marker(
        args,
        kind="camera",
        alias=args.alias or _logical_name(args),
        resolution=(args.width, args.height),
    )
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom
stage = omni.usd.get_context().get_stage()
viewport = get_active_viewport()
source_path = str(viewport.camera_path)
source = stage.GetPrimAtPath(source_path)
if not source or not source.IsValid():
    raise RuntimeError("missing active viewport camera: " + source_path)
target_path = {args.prim_path!r}
stage.RemovePrim(target_path)
camera = UsdGeom.Camera.Define(stage, target_path)
camera.GetPrim().SetCustomDataByKey("i4h_authoring", {_literal(marker)})
source_camera = UsdGeom.Camera(source)
matrix = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
xform = UsdGeom.Xformable(camera.GetPrim())
xform.ClearXformOpOrder()
xform.AddTransformOp().Set(matrix)
focal_length = {args.focal_length!r}
if focal_length is None:
    focal_length = source_camera.GetFocalLengthAttr().Get()
focus_distance = {args.focus_distance!r}
if focus_distance is None:
    focus_distance = source_camera.GetFocusDistanceAttr().Get()
horizontal_aperture = {args.horizontal_aperture!r}
if horizontal_aperture is None:
    horizontal_aperture = source_camera.GetHorizontalApertureAttr().Get()
vertical_aperture = {args.vertical_aperture!r}
if vertical_aperture is None:
    vertical_aperture = source_camera.GetVerticalApertureAttr().Get()
clipping_range = {args.clipping_range!r}
if clipping_range is None:
    clipping_range = source_camera.GetClippingRangeAttr().Get()
camera.GetFocalLengthAttr().Set(focal_length)
camera.GetFocusDistanceAttr().Set(focus_distance)
camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
camera.GetVerticalApertureAttr().Set(vertical_aperture)
camera.GetClippingRangeAttr().Set(Gf.Vec2f(*clipping_range))
""" + _visible_tail(args.prim_path, steps=args.steps)


def _export_scene_code(args: argparse.Namespace) -> str:
    baseline_path = args.baseline
    if args.root_path is None and baseline_path is None:
        raise ValueError("export-scene requires --root-path when no workflow authoring baseline exists")
    return (
        """
import json
import os
import omni.usd
from pxr import Gf, Usd, UsdGeom
stage = omni.usd.get_context().get_stage()
seed_by_path = {}
baseline = {}
baseline_path = """
        + repr(baseline_path)
        + """
if baseline_path is not None:
    with open(baseline_path, encoding="utf-8") as stream:
        baseline = json.load(stream)
    if baseline.get("schema_version") != 1 or baseline.get("workflow") != """
        + repr(args.workflow)
        + """:
        raise RuntimeError("invalid authoring baseline: " + baseline_path)
    for seed in baseline.get("items", []):
        seed_by_path[seed["relative_prim_path"]] = seed
root_path = """
        + repr(args.root_path)
        + """ or baseline["environment_root"]
root = stage.GetPrimAtPath(root_path)
if not root or not root.IsValid():
    raise RuntimeError("missing environment root: " + root_path)
for prim in Usd.PrimRange(root):
    marker_raw = prim.GetCustomDataByKey("i4h_authoring")
    if not marker_raw:
        continue
    try:
        marker = json.loads(marker_raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid i4h authoring marker on " + str(prim.GetPath())) from exc
    relative_path = str(prim.GetPath()).removeprefix(root_path + "/")
    seed_by_path[relative_path] = marker
items = []
for relative_path, marker in sorted(seed_by_path.items()):
    prim_path = root_path + "/" + relative_path
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        continue
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    transform = Gf.Transform(matrix)
    quat = transform.GetRotation().GetQuat()
    imag = quat.GetImaginary()
    item = dict(marker)
    item["prim_path"] = prim_path
    item["relative_prim_path"] = relative_path
    item["position_m"] = [float(value) for value in transform.GetTranslation()]
    item["rotation_xyzw"] = [
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(quat.GetReal()),
    ]
    item["scale"] = [float(value) for value in transform.GetScale()]
    if marker["kind"] == "camera":
        camera = UsdGeom.Camera(prim)
        item["focal_length"] = float(camera.GetFocalLengthAttr().Get())
        item["focus_distance"] = float(camera.GetFocusDistanceAttr().Get())
        item["horizontal_aperture"] = float(camera.GetHorizontalApertureAttr().Get())
        item["vertical_aperture"] = float(camera.GetVerticalApertureAttr().Get())
        item["clipping_range_m"] = [
            float(value) for value in camera.GetClippingRangeAttr().Get()
        ]
    items.append(item)
payload = {
    "schema_version": 1,
    "workflow": """
        + repr(args.workflow)
        + """,
    "environment_root": root_path,
    "items": items,
}
if "world" in baseline:
    payload["world"] = baseline["world"]
output_path = """
        + repr(args.output_path)
        + """
parent = os.path.dirname(output_path)
if parent:
    os.makedirs(parent, exist_ok=True)
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\\n")
os.replace(temporary, output_path)
print(
    "live_export_complete workflow="
    + payload["workflow"]
    + " items="
    + str(len(items))
    + " output="
    + output_path
)
"""
    )


def _activate_camera_code(args: argparse.Namespace) -> str:
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from omni.kit.viewport.utility import get_active_viewport
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
prim = stage.GetPrimAtPath(path)
if not prim or not prim.IsValid() or prim.GetTypeName() != "Camera":
    raise RuntimeError("missing camera prim: " + path)
get_active_viewport().camera_path = path
app_utils.update_app(steps={args.steps})
print("live_edit_complete active_camera=" + path)
"""


def _capture_camera_code(args: argparse.Namespace) -> str:
    return f"""
import os
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.widget.viewport.capture import FileCapture
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
prim = stage.GetPrimAtPath(path)
if not prim or not prim.IsValid() or prim.GetTypeName() != "Camera":
    raise RuntimeError("missing camera prim: " + path)
output_path = {args.output_path!r}
parent = os.path.dirname(output_path)
if parent:
    os.makedirs(parent, exist_ok=True)
previous_mtime = os.stat(output_path).st_mtime_ns if os.path.exists(output_path) else None
viewport = get_active_viewport()
viewport.camera_path = path
app_utils.update_app(steps={args.steps})
viewport.schedule_capture(FileCapture(output_path))
app_utils.update_app(steps={args.steps})
if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
    raise RuntimeError("camera capture did not create a non-empty file: " + output_path)
current_mtime = os.stat(output_path).st_mtime_ns
if previous_mtime is not None and current_mtime == previous_mtime:
    raise RuntimeError("camera capture did not update the existing file: " + output_path)
print(
    "live_edit_complete captured_camera="
    + path
    + " output="
    + output_path
    + " bytes="
    + str(os.path.getsize(output_path))
)
"""


def _capture_viewport_code(args: argparse.Namespace) -> str:
    return f"""
import os
from isaacsim.core.experimental.utils import app as app_utils
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.widget.viewport.capture import FileCapture
output_path = {args.output_path!r}
parent = os.path.dirname(output_path)
if parent:
    os.makedirs(parent, exist_ok=True)
previous_mtime = os.stat(output_path).st_mtime_ns if os.path.exists(output_path) else None
viewport = get_active_viewport()
app_utils.update_app(steps={args.steps})
viewport.schedule_capture(FileCapture(output_path))
app_utils.update_app(steps={args.steps})
if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
    raise RuntimeError("viewport capture did not create a non-empty file: " + output_path)
current_mtime = os.stat(output_path).st_mtime_ns
if previous_mtime is not None and current_mtime == previous_mtime:
    raise RuntimeError("viewport capture did not update the existing file: " + output_path)
print(
    "live_edit_complete captured_viewport="
    + str(viewport.camera_path)
    + " output="
    + output_path
    + " bytes="
    + str(os.path.getsize(output_path))
)
"""


def _inspect_code(args: argparse.Namespace) -> str:
    return f"""
import omni.usd
from pxr import Usd, UsdGeom
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
prim = stage.GetPrimAtPath(path)
if not prim or not prim.IsValid():
    raise RuntimeError("missing prim: " + path)
bounds = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    useExtentsHint=True,
).ComputeWorldBound(prim).ComputeAlignedRange()
print("prim=" + path)
print("type=" + prim.GetTypeName())
print("bounds_min=" + str(tuple(bounds.GetMin())))
print("bounds_max=" + str(tuple(bounds.GetMax())))
print("children=" + str([(p.GetName(), p.GetTypeName()) for p in prim.GetChildren()]))
"""


def _remove_code(args: argparse.Namespace) -> str:
    return f"""
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
stage = omni.usd.get_context().get_stage()
path = {args.prim_path!r}
if not stage.GetPrimAtPath(path).IsValid():
    raise RuntimeError("missing prim: " + path)
stage.RemovePrim(path)
omni.usd.get_context().get_selection().set_selected_prim_paths([], True)
app_utils.update_app(steps={args.steps})
print("live_edit_complete removed=" + path)
"""


def _vec3(raw: str) -> tuple[float, float, float]:
    values = tuple(float(value.strip()) for value in raw.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return values


def _vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(value) for value in raw.split(","))
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers")
    if values[0] <= 0.0 or values[1] <= values[0]:
        raise argparse.ArgumentTypeError("expected positive near,far values with far > near")
    return values


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=20, help="render updates after this edit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_usd = subparsers.add_parser("add-usd", help="add one transformed USD reference")
    add_usd.add_argument("--prim-path", required=True)
    add_usd.add_argument("--usd-path", required=True)
    add_usd.add_argument("--position", type=_vec3, default=(0.0, 0.0, 0.0))
    add_usd.add_argument("--rotation", type=_vec3, default=(0.0, 0.0, 0.0))
    add_usd.add_argument("--scale", type=_vec3, default=(1.0, 1.0, 1.0))
    add_usd.add_argument("--name")
    add_usd.add_argument("--physics", choices=("static", "rigid"), default="static")
    add_usd.add_argument("--mass", type=_positive_float)
    add_usd.set_defaults(code_builder=_add_usd_code)

    known = subparsers.add_parser("add-known-asset", help="add one asset with its canonical scale")
    known.add_argument("--asset", choices=sorted(AUTHORING_ASSETS), required=True)
    known.add_argument("--prim-path", required=True)
    known.add_argument("--position", type=_vec3, default=(0.0, 0.0, 0.0))
    known.add_argument("--rotation", type=_vec3)
    known.add_argument("--name")
    known.set_defaults(code_builder=_add_known_asset_code)

    transform = subparsers.add_parser("set-transform", help="change one live prim transform")
    transform.add_argument("--prim-path", required=True)
    transform.add_argument("--position", type=_vec3)
    transform.add_argument("--rotation", type=_vec3)
    transform.add_argument("--scale", type=_vec3)
    transform.set_defaults(code_builder=_transform_code)

    scale_by = subparsers.add_parser("scale-by", help="multiply one live prim's current scale")
    scale_by.add_argument("--prim-path", required=True)
    scale_by.add_argument("--factor", type=_positive_float, required=True)
    scale_by.set_defaults(code_builder=_scale_by_code)

    add_cube = subparsers.add_parser("add-cube", help="add one colored collision box")
    add_cube.add_argument("--prim-path", required=True)
    add_cube.add_argument("--position", type=_vec3, default=(0.0, 0.0, 0.0))
    add_cube.add_argument("--size", type=_positive_float, default=0.1, help="edge length in metres")
    add_cube.add_argument(
        "--size-xyz",
        type=_vec3,
        help="box dimensions in metres; overrides --size and requires three positive values",
    )
    add_cube.add_argument("--color", type=_vec3, default=(1.0, 0.0, 0.0))
    add_cube.add_argument("--name")
    add_cube.add_argument("--physics", choices=("static", "rigid"), default="static")
    add_cube.add_argument("--mass", type=_positive_float)
    add_cube.set_defaults(code_builder=_add_cube_code)

    set_view = subparsers.add_parser("set-view", help="move and activate the perspective camera")
    set_view.add_argument("--eye", type=_vec3, required=True)
    set_view.add_argument("--target", type=_vec3, required=True)
    set_view.set_defaults(code_builder=_set_view_code)

    camera = subparsers.add_parser("camera-from-view", help="copy the active view into a camera")
    camera.add_argument("--prim-path", required=True)
    camera.add_argument("--focal-length", type=_positive_float)
    camera.add_argument("--focus-distance", type=float)
    camera.add_argument("--horizontal-aperture", type=_positive_float)
    camera.add_argument("--vertical-aperture", type=_positive_float)
    camera.add_argument("--clipping-range", type=_vec2)
    camera.add_argument("--alias")
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=480)
    camera.set_defaults(code_builder=_camera_from_view_code)

    export_scene = subparsers.add_parser(
        "export-scene",
        help="write all tagged live edits to one deterministic JSON snapshot",
    )
    export_scene.add_argument("--workflow", required=True)
    export_scene.add_argument("--output-path", required=True)
    export_scene.add_argument(
        "--root-path",
        help="environment root; required when the workflow has no authoring baseline",
    )
    export_scene.add_argument("--baseline")
    export_scene.set_defaults(code_builder=_export_scene_code)

    activate = subparsers.add_parser("activate-camera", help="show a camera in the active viewport")
    activate.add_argument("--prim-path", required=True)
    activate.set_defaults(code_builder=_activate_camera_code)

    capture = subparsers.add_parser(
        "capture-camera",
        help="activate a camera and synchronously capture its viewport image",
    )
    capture.add_argument("--prim-path", required=True)
    capture.add_argument("--output-path", required=True)
    capture.set_defaults(code_builder=_capture_camera_code)

    capture_viewport = subparsers.add_parser(
        "capture-viewport",
        help="synchronously capture the currently active viewport",
    )
    capture_viewport.add_argument("--output-path", required=True)
    capture_viewport.set_defaults(code_builder=_capture_viewport_code)

    inspect = subparsers.add_parser("inspect", help="print one live prim's type and world bounds")
    inspect.add_argument("--prim-path", required=True)
    inspect.set_defaults(code_builder=_inspect_code)

    remove = subparsers.add_parser("remove", help="remove one live prim")
    remove.add_argument("--prim-path", required=True)
    remove.set_defaults(code_builder=_remove_code)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "size_xyz", None) is not None and any(value <= 0.0 for value in args.size_xyz):
        raise SystemExit("--size-xyz requires three positive values")
    _send(args.code_builder(args), args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
