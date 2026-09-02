# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate declarative live-scene snapshots for coding-agent utilities."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from i4h_arena.assets.authoring_catalog import AUTHORING_ASSETS, authoring_asset

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
SUPPORTED_KINDS = frozenset({"camera", "cube", "known_asset", "usd"})


def _tuple(values: object, *, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, list | tuple) or len(values) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _absolute_prim_path(value: object, *, label: str) -> str:
    path = str(value)
    if not path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"{label} must be an absolute prim path")
    return path.rstrip("/") or "/"


def normalize_snapshot(raw: object, workflow: str) -> dict[str, Any]:
    """Validate and normalize one live-authoring snapshot."""
    if not isinstance(raw, dict):
        raise TypeError("snapshot must be a JSON object")
    snapshot = dict(raw)
    if snapshot.get("schema_version") != 1:
        raise ValueError("snapshot schema_version must be 1")
    if snapshot.get("workflow") != workflow:
        raise ValueError(f"snapshot belongs to workflow {snapshot.get('workflow')!r}, expected {workflow!r}")
    environment_root = _absolute_prim_path(snapshot.get("environment_root"), label="snapshot environment_root")
    snapshot["environment_root"] = environment_root
    if "world" in snapshot and not isinstance(snapshot["world"], dict):
        raise TypeError("snapshot world must be an object when present")
    items = snapshot.get("items")
    if not isinstance(items, list):
        raise TypeError("snapshot items must be a list")

    names: set[str] = set()
    aliases: set[str] = set()
    relative_paths: set[str] = set()
    robot_count = 0
    robot_preset_name: str | None = None
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(items):
        if not isinstance(source, dict):
            raise TypeError(f"snapshot item {index} must be an object")
        item = dict(source)
        kind = str(item.get("kind", ""))
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"snapshot item {index} has unsupported kind {kind!r}")
        name = str(item.get("name", ""))
        if not IDENTIFIER.fullmatch(name):
            raise ValueError(f"snapshot item {index} has invalid name {name!r}")
        if name in names or name in {"ground", "light", "dome_light"}:
            raise ValueError(f"snapshot contains duplicate or reserved source name {name!r}")
        names.add(name)

        relative_path = str(item.get("relative_prim_path", "")).strip("/")
        if not relative_path or ".." in relative_path.split("/"):
            raise ValueError(f"snapshot item {name!r} has invalid relative_prim_path")
        if relative_path in relative_paths:
            raise ValueError(f"snapshot contains duplicate relative path {relative_path!r}")
        relative_paths.add(relative_path)
        item["relative_prim_path"] = relative_path
        item["prim_path"] = f"{environment_root}/{relative_path}"
        item["position_m"] = _tuple(item.get("position_m"), length=3, label=f"{name}.position_m")
        item["rotation_xyzw"] = _tuple(item.get("rotation_xyzw"), length=4, label=f"{name}.rotation_xyzw")
        quaternion_norm = sum(value * value for value in item["rotation_xyzw"]) ** 0.5
        if not 0.99 <= quaternion_norm <= 1.01:
            raise ValueError(f"snapshot item {name!r} rotation must be a unit quaternion")
        item["scale"] = _tuple(item.get("scale"), length=3, label=f"{name}.scale")
        if any(value <= 0.0 for value in item["scale"]):
            raise ValueError(f"snapshot item {name!r} scale must be positive")
        if item.get("mass_kg") is not None:
            item["mass_kg"] = float(item["mass_kg"])
            if not math.isfinite(item["mass_kg"]) or item["mass_kg"] <= 0.0:
                raise ValueError(f"snapshot item {name!r} mass_kg must be positive")

        if kind == "known_asset":
            preset_name = str(item.get("preset", ""))
            if preset_name not in AUTHORING_ASSETS:
                raise ValueError(f"snapshot item {name!r} has unknown preset {preset_name!r}")
            preset = authoring_asset(preset_name)
            if preset.physics == "articulation":
                robot_count += 1
                robot_preset_name = preset_name
                if preset.embodiment is None:
                    raise ValueError(f"robot preset {preset_name!r} lacks runtime metadata")
                if relative_path != preset.embodiment.runtime_prim_path:
                    raise ValueError(
                        f"robot preset {preset_name!r} must use runtime path "
                        f"{preset.embodiment.runtime_prim_path!r}, got {relative_path!r}"
                    )
                if any(
                    abs(actual - canonical) >= 1.0e-6
                    for actual, canonical in zip(item["scale"], preset.scale, strict=True)
                ):
                    raise ValueError(f"registered robot preset {preset_name!r} does not support a scale override")
        elif kind == "usd":
            usd_path = str(item.get("usd_path", ""))
            if not usd_path or ".." in usd_path.split("/"):
                raise ValueError(f"snapshot item {name!r} has invalid usd_path")
            if item.get("physics") not in {"static", "rigid"}:
                raise ValueError(f"snapshot item {name!r} has invalid physics role")
        elif kind == "cube":
            item["color"] = _tuple(item.get("color"), length=3, label=f"{name}.color")
            if any(value < 0.0 or value > 1.0 for value in item["color"]):
                raise ValueError(f"snapshot item {name!r} color must be within [0, 1]")
            if item.get("physics") not in {"static", "rigid"}:
                raise ValueError(f"snapshot item {name!r} has invalid physics role")
        else:
            alias = str(item.get("alias", ""))
            if not IDENTIFIER.fullmatch(alias):
                raise ValueError(f"snapshot camera {name!r} has invalid alias {alias!r}")
            if alias in aliases:
                raise ValueError(f"snapshot contains duplicate camera alias {alias!r}")
            aliases.add(alias)
            resolution = item.get("resolution")
            if not isinstance(resolution, list | tuple) or len(resolution) != 2:
                raise ValueError(f"snapshot camera {name!r} requires a two-value resolution")
            item["resolution"] = tuple(int(value) for value in resolution)
            if any(value <= 0 for value in item["resolution"]):
                raise ValueError(f"snapshot camera {name!r} resolution must be positive")
            for field in (
                "focal_length",
                "focus_distance",
                "horizontal_aperture",
                "vertical_aperture",
            ):
                item[field] = float(item[field])
                if not math.isfinite(item[field]):
                    raise ValueError(f"snapshot camera {name!r} {field} must be finite")
            for field in ("focal_length", "horizontal_aperture", "vertical_aperture"):
                if item[field] <= 0.0:
                    raise ValueError(f"snapshot camera {name!r} {field} must be positive")
            item["clipping_range_m"] = _tuple(
                item.get("clipping_range_m"),
                length=2,
                label=f"{name}.clipping_range_m",
            )
            if item["clipping_range_m"][0] <= 0.0 or item["clipping_range_m"][1] <= item["clipping_range_m"][0]:
                raise ValueError(f"snapshot camera {name!r} clipping range must be positive with far > near")
        normalized.append(item)

    if robot_count > 1:
        raise ValueError("one authored environment may contain only one embodiment")
    if robot_preset_name is not None:
        embodiment = authoring_asset(robot_preset_name).embodiment
        if embodiment is None:
            raise ValueError(f"robot preset {robot_preset_name!r} lacks runtime metadata")
        conflicts = aliases.intersection(alias for alias, _ in embodiment.camera_aliases)
        if conflicts:
            raise ValueError(f"scene camera aliases conflict with attached robot cameras: {sorted(conflicts)}")
    snapshot["items"] = sorted(normalized, key=lambda item: item["relative_prim_path"])
    return snapshot


def load_snapshot(path: Path, workflow: str) -> dict[str, Any]:
    """Load and validate a snapshot from disk."""
    return normalize_snapshot(json.loads(path.read_text(encoding="utf-8")), workflow)


def robot_item(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Return the snapshot's registered embodiment, if present."""
    for item in snapshot["items"]:
        if item["kind"] == "known_asset" and authoring_asset(item["preset"]).physics == "articulation":
            return item
    return None


def manifest_capabilities(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Derive manifest capability fields from normalized authored content."""
    robot = robot_item(snapshot)
    if robot is None:
        embodiment = "none"
        action_space = "joint_position"
        dof = 0
        gripper = False
        robots: list[str] = []
        attached_cameras: list[str] = []
        control_hz: float | None = None
    else:
        metadata = authoring_asset(robot["preset"]).embodiment
        if metadata is None:
            raise ValueError(f"robot preset {robot['preset']!r} lacks runtime metadata")
        embodiment = metadata.manifest_name
        action_space = metadata.action_space
        dof = metadata.dof
        gripper = metadata.gripper
        robots = [metadata.robot_name]
        attached_cameras = [alias for alias, _ in metadata.camera_aliases]
        control_hz = metadata.control_hz
    scene_cameras = [item["alias"] for item in snapshot["items"] if item["kind"] == "camera"]
    objects = [
        item["name"]
        for item in snapshot["items"]
        if item["kind"] in {"cube", "usd"}
        or (item["kind"] == "known_asset" and authoring_asset(item["preset"]).physics != "articulation")
    ]
    result = {
        "embodiment": embodiment,
        "action_space": action_space,
        "dof": dof,
        "gripper": gripper,
        "cameras": [*attached_cameras, *scene_cameras],
        "objects": objects,
        "robots": robots,
    }
    if control_hz is not None:
        result["control_hz"] = control_hz
    return result
