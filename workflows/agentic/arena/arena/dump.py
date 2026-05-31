# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero-action debug helpers for frame and scene-pose dumps."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
from common.utils import resolve_path
from PIL import Image

logger = logging.getLogger("arena")

DEFAULT_SCENE_POSE_STEPS = (0, 1, 10, 20, 30)


class SceneDumper:
    def __init__(self, *, output_dir: Path, camera_names: set[str] | None) -> None:
        self.output_dir = output_dir
        self.camera_names = camera_names
        self.manifest: list[dict[str, Any]] = []
        self.frames_by_step: dict[int, list[dict[str, Any]]] = {}
        self.pose_records: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args: Any, *, env_id: str) -> "SceneDumper | None":
        value = getattr(args, "dump_scene", None)
        if value is None:
            return None
        output_dir = resolve_path(value, env_id, default="scene_dumps")
        dumper = cls(output_dir=output_dir, camera_names=_csv_set(getattr(args, "dump_scene_cameras", None)))
        logger.info("zero-action debug dump enabled: %s", dumper.output_dir)
        return dumper

    def dump_frames(self, *, step: int, observation: dict[str, Any] | None, env: Any) -> None:
        cameras = camera_images(observation, env)
        if self.camera_names is not None:
            cameras = {name: image for name, image in cameras.items() if name in self.camera_names}
        if not cameras:
            logger.warning("zero-action debug dump found no cameras at step %s", step)
            return
        records = []
        for camera_name, image in sorted(cameras.items()):
            filename = f"step_{step:04d}_{_safe_name(camera_name)}.jpg"
            Image.fromarray(rgb_uint8(image)).save(self.output_dir / filename, format="JPEG", quality=90)
            record = {"step": step, "camera": camera_name, "path": filename}
            self.manifest.append(record)
            records.append(record)
        self.frames_by_step.setdefault(step, []).extend(records)

    def dump_pose(
        self, ctx: Any, label: str, *, step: int, names: Iterable[str] | None = None, actions: Any = None
    ) -> None:
        images = self.frames_by_step.get(step, [])
        if not images:
            logger.warning("skip zero-action pose dump at step %s because no images were dumped for that step", step)
            return
        record = scene_pose_record(ctx, label, step=step, names=names, actions=actions)
        record["images"] = list(images)
        self.pose_records.append(record)

    def close(self) -> None:
        if not self.manifest:
            raise RuntimeError(f"zero-action scene dump produced no camera frames: {self.output_dir}")
        if self.manifest:
            (self.output_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        if self.pose_records:
            (self.output_dir / "scene_poses.json").write_text(json.dumps(self.pose_records, indent=2) + "\n")
        if self.manifest or self.pose_records:
            logger.info(
                "zero-action debug dump complete: wrote %s frames and %s pose records to %s",
                len(self.manifest),
                len(self.pose_records),
                self.output_dir,
            )


def parse_scene_pose_names(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Parse a comma-separated scene entity list."""
    if value is None:
        return None
    parts = value.split(",") if isinstance(value, str) else value
    names = tuple(str(part).strip() for part in parts if str(part).strip())
    return names or None


def parse_scene_pose_steps(value: str | Iterable[int] | None) -> set[int]:
    """Parse zero-action step indices for pose logging."""
    if value is None:
        return set(DEFAULT_SCENE_POSE_STEPS)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return {int(part) for part in parts} if parts else set(DEFAULT_SCENE_POSE_STEPS)
    return {int(part) for part in value}


def should_dump_scene_step(args: Any, step: int) -> bool:
    """Return whether the current zero-action step should emit pose dumps."""
    if getattr(args, "dump_scene", None) is None:
        return False
    steps = parse_scene_pose_steps(getattr(args, "dump_scene_steps", None))
    return step in steps


def scene_pose_record(
    ctx: Any,
    label: str,
    *,
    step: int,
    names: Iterable[str] | None = None,
    actions: Any | None = None,
) -> dict[str, Any]:
    """Return structured pose data for scene entities if present."""
    scene = ctx.env.unwrapped.scene
    entities: dict[str, Any] = {}
    entity_names = tuple(names) if names is not None else discover_scene_pose_names(scene)
    for name in entity_names:
        try:
            entity = scene[name]
        except KeyError:
            continue
        try:
            pos, quat = scene_entity_pose(entity)
        except Exception as exc:  # noqa: BLE001 - keep debug best-effort
            entities[name] = {"available": False, "error": str(exc)}
            continue
        entities[name] = {"available": True, "position": pos, "quaternion": quat}
    return {
        "step": step,
        "label": label,
        "action_summary": action_summary(actions),
        "entities": entities,
    }


def discover_scene_pose_names(scene: Any) -> tuple[str, ...]:
    """Discover loggable scene entity names from an IsaacLab InteractiveScene."""
    names: list[str] = []

    # IsaacLab InteractiveScene stores entities by category. Attribute names
    # differ across versions, so check the common mapping-like containers.
    for attr in (
        "articulations",
        "rigid_objects",
        "deformable_objects",
        "sensors",
        "extras",
        "_articulations",
        "_rigid_objects",
        "_deformable_objects",
        "_sensors",
        "_extras",
    ):
        mapping = getattr(scene, attr, None)
        if isinstance(mapping, Mapping):
            names.extend(str(key) for key in mapping.keys())

    return tuple(dict.fromkeys(names))


def scene_entity_pose(entity: Any) -> tuple[list[float], list[float]]:
    """Return first-env world pose for common IsaacLab scene entity types."""
    data = getattr(entity, "data", None)
    if data is not None and hasattr(data, "root_pos_w"):
        pos = data.root_pos_w[0].detach().cpu().tolist()
        quat = data.root_quat_w[0].detach().cpu().tolist()
        return pos, quat
    if data is not None and hasattr(data, "pos_w"):
        pos = data.pos_w[0].detach().cpu().tolist()
        quat = data.quat_w[0].detach().cpu().tolist() if hasattr(data, "quat_w") else [1.0, 0.0, 0.0, 0.0]
        return pos, quat
    positions, orientations = entity.get_world_poses()
    pos = positions[0, :3].detach().cpu().tolist()
    quat = orientations[0].detach().cpu().tolist()
    return pos, quat


def action_summary(actions: Any | None) -> dict[str, float]:
    if actions is None:
        return {}
    first_action = actions[0] if actions.ndim > 1 else actions
    if first_action.numel() != 23:
        return {}
    return {"base_height_cmd": float(first_action[19].detach().cpu().item())}


def camera_images(observation: dict[str, Any] | None, env: Any) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    if observation:
        for name, value in (observation.get("camera_obs") or {}).items():
            image = _try_first_env_rgb(name, value)
            if image is not None:
                cameras[str(name)] = image
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    for name, entity in _scene_items(scene):
        output = getattr(getattr(entity, "data", None), "output", None)
        if isinstance(output, dict) and "rgb" in output:
            image = _try_first_env_rgb(name, output["rgb"])
            if image is not None:
                cameras.setdefault(str(name), image)
    return cameras


def _csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def _scene_items(scene: Any) -> list[tuple[str, Any]]:
    if scene is None:
        return []
    keys = getattr(scene, "keys", None)
    if callable(keys):
        try:
            return [(str(name), scene[name]) for name in cast(Iterable[Any], keys())]
        except Exception:
            pass
    items = getattr(scene, "items", None)
    if not callable(items):
        return []
    try:
        return [(str(name), entity) for name, entity in cast(Iterable[tuple[Any, Any]], items())]
    except Exception:
        return []


def _first_env_rgb(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected RGB image with shape HxWxC, got {array.shape}")
    return array[..., :3]


def _try_first_env_rgb(name: Any, value: Any) -> Any | None:
    try:
        return _first_env_rgb(value)
    except ValueError as exc:
        logger.warning("skip non-RGB scene dump image %s: %s", name, exc)
        return None


def rgb_uint8(image: Any) -> Any:
    """Return an RGB image as uint8, accepting float or integer arrays."""
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = array * (255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0)
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8, copy=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "camera"
