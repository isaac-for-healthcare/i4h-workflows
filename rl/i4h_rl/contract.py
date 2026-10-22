# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-component validation for an RL profile and its workflow environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from i4h_rl.profile import ProfileError, RLProfile


def _mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"cannot read contract {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: expected a YAML mapping")
    return raw


def validate_workflow_contract(profile: RLProfile, workflows_root: Path) -> None:
    """Verify that common profile facts agree with the Scene the profile trains in."""
    manifest = workflows_root / "arena/i4h_arena/scenes/manifest" / f"{profile.scene}.yaml"
    if not manifest.is_file():
        raise ProfileError(f"{profile.workflow}: Scene manifest does not exist: {manifest}")
    scene = _mapping(manifest)

    if scene.get("dof") != profile.action_dof:
        raise ProfileError(
            f"{profile.workflow}: profile action_dof={profile.action_dof} does not match "
            f"Scene {profile.scene!r} dof={scene.get('dof')!r}"
        )
    scene_cameras = scene.get("cameras", [])
    if not isinstance(scene_cameras, list) or any(not isinstance(camera, str) for camera in scene_cameras):
        raise ProfileError(f"{manifest}: cameras must be a string list")
    missing_cameras = sorted(set(profile.cameras) - set(scene_cameras))
    if missing_cameras:
        raise ProfileError(
            f"{profile.workflow}: profile cameras are missing from Scene {profile.scene!r}: "
            f"{', '.join(missing_cameras)}"
        )
