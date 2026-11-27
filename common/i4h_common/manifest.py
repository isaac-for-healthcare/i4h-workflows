# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene manifests: ``arena/i4h_arena/scenes/manifest/<scene>.yaml``.

What a scene *provides* cannot be derived from its class without importing Isaac,
so it is declared as data and read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from i4h_common.annotate import Annotate


class ManifestError(ValueError):
    """A manifest is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """What a scene promises, so lint can match tasks to it.

    Scene *structure* stays Python in ``arena/i4h_arena/scenes/``; only the
    capability surface lives here, because lint must read it without Isaac.
    """

    name: str
    impl: str
    embodiment: str
    action_space: str
    dof: int
    cameras: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    robots: tuple[str, ...] = ("robot",)
    gripper: bool = True
    max_steps: int = 600
    control_hz: float = 60.0
    description: str = ""
    #: How a vision model grades this scene's recordings, whoever produced them.
    annotate: Annotate | None = None
    source: Path | None = None

    def provides(self) -> dict[str, Any]:
        """Capability map, matched against a task's ``requires``."""
        return {
            "embodiment": self.embodiment,
            "action_space": self.action_space,
            "dof": self.dof,
            "cameras": list(self.cameras),
            "objects": list(self.objects),
            "robots": list(self.robots),
            "gripper": self.gripper,
        }


def load_scene_spec(path: Path) -> SceneSpec:
    """Read one scene manifest. The filename is the scene name."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: expected a mapping")

    declared = raw.get("name")
    if declared is not None and str(declared) != path.stem:
        raise ManifestError(f"{path}: declares name {declared!r} but the file is named {path.stem!r}")
    for required in ("impl", "embodiment", "action_space", "dof"):
        if required not in raw:
            raise ManifestError(f"{path}: missing {required}")
    cameras = tuple(raw.get("cameras", ()) or ())
    annotate = None
    if raw.get("annotate") is not None:
        if not isinstance(raw["annotate"], dict):
            raise ManifestError(f"{path}: annotate must be a mapping")
        try:
            annotate = Annotate(**raw["annotate"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{path}: annotate: {exc}") from exc
        unrecorded = [name for name in annotate.cameras if name not in cameras]
        if unrecorded:
            raise ManifestError(
                f"{path}: annotate grades cameras {unrecorded} the scene does not record {list(cameras)}"
            )

    return SceneSpec(
        name=path.stem,
        impl=str(raw["impl"]),
        embodiment=str(raw["embodiment"]),
        action_space=str(raw["action_space"]),
        dof=int(raw["dof"]),
        cameras=cameras,
        objects=tuple(raw.get("objects", ()) or ()),
        robots=tuple(raw["robots"] or ()) if "robots" in raw else ("robot",),
        gripper=bool(raw.get("gripper", True)),
        max_steps=int(raw.get("max_steps", 600)),
        control_hz=float(raw.get("control_hz", 60.0)),
        description=str(raw.get("description", "")),
        annotate=annotate,
        source=path,
    )


def load_scene_manifest(directory: Path) -> tuple[SceneSpec, ...]:
    """Read every scene manifest in a ``manifest/`` directory."""
    if not directory.is_dir():
        raise ManifestError(f"{directory}: no such manifest directory")
    specs = [load_scene_spec(path) for path in sorted(directory.glob("*.yaml"))]
    if not specs:
        raise ManifestError(f"{directory}: no scene manifests")
    return tuple(specs)
