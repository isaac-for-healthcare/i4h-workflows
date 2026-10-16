# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry records for tasks and scenes.

:class:`TaskSpec` is what the registry hands to lint, workflows and the engine. It
is built by :mod:`i4h_engine.discover` from a :class:`~i4h_common.taskdef.TaskDef`
plus what the file's location already implies — project, name, runtime.

Scenes are read here directly: what a scene *provides* cannot be derived from
its class without importing Isaac, so it is declared as data.
"""

from __future__ import annotations

import importlib
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """A manifest is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """How to launch the process serving a remote task."""

    project: str
    entry: str


@dataclass(slots=True)
class TaskSpec:
    """One task, as the registry sees it."""

    project: str
    name: str
    runtime: str
    summary: str = ""
    prompt: str = ""
    impl: str | None = None
    backend: BackendSpec | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    requires: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    pre: dict[str, Any] = field(default_factory=dict)
    post: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None
    _trainable: bool = False
    _resolved: bool = False

    @property
    def id(self) -> str:
        """Registry key: ``basic/grasp``, ``gr00t_n15/scissor_pick_and_place``."""
        return f"{self.project}/{self.name}"

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def effective_prompt(self) -> str:
        """Detailed prompt when declared, otherwise the task summary."""
        return self.prompt or self.summary

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Inputs without a ``?`` suffix — must be wired or given a value."""
        return tuple(name for name, declared in self.inputs.items() if not declared.endswith("?"))

    def resolve(self) -> TaskSpec:
        """Fill what an in-process manifest left to the class.

        In-process manifests carry ``impl`` plus author-facing metadata. Ports
        and scene requirements live on the class and are filled here on demand.
        Idempotent, and a no-op for remote tasks.
        """
        if self._resolved:
            return self
        self._resolved = True
        if not self.inputs:
            self.inputs = ports_of(self._class_attr("Inputs", None))
        if not self.outputs:
            self.outputs = ports_of(self._class_attr("Outputs", None), optional_marker=False)
        if not self.requires:
            self.requires = dict(self._class_attr("requires", {}))
        if not self.pre:
            self.pre = dict(self._class_attr("precondition", {}))
        if not self.post:
            self.post = dict(self._class_attr("postcondition", {}))
        return self

    def _class_attr(self, name: str, default: Any) -> Any:
        if not self.impl:
            return default
        module_name, _, attribute = self.impl.partition(":")
        try:
            cls = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ManifestError(
                f"{self.id}: cannot read {name} — {self.impl} is not importable here ({exc}). "
                f"Is {self.project} installed in this venv?"
            ) from exc
        return getattr(cls, name, default)


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
    mode_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
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

    def for_mode(self, mode: str | None) -> SceneSpec:
        """Resolve controller-specific capabilities without creating another scene."""
        values = self.mode_overrides.get(mode or "")
        if not values:
            return self
        return replace(
            self,
            embodiment=str(values.get("embodiment", self.embodiment)),
            action_space=str(values.get("action_space", self.action_space)),
            dof=int(values.get("dof", self.dof)),
            cameras=tuple(values.get("cameras", self.cameras) or ()),
            objects=tuple(values.get("objects", self.objects) or ()),
            robots=tuple(values.get("robots", self.robots) or ()),
            gripper=bool(values.get("gripper", self.gripper)),
            max_steps=int(values.get("max_steps", self.max_steps)),
            control_hz=float(values.get("control_hz", self.control_hz)),
            description=str(values.get("description", self.description)),
        )


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
    mode_overrides = raw.get("mode_overrides", {}) or {}
    if not isinstance(mode_overrides, dict) or any(not isinstance(value, dict) for value in mode_overrides.values()):
        raise ManifestError(f"{path}: mode_overrides must map mode names to mappings")
    allowed_override_keys = {
        "embodiment",
        "action_space",
        "dof",
        "cameras",
        "objects",
        "robots",
        "gripper",
        "max_steps",
        "control_hz",
        "description",
    }
    for mode, values in mode_overrides.items():
        unknown = set(values) - allowed_override_keys
        if unknown:
            raise ManifestError(f"{path}: mode_overrides.{mode} has unknown keys: {sorted(unknown)}")

    return SceneSpec(
        name=path.stem,
        impl=str(raw["impl"]),
        embodiment=str(raw["embodiment"]),
        action_space=str(raw["action_space"]),
        dof=int(raw["dof"]),
        cameras=tuple(raw.get("cameras", ()) or ()),
        objects=tuple(raw.get("objects", ()) or ()),
        robots=tuple(raw["robots"] or ()) if "robots" in raw else ("robot",),
        gripper=bool(raw.get("gripper", True)),
        max_steps=int(raw.get("max_steps", 600)),
        control_hz=float(raw.get("control_hz", 60.0)),
        description=str(raw.get("description", "")),
        mode_overrides={str(mode): dict(values) for mode, values in mode_overrides.items()},
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


def normalize_type(annotation: object) -> tuple[str, bool]:
    """Reduce a field annotation to ``(base_type, is_optional)``.

    Every module here uses ``from __future__ import annotations``, so
    annotations arrive as strings — ``Pose | None`` literally, not as a type.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    text = text.strip()
    optional = False
    if text.startswith("Optional[") and text.endswith("]"):
        text, optional = text[len("Optional[") : -1].strip(), True
    parts = [part.strip() for part in text.split("|")]
    if len(parts) > 1:
        present = [part for part in parts if part not in ("None", "NoneType")]
        optional = optional or len(present) < len(parts)
        text = present[0] if present else "None"
    return text, optional


def ports_of(dataclass_type: type | None, *, optional_marker: bool = True) -> dict[str, str]:
    """Map an ``Inputs``/``Outputs`` dataclass to port types.

    ``?`` answers one question — must this be wired before the node runs? — and
    that only applies to inputs. ``on_exit`` always returns the whole ``Outputs``
    dataclass, so pass ``optional_marker=False`` for outputs.
    """
    if dataclass_type is None or not is_dataclass(dataclass_type):
        return {}
    ports: dict[str, str] = {}
    for entry in fields(dataclass_type):
        base, nullable = normalize_type(entry.type)
        defaulted = entry.default is not MISSING or entry.default_factory is not MISSING  # type: ignore[misc]
        ports[entry.name] = f"{base}?" if (optional_marker and (nullable or defaulted)) else base
    return ports
