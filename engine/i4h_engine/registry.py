# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task and scene discovery.

Tasks come from :mod:`i4h_engine.discover` — imported for in-process skills,
loaded by file path for remote stacks. Scenes stay on YAML, because what a scene
*provides* is not derivable from its class without importing Isaac.

Instantiation happens only in the venv that owns the dependency:

* ``runtime = "inprocess"`` → the class discovery already found
* ``runtime = "remote"``    → a generic :class:`~i4h_engine.remote.RemoteTask`
  proxy, which imports nothing from the backend at all
"""

from __future__ import annotations

import difflib
import importlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from i4h_common.manifest import ManifestError, SceneSpec, TaskSpec, load_scene_manifest
from i4h_common.paths import workflow_root
from i4h_engine.discover import discover_tasks

# Each subsystem owns a manifest/ folder inside the package that defines those
# entities, so the patterns are specific and cannot collide.
SCENE_GLOBS = ("*/*/scenes/manifest",)


@dataclass
class Registry:
    """An index of every task and scene declared under ``root``."""

    root: Path
    tasks: dict[str, TaskSpec] = field(default_factory=dict)
    scenes: dict[str, SceneSpec] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def discover(cls, root: Path | str | None = None) -> Registry:
        root = Path(root) if root is not None else workflow_root()
        registry = cls(root=root)
        registry._scan_tasks()
        registry._scan_scenes()
        return registry

    def _scan_tasks(self) -> None:
        found, errors = discover_tasks(self.root)
        self.tasks.update(found)
        self.errors.extend(errors)

    def _scan_scenes(self) -> None:
        for pattern in SCENE_GLOBS:
            for directory in sorted(self.root.glob(pattern)):
                if ".venv" in directory.parts or not directory.is_dir():
                    continue
                try:
                    specs = load_scene_manifest(directory)
                except ManifestError as exc:
                    self.errors.append(str(exc))
                    continue
                for spec in specs:
                    if spec.name in self.scenes:
                        self.errors.append(
                            f"{spec.source}: duplicate scene {spec.name!r} "
                            f"(already from {self.scenes[spec.name].source})"
                        )
                        continue
                    self.scenes[spec.name] = spec

    # -- lookup ----------------------------------------------------------
    def task(self, task_id: str) -> TaskSpec:
        spec = self.tasks.get(task_id)
        if spec is None:
            raise KeyError(f"unknown task {task_id!r}{self._suggest(task_id, self.tasks)}")
        # Reads the class for in-process tasks; a no-op for remote ones.
        return spec.resolve()

    def scene(self, name: str) -> SceneSpec:
        spec = self.scenes.get(name)
        if spec is None:
            raise KeyError(f"unknown scene {name!r}{self._suggest(name, self.scenes)}")
        return spec

    @staticmethod
    def _suggest(wanted: str, pool: dict[str, Any]) -> str:
        close = difflib.get_close_matches(wanted, pool, n=3, cutoff=0.6)
        if close:
            return f"; did you mean {close}?"
        return f"; known: {sorted(pool) if len(pool) <= 20 else f'{len(pool)} entries'}"

    # -- instantiation ---------------------------------------------------
    def instantiate(self, task_id: str, params: dict[str, Any] | None = None) -> Any:
        """Build a live task object. Imports only for ``inprocess`` tasks."""
        spec = self.task(task_id)
        params = dict(params or {})
        if spec.runtime == "remote":
            from i4h_engine.remote import RemoteTask  # noqa: PLC0415

            return RemoteTask(spec, **params)
        if not spec.impl:
            raise ManifestError(f"{spec.id}: inprocess task has no impl")
        module_name, _, attr = spec.impl.partition(":")
        if not attr:
            raise ManifestError(f"{spec.id}: impl must be 'module:Class', got {spec.impl!r}")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"{spec.id}: cannot import {module_name!r} — is its project installed "
                f"in this venv? (declared in {spec.source})"
            ) from exc
        try:
            cls = getattr(module, attr)
        except AttributeError as exc:
            raise ManifestError(f"{spec.id}: {module_name} has no attribute {attr!r}") from exc
        return cls(**params)

    def summary(self) -> str:
        return (
            f"{len(self.tasks)} tasks from {len({s.project for s in self.tasks.values()})} projects, "
            f"{len(self.scenes)} scenes"
        )


@lru_cache(maxsize=4)
def default_registry(root: str | None = None) -> Registry:
    """Cached registry for the workflow tree. Discovery is cheap but not free."""
    return Registry.discover(root)
