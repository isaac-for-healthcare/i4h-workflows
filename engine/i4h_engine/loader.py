# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover authored workflows and resolve one mode into a runnable task graph.

Every ``workflows/i4h_workflows/<specialty>/<name>.py`` module exports one
author-facing ``WORKFLOW`` value. Selecting a mode invokes one of its builders
and produces a :class:`~i4h_engine.graph.TaskGraph`. The resulting
:class:`ResolvedWorkflow` adds the filename-derived name and selected mode
needed by lint, Arena, and logs.

Listing scans the specialty directories. Resolving loads only the selected
workflow source file, which remains free of Isaac and policy-stack imports.
Specialty directory names are organizational labels and may contain hyphens;
they are deliberately not Python import namespaces.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from i4h_common.paths import workflow_root
from i4h_engine.graph import TaskGraph
from i4h_engine.interface import Workflow

WORKFLOWS_PACKAGE = "i4h_workflows"


@dataclass(frozen=True, slots=True)
class ResolvedWorkflow:
    """One authored Workflow with a selected mode and built TaskGraph."""

    name: str
    scene: str
    mode: str
    graph: TaskGraph

    @property
    def max_steps(self) -> int | None:
        return self.graph.max_steps

    @property
    def description(self) -> str:
        return self.graph.description


@dataclass(frozen=True, slots=True)
class WorkflowModule:
    """A discovered Python module and its single exported Workflow value."""

    name: str
    module: ModuleType

    @property
    def workflow(self) -> Workflow:
        try:
            authored = self.module.WORKFLOW
        except AttributeError as exc:
            raise AttributeError(f"workflow {self.name!r} must export WORKFLOW") from exc
        if not isinstance(authored, Workflow):
            raise TypeError(
                f"workflow {self.name!r} exports WORKFLOW as {type(authored).__name__}, "
                "expected i4h_engine.interface.Workflow"
            )
        return authored

    @property
    def scene(self) -> str:
        return self.workflow.scene

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self.workflow.modes)

    @property
    def default_mode(self) -> str:
        return self.workflow.default_mode

    @property
    def description(self) -> str:
        return (self.module.__doc__ or "").strip().splitlines()[0] if self.module.__doc__ else ""


def _workflow_sources(root: Path | None = None) -> dict[str, Path]:
    """Map public workflow names to source files in specialty directories."""
    directory = (root or workflow_root()) / "workflows" / WORKFLOWS_PACKAGE
    if not directory.is_dir():
        return {}

    sources: dict[str, Path] = {}
    for specialty in sorted(directory.iterdir()):
        if not specialty.is_dir() or specialty.name.startswith("_"):
            continue
        for path in sorted(specialty.glob("*.py")):
            if path.stem.startswith("_"):
                continue
            previous = sources.get(path.stem)
            if previous is not None:
                raise ValueError(
                    f"duplicate workflow {path.stem!r}: "
                    f"{previous.relative_to(directory)} and {path.relative_to(directory)}"
                )
            sources[path.stem] = path
    return sources


def available_workflows(root: Path | None = None) -> tuple[str, ...]:
    """Workflow names, discovered without importing their source modules."""
    return tuple(sorted(_workflow_sources(root)))


def load_workflow_module(name: str, root: Path | None = None) -> WorkflowModule:
    """Load one authored workflow source file from its specialty directory."""
    sources = _workflow_sources(root)
    if name not in sources:
        import difflib  # noqa: PLC0415

        known = tuple(sorted(sources))
        close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
        hint = f"; did you mean {close}?" if close else f"; known workflows: {list(known)}"
        raise KeyError(f"unknown workflow {name!r}{hint}")

    source = sources[name]
    digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    module_name = f"{WORKFLOWS_PACKAGE}._authored_{digest}_{name}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load workflow {name!r} from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return WorkflowModule(name=name, module=module)


def resolve_workflow(
    name: str,
    mode: str | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> ResolvedWorkflow:
    """Select one authored mode and build its runnable TaskGraph."""
    source = load_workflow_module(name, root)
    authored = source.workflow
    chosen = mode or authored.default_mode
    if chosen not in authored.modes:
        raise KeyError(f"workflow {name!r} has no mode {chosen!r}; available: {sorted(authored.modes)}")
    graph = authored.modes[chosen](**kwargs)
    if not isinstance(graph, TaskGraph):
        raise TypeError(f"{name}[{chosen}] must build a TaskGraph, got {type(graph).__name__}")
    return ResolvedWorkflow(name=name, scene=authored.scene, mode=chosen, graph=graph)


def apply_overrides(
    graph: TaskGraph,
    *,
    checkpoint: str | None = None,
    prompt: str | None = None,
    extra: dict[str, Any] | None = None,
    model_device: str | None = None,
) -> tuple[str, ...]:
    """Push runtime overrides onto nodes that declare support for them."""
    if not (checkpoint or prompt or extra or model_device):
        return ()
    touched: list[str] = []
    for node_obj in graph.nodes:
        spec = node_obj.spec
        if spec is None:
            continue
        changed = False
        if checkpoint and (spec.runtime == "remote" or bool(spec.model)):
            node_obj.params["checkpoint"] = checkpoint
            changed = True
        if model_device and spec.runtime == "inprocess" and bool(spec.model):
            node_obj.params["device"] = model_device
            changed = True
        if prompt and spec.runtime == "remote":
            node_obj.params["prompt"] = prompt
            changed = True
        if extra and spec.runtime == "remote":
            node_obj.params.update(extra)
            changed = True
        if changed:
            touched.append(node_obj.id)
    return tuple(touched)
