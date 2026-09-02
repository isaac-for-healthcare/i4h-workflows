# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static validation of a workflow, before anything heavy starts.

Everything here runs off manifests and the graph — no Isaac, no policy stack, no
imports of task implementations. A typo that today costs a 60-second Kit launch
to discover is reported in milliseconds.

The core check is ``requires`` (task) against ``provides`` (scene). Those two
halves are also exactly what an automatic workflow composer would need, which is why
the manifests carry them.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from i4h_common.manifest import SceneSpec
from i4h_engine.executor import autowire
from i4h_engine.graph import TaskGraph
from i4h_engine.loader import ResolvedWorkflow
from i4h_engine.ports import types_compatible


@dataclass(frozen=True, slots=True)
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    node: str = ""

    def __str__(self) -> str:
        marker = "✗" if self.severity == "error" else "!"
        where = f" [{self.node}]" if self.node else ""
        return f"  {marker} {self.message}{where}"


@dataclass
class LintReport:
    workflow: ResolvedWorkflow
    scene: SceneSpec | None
    issues: tuple[Issue, ...]
    autowired: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines: list[str] = []
        if self.scene is not None:
            provides = self.scene.provides()
            lines.append(
                f"  scene {self.scene.name} provides "
                f"embodiment={provides['embodiment']} action_space={provides['action_space']} "
                f"dof={provides['dof']} cameras={provides['cameras']}"
            )
        for wire in self.autowired:
            lines.append(f"  · auto-wired {wire}")
        lines.extend(str(issue) for issue in self.issues)
        if self.ok:
            lines.append(
                f"  ok — {len(self.workflow.graph)} nodes, {len(self.workflow.graph.edges)} edges, "
                f"{len(self.workflow.graph.data_edges)} data edges"
            )
        else:
            lines.append(f"  {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


def lint_workflow(workflow: ResolvedWorkflow, registry: Any | None = None) -> LintReport:
    """Validate ``workflow`` against the registry. Never imports task implementations."""
    if registry is None:
        from i4h_engine.registry import default_registry  # noqa: PLC0415

        registry = default_registry()

    issues: list[Issue] = []
    for error in registry.errors:
        issues.append(Issue("error", "manifest", error))

    graph = workflow.graph
    added = autowire(graph)
    autowired = tuple(f"{e.src} → {e.dst}" for e in added)

    scene = _check_scene(workflow, registry, issues)
    _check_graph(graph, workflow.name, issues)
    _check_data_edges(graph, issues)
    _check_requirements(graph, scene, registry, issues)

    return LintReport(workflow=workflow, scene=scene, issues=tuple(issues), autowired=autowired)


def _check_scene(workflow: ResolvedWorkflow, registry: Any, issues: list[Issue]) -> SceneSpec | None:
    try:
        return registry.scene(workflow.scene).for_mode(workflow.mode)
    except KeyError as exc:
        issues.append(Issue("error", "unknown-scene", str(exc).strip("'")))
        return None


def _check_graph(graph: TaskGraph, workflow_name: str, issues: list[Issue]) -> None:
    if not len(graph):
        issues.append(Issue("error", "empty-workflow", f"workflow {workflow_name!r} has no nodes"))
        return

    cycle = graph.find_cycle()
    if cycle:
        issues.append(Issue("error", "cycle", f"control edges form a cycle: {' → '.join(cycle)}"))
        return

    if not graph.roots():
        issues.append(Issue("error", "no-root", "every node has a predecessor, so nothing can start"))
    if not graph.terminals():
        issues.append(Issue("error", "no-terminal", "every node has a successor, so the workflow never finishes"))

    reachable: set[str] = set()
    frontier = [n.id for n in graph.roots()]
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        frontier.extend(graph.successors(current))
    for node_obj in graph.nodes:
        if node_obj.id not in reachable:
            issues.append(
                Issue("warning", "unreachable", f"{node_obj.id!r} is not reachable from any root", node=node_obj.id)
            )


def _check_data_edges(graph: TaskGraph, issues: list[Issue]) -> None:
    order: dict[str, int] = {}
    with contextlib.suppress(ValueError):
        order = {node_id: index for index, node_id in enumerate(graph.topological_order())}

    for edge in graph.data_edges:
        source = graph.node_by_id(edge.src.node_id)
        target = graph.node_by_id(edge.dst.node_id)
        src_type = source.output_ports.get(edge.src.port)
        dst_type = target.input_ports.get(edge.dst.port)
        if src_type is None:
            issues.append(
                Issue("error", "unknown-port", f"{edge.src} is not an output of {source.task_id}", node=source.id)
            )
            continue
        if dst_type is None:
            issues.append(
                Issue("error", "unknown-port", f"{edge.dst} is not an input of {target.task_id}", node=target.id)
            )
            continue
        if not types_compatible(src_type, dst_type):
            issues.append(
                Issue(
                    "error",
                    "type-mismatch",
                    f"{edge.src} is {src_type} but {edge.dst} expects {dst_type}",
                    node=target.id,
                )
            )
        if order and order.get(edge.src.node_id, 0) >= order.get(edge.dst.node_id, 0):
            issues.append(
                Issue(
                    "error",
                    "data-before-control",
                    f"{edge.dst} reads {edge.src}, but {edge.src.node_id} does not run first",
                    node=target.id,
                )
            )

    for node_obj in graph.nodes:
        wired = {e.dst.port for e in graph.inputs_for(node_obj.id)}
        for name in node_obj.required_inputs:
            if name not in wired and name not in node_obj.params:
                issues.append(
                    Issue(
                        "error",
                        "unwired-input",
                        f"{node_obj.id}.in_.{name} ({node_obj.input_ports[name]}) is required but not wired "
                        f"and has no constructor value",
                        node=node_obj.id,
                    )
                )


def _check_requirements(graph: TaskGraph, scene: SceneSpec | None, registry: Any, issues: list[Issue]) -> None:
    if scene is None:
        return
    provides = scene.provides()
    for node_obj in graph.nodes:
        spec = node_obj.spec
        if spec is None:
            spec = registry.tasks.get(node_obj.task_id)
        if spec is None:
            continue  # locally-instantiated task; the drift test covers it instead
        for key, wanted in spec.requires.items():
            have = provides.get(key)
            if have is None:
                issues.append(
                    Issue(
                        "warning",
                        "unknown-requirement",
                        f"{node_obj.id} requires {key}={wanted!r}, which scene {scene.name} does not describe",
                        node=node_obj.id,
                    )
                )
                continue
            if isinstance(wanted, list):
                missing = [item for item in wanted if item not in have]
                if missing:
                    issues.append(
                        Issue(
                            "error",
                            "requirement",
                            f"{node_obj.id} requires {key}={wanted}, scene {scene.name} provides {have} "
                            f"(missing {missing})",
                            node=node_obj.id,
                        )
                    )
            elif wanted != have:
                issues.append(
                    Issue(
                        "error",
                        "requirement",
                        f"{node_obj.id} requires {key}={wanted!r}, scene {scene.name} provides {have!r}",
                        node=node_obj.id,
                    )
                )

        # A task parameter naming a scene object is the most common typo there is.
        for param, value in node_obj.params.items():
            if (
                param in ("object", "target_object", "name")
                and isinstance(value, str)
                and scene.objects
                and value not in scene.objects
            ):
                issues.append(
                    Issue(
                        "error",
                        "unknown-object",
                        f"{node_obj.id} references object {value!r} — scene {scene.name} has {list(scene.objects)}",
                        node=node_obj.id,
                    )
                )
