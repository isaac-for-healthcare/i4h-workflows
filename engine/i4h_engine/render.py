# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendering a workflow for humans: an outline, or a mermaid graph.

Runs off the graph alone, so ``run.sh show <workflow>`` works in the light venv and
answers "what will this actually do?" without launching anything.
"""

from __future__ import annotations

from i4h_engine.graph import Node
from i4h_engine.loader import ResolvedWorkflow
from i4h_engine.registry import default_registry


def _effective_prompt(node_obj: Node) -> str:
    if node_obj.spec is not None:
        return node_obj.spec.effective_prompt
    implementation = f"{type(node_obj.ref).__module__}:{type(node_obj.ref).__name__}"
    for spec in default_registry().tasks.values():
        if spec.impl == implementation:
            return spec.effective_prompt
    return ""


def to_text(workflow: ResolvedWorkflow) -> str:
    """Indented outline: nodes in topological order, with their wiring."""
    lines = [f"workflow {workflow.name} (mode={workflow.mode}, scene={workflow.scene})"]
    graph = workflow.graph
    if graph.description:
        lines.append(f"  {graph.description}")

    try:
        order = graph.topological_order()
    except ValueError as exc:
        return "\n".join([*lines, f"  !! {exc}"])

    for node_id in order:
        node_obj = graph.node_by_id(node_id)
        preds = graph.predecessors(node_id)
        marker = "▸" if not preds else "├"
        lines.append(f"  {marker} {node_id}  [{node_obj.task_id}]")
        prompt = _effective_prompt(node_obj)
        if prompt:
            lines.append(f"      prompt: {prompt}")
        if node_obj.params:
            rendered = ", ".join(f"{k}={v!r}" for k, v in node_obj.params.items())
            lines.append(f"      params: {rendered}")
        for edge in graph.inputs_for(node_id):
            lines.append(f"      in  {edge.dst.port} ← {edge.src}")
        for name, type_name in node_obj.output_ports.items():
            lines.append(f"      out {name}: {type_name}")
        policy = node_obj.failure_policy
        if policy.action == "retry":
            lines.append(f"      on failure: retry {policy.times}x")

    roots = ", ".join(n.id for n in graph.roots())
    terminals = ", ".join(n.id for n in graph.terminals())
    lines.append(f"  starts at: {roots}")
    lines.append(f"  finishes when: {terminals}")
    return "\n".join(lines)


def to_mermaid(workflow: ResolvedWorkflow) -> str:
    """Mermaid flowchart. Control edges solid, data edges dotted and labelled."""
    lines = ["```mermaid", "flowchart TD"]
    graph = workflow.graph
    for node_obj in graph.nodes:
        label = (
            node_obj.id if node_obj.id == node_obj.task_id.split("/")[-1] else f"{node_obj.id}<br/>{node_obj.task_id}"
        )
        shape_open, shape_close = ("([", "])") if not graph.predecessors(node_obj.id) else ("[", "]")
        lines.append(f'    {_safe(node_obj.id)}{shape_open}"{label}"{shape_close}')
    for edge in graph.edges:
        lines.append(f"    {_safe(edge.src)} --> {_safe(edge.dst)}")
    for edge in graph.data_edges:
        source = _safe(edge.src.node_id)
        target = _safe(edge.dst.node_id)
        lines.append(f"    {source} -.->|{edge.src.port} → {edge.dst.port}| {target}")
    lines.append("```")
    return "\n".join(lines)


def _safe(node_id: str) -> str:
    return node_id.replace("-", "_").replace(".", "_").replace("/", "_")
