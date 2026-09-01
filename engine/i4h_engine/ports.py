# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed ports and the references that name them.

A workflow has two edge kinds and they are deliberately separate:

* ``a >> b``                       control edge — ordering
* ``workflow.wire(a.out.x, b.in_.y)``  data edge — a typed value handoff

Keeping them apart is what makes "each task gets an input and produces an
output" checkable rather than a convention: the control graph decides *when*
something runs, the data graph decides *what it runs on*, and lint can verify
each independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from i4h_engine.graph import Node


@dataclass(frozen=True, slots=True)
class PortRef:
    """Names one port on one node, e.g. ``locate.out.pose``."""

    node_id: str
    port: str
    direction: str  # "in" | "out"

    def __str__(self) -> str:
        return f"{self.node_id}.{'out' if self.direction == 'out' else 'in'}.{self.port}"


class PortAccessor:
    """Attribute access over a node's declared ports.

    Raises immediately on an unknown port name so a typo surfaces while the workflow
    is being built, not when the node finally runs three minutes into a rollout.
    """

    __slots__ = ("_node", "_direction")

    def __init__(self, node: Node, direction: str) -> None:
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_direction", direction)

    def __getattr__(self, name: str) -> PortRef:
        node = object.__getattribute__(self, "_node")
        direction = object.__getattribute__(self, "_direction")
        declared = node.output_ports if direction == "out" else node.input_ports
        if name not in declared:
            kind = "output" if direction == "out" else "input"
            raise AttributeError(f"{node.id} has no {kind} port {name!r}; declared: {sorted(declared) or '(none)'}")
        return PortRef(node_id=node.id, port=name, direction=direction)

    def __getitem__(self, name: str) -> PortRef:
        return getattr(self, name)

    def __dir__(self) -> list[str]:
        node = object.__getattribute__(self, "_node")
        direction = object.__getattribute__(self, "_direction")
        return sorted(node.output_ports if direction == "out" else node.input_ports)


def types_compatible(source: str, target: str) -> bool:
    """Can a value of declared type ``source`` feed a port of type ``target``?

    Optional inputs carry a trailing ``?``, which affects whether a port *must*
    be wired, not what may be wired into it. Numeric widening (int → float) is
    allowed; nothing else is.
    """
    src = source.removesuffix("?")
    dst = target.removesuffix("?")
    if src == dst:
        return True
    return (src, dst) == ("int", "float")


def coerce_value(value: Any, declared: str) -> Any:
    """Light run-time coercion at a data edge, matching :func:`types_compatible`."""
    if declared.removesuffix("?") == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value
