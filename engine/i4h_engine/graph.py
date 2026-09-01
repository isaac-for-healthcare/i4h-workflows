# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A task graph: nodes, control edges, and typed data edges.

Authoring is Python, not a schema, because we already need arbitrary conditions
and refactor safety and did not want a second description to keep in sync.

    graph = (
        TaskGraph()
        .flow(locate >> approach >> grasp >> lift >> place >> home)
        .wire(locate.out.pose, grasp.in_.target)
    )

Nodes are referenced either by registry id — ``task("basic/grasp", width=0.02)``
— or as a live instance when the providing project is importable. The registry
form imports nothing, which keeps authored workflows light enough to lint without
Isaac or a policy stack.

Fan-out and fan-in are written with tuples::

    a >> (left, right) >> join
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from i4h_common.manifest import TaskSpec, ports_of
from i4h_engine.ports import PortAccessor, PortRef
from i4h_engine.task import Task


@dataclass(frozen=True, slots=True)
class Edge:
    """Control edge: ``src`` must succeed before ``dst`` may start."""

    src: str
    dst: str


@dataclass(frozen=True, slots=True)
class DataEdge:
    """Data edge: ``src`` node's output feeds ``dst`` node's input."""

    src: PortRef
    dst: PortRef


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    """What to do when a node reports FAILURE: give up, or try again."""

    #: ``abort`` | ``retry``
    action: str = "abort"
    #: Extra attempts when retrying.
    times: int = 0


class Node:
    """One vertex. Wraps either a registry id or a live :class:`Task` instance."""

    __slots__ = ("id", "ref", "params", "spec", "_inputs", "_outputs", "_failure", "in_", "out")

    def __init__(
        self,
        ref: str | Task,
        *,
        id: str | None = None,  # noqa: A002 - reads naturally at the call site
        spec: TaskSpec | None = None,
        **params: Any,
    ) -> None:
        self.ref = ref
        self.params: dict[str, Any] = dict(params)
        self.spec = spec
        self._failure = FailurePolicy()

        if isinstance(ref, Task):
            self._inputs = ports_of(type(ref).Inputs)
            self._outputs = ports_of(type(ref).Outputs)
            default_id = ref.name
        else:
            if spec is None:
                from i4h_engine.registry import default_registry  # noqa: PLC0415

                spec = default_registry().task(ref)
                self.spec = spec
            self._inputs = dict(spec.inputs)
            self._outputs = dict(spec.outputs)
            default_id = spec.name

        self.id = id or default_id
        self.in_ = PortAccessor(self, "in")
        self.out = PortAccessor(self, "out")

    @property
    def task_id(self) -> str:
        """Registry id, or ``local/<name>`` for a directly-instantiated task."""
        if self.spec is not None:
            return self.spec.id
        return f"local/{self.ref.name}" if isinstance(self.ref, Task) else str(self.ref)

    @property
    def input_ports(self) -> dict[str, str]:
        return self._inputs

    @property
    def output_ports(self) -> dict[str, str]:
        return self._outputs

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return tuple(name for name, decl in self._inputs.items() if not decl.endswith("?"))

    @property
    def failure_policy(self) -> FailurePolicy:
        return self._failure

    def __rshift__(self, other: Any) -> Chain:
        return Chain.of(self) >> other

    def __repr__(self) -> str:
        return f"Node({self.id!r}, task={self.task_id!r})"


def node(ref: str | Task, *, id: str | None = None, **params: Any) -> Node:  # noqa: A002
    """Wrap a registry id or task instance as a graph node. Idempotent on ``Node``."""
    if isinstance(ref, Node):
        return ref
    return Node(ref, id=id, **params)


#: Reads better in workflow modules: ``task("gr00t_n15/scissor_pick_and_place")``.
task = node


def _as_nodes(value: Any) -> list[Node]:
    if isinstance(value, Node | Task | str):
        return [node(value)]
    if isinstance(value, Chain):
        return list(value.tails)
    if isinstance(value, Sequence):
        out: list[Node] = []
        for item in value:
            out.extend(_as_nodes(item))
        return out
    raise TypeError(f"cannot use {value!r} as a workflow node")


class Chain:
    """An in-progress ``>>`` expression: the nodes and edges written so far."""

    __slots__ = ("nodes", "edges", "heads", "tails")

    def __init__(self, nodes: list[Node], edges: list[Edge], heads: list[Node], tails: list[Node]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.heads = heads
        self.tails = tails

    @classmethod
    def of(cls, value: Any) -> Chain:
        if isinstance(value, Chain):
            return value
        nodes = _as_nodes(value)
        return cls(nodes=list(nodes), edges=[], heads=list(nodes), tails=list(nodes))

    def __rshift__(self, other: Any) -> Chain:
        right = Chain.of(other)
        # Fan-out from every current tail into every new head: a >> (b, c) makes
        # two edges, (b, c) >> d makes two more, so joins are implicit.
        new_edges = [Edge(src=tail.id, dst=head.id) for tail in self.tails for head in right.heads]
        merged: list[Node] = list(self.nodes)
        seen = {n.id for n in merged}
        for candidate in right.nodes:
            if candidate.id not in seen:
                merged.append(candidate)
                seen.add(candidate.id)
        return Chain(
            nodes=merged,
            edges=self.edges + right.edges + new_edges,
            heads=list(self.heads),
            tails=list(right.tails),
        )

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)


class TaskGraph:
    """A directed acyclic graph of task invocations."""

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        timeout_success: Callable[[Any], Any] | None = None,
        description: str = "",
    ) -> None:
        self.max_steps = max_steps
        #: Optional compatibility fallback evaluated only when the step budget
        #: is exhausted. Some legacy evaluations accept a weaker terminal
        #: condition than their early-success predicate.
        self.timeout_success = timeout_success
        self.description = description
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._data_edges: list[DataEdge] = []

    # -- construction ----------------------------------------------------
    def add(self, *items: Any) -> TaskGraph:
        """Register nodes without ordering them (useful for isolated roots)."""
        for item in items:
            for candidate in _as_nodes(item):
                self._register(candidate)
        return self

    def flow(self, *chains: Any) -> TaskGraph:
        """Add one or more ``>>`` expressions as control edges."""
        for item in chains:
            chain = Chain.of(item)
            for candidate in chain.nodes:
                self._register(candidate)
            for edge in chain.edges:
                if edge not in self._edges:
                    self._edges.append(edge)
        return self

    def wire(self, src: PortRef, dst: PortRef) -> TaskGraph:
        """Route ``src`` (an output port) into ``dst`` (an input port)."""
        if src.direction != "out":
            raise ValueError(f"wire source must be an output port, got {src}")
        if dst.direction != "in":
            raise ValueError(f"wire target must be an input port, got {dst}")
        for ref in (src, dst):
            if ref.node_id not in self._nodes:
                raise KeyError(f"{ref.node_id!r} is not in the task graph; add it with .flow() first")
        existing = next((e for e in self._data_edges if e.dst == dst), None)
        if existing is not None:
            raise ValueError(f"{dst} is already wired from {existing.src}")
        self._data_edges.append(DataEdge(src=src, dst=dst))
        return self

    def on_failure(self, target: Node | str, action: str = "abort", *, times: int = 0) -> TaskGraph:
        """Set a node's failure behaviour: ``abort`` or ``retry``."""
        node_id = target.id if isinstance(target, Node) else target
        if node_id not in self._nodes:
            raise KeyError(f"{node_id!r} is not in the task graph")
        if action not in ("abort", "retry"):
            raise ValueError(f"unknown failure action {action!r}; expected abort or retry")
        self._nodes[node_id]._failure = FailurePolicy(action=action, times=times)
        return self

    def _register(self, candidate: Node) -> None:
        existing = self._nodes.get(candidate.id)
        if existing is None:
            self._nodes[candidate.id] = candidate
            return
        if existing is not candidate:
            raise ValueError(
                f"task graph already has a different node with id {candidate.id!r}; "
                f"pass id= to disambiguate, e.g. task({candidate.task_id!r}, id='{candidate.id}_2')"
            )

    # -- inspection ------------------------------------------------------
    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges)

    @property
    def data_edges(self) -> tuple[DataEdge, ...]:
        return tuple(self._data_edges)

    def node_by_id(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"task graph has no node {node_id!r}") from exc

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        return tuple(e.src for e in self._edges if e.dst == node_id)

    def successors(self, node_id: str) -> tuple[str, ...]:
        return tuple(e.dst for e in self._edges if e.src == node_id)

    def roots(self) -> tuple[Node, ...]:
        """Nodes with no control predecessors — where a rollout starts."""
        return tuple(n for n in self._nodes.values() if not self.predecessors(n.id))

    def terminals(self) -> tuple[Node, ...]:
        """Nodes with no control successors — the workflow is done when these are."""
        return tuple(n for n in self._nodes.values() if not self.successors(n.id))

    def inputs_for(self, node_id: str) -> tuple[DataEdge, ...]:
        return tuple(e for e in self._data_edges if e.dst.node_id == node_id)

    def topological_order(self) -> tuple[str, ...]:
        """Kahn's algorithm. Raises on a cycle — lint reports it more gently."""
        indegree = {n.id: len(self.predecessors(n.id)) for n in self._nodes.values()}
        queue = [n for n, d in indegree.items() if d == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for successor in self.successors(current):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if len(order) != len(self._nodes):
            remaining = sorted(set(self._nodes) - set(order))
            raise ValueError(f"task graph has a cycle involving {remaining}")
        return tuple(order)

    def find_cycle(self) -> tuple[str, ...]:
        """A cycle as a node-id sequence, or ``()`` if the graph is acyclic."""
        colour: dict[str, int] = dict.fromkeys(self._nodes, 0)
        stack: list[str] = []

        def visit(current: str) -> tuple[str, ...]:
            colour[current] = 1
            stack.append(current)
            for successor in self.successors(current):
                if colour[successor] == 1:
                    return tuple(stack[stack.index(successor) :] + [successor])
                if colour[successor] == 0:
                    found = visit(successor)
                    if found:
                        return found
            colour[current] = 2
            stack.pop()
            return ()

        for node_id in self._nodes:
            if colour[node_id] == 0:
                found = visit(node_id)
                if found:
                    return found
        return ()

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        return f"TaskGraph(nodes={len(self._nodes)}, edges={len(self._edges)}, " f"data_edges={len(self._data_edges)})"
