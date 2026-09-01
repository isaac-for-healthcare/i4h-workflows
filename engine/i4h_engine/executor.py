# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workflow execution engine.

One engine tick evaluates the active workflow nodes once. The simulation runner
owns the loop and advances physics only when the engine requests it::

    engine = Engine(graph, registry, workflow_name="scissor_pick_and_place")
    engine.start(ctx)
    while engine.status is WorkflowStatus.RUNNING and app.is_running():
        engine.tick(ctx)                     # active nodes write into ctx.act
        if engine.advance_requested:
            env.step(ctx.act.tensor())

An all-``WAITING`` tick requests no simulation advance. Idle mode renders
without stepping physics.

Per tick, in order:

1. Resolve the ready set — nodes whose control predecessors have all succeeded.
2. ``on_enter`` newly-ready nodes with their resolved inputs.
3. ``tick()`` every active node exactly once.
4. On SUCCESS: collect ``on_exit()`` outputs, route them along data edges.
5. On FAILURE: apply the node's failure policy.
6. Finish when every terminal node is done, or on abort.

**Actuation conflict is a hard error.** Two concurrently active nodes writing
the same actuator in one tick raises rather than silently letting the last
writer win — that is what makes parallel branches (both PSM arms reaching at
once) safe to express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from i4h_common.types import satisfied
from i4h_common.world import Actuation, ActuationConflict
from i4h_engine.events import EventKind, EventSink, WorkflowEvent
from i4h_engine.graph import DataEdge, Node, TaskGraph
from i4h_engine.ports import coerce_value, types_compatible
from i4h_engine.status import Status, WorkflowStatus
from i4h_engine.task import Task, TickContext


def autowire(graph: TaskGraph) -> tuple[DataEdge, ...]:
    """Infer the obvious data edges and add them to ``graph``.

    Rule, kept deliberately narrow so nothing surprising is inferred: when a node
    has exactly one unwired required input, exactly one control predecessor, and
    that predecessor has exactly one output of a compatible type, connect them.
    Everything else must be written explicitly.

    Returns the edges that were added.
    """
    added: list[DataEdge] = []
    for node_obj in graph.nodes:
        wired = {edge.dst.port for edge in graph.inputs_for(node_obj.id)}
        unwired = [name for name in node_obj.required_inputs if name not in wired]
        if len(unwired) != 1:
            continue
        preds = graph.predecessors(node_obj.id)
        if len(preds) != 1:
            continue
        source = graph.node_by_id(preds[0])
        if len(source.output_ports) != 1:
            continue
        ((out_name, out_type),) = source.output_ports.items()
        in_name = unwired[0]
        if not types_compatible(out_type, node_obj.input_ports[in_name]):
            continue
        edge = DataEdge(src=source.out[out_name], dst=node_obj.in_[in_name])
        graph._data_edges.append(edge)
        added.append(edge)
    return tuple(added)


class ActuationTracker:
    """Wraps an :class:`~i4h_common.world.Actuation` and attributes writes to nodes.

    Every write is recorded against whichever node is currently ticking. A
    second write to the same ``(robot, channel)`` within one tick, from a
    different node, raises :class:`~i4h_common.world.ActuationConflict`.
    """

    def __init__(self, actuation: Actuation) -> None:
        self._actuation = actuation
        self._owner: str = ""
        self._writes: dict[tuple[str, str], str] = {}
        #: ``(node, robot, channel)`` for the tick just completed — the recorder uses it.
        self.last_tick_writes: tuple[tuple[str, str, str], ...] = ()

    # -- engine-facing ---------------------------------------------------
    def begin_tick(self) -> None:
        self.last_tick_writes = tuple((n, r, c) for (r, c), n in self._writes.items())
        self._writes.clear()

    def set_owner(self, node_id: str) -> None:
        self._owner = node_id

    def replace_actuation(self, actuation: Actuation) -> None:
        """Retarget the tracker after the runner resets and rebuilds adapters."""
        self._actuation = actuation
        self._writes.clear()
        self.last_tick_writes = ()

    def _claim(self, robot: str, channel: str) -> None:
        key = (robot, channel)
        holder = self._writes.get(key)
        if holder is not None and holder != self._owner:
            raise ActuationConflict(
                f"nodes {holder!r} and {self._owner!r} both wrote {channel} of robot {robot!r} "
                f"in one tick; parallel branches must drive different actuators"
            )
        self._writes[key] = self._owner

    # -- Actuation protocol ----------------------------------------------
    @property
    def dof(self) -> int:
        return self._actuation.dof

    @property
    def action_space(self) -> str:
        return self._actuation.action_space

    def set_joint_targets(self, values: Any, robot: str = "robot") -> None:
        self._claim(robot, "joint_targets")
        self._actuation.set_joint_targets(values, robot)

    def set_gripper(self, width: Any, robot: str = "robot") -> None:
        self._claim(robot, "gripper")
        self._actuation.set_gripper(width, robot)

    def set_ee_target(self, pose: Any, robot: str = "robot") -> None:
        self._claim(robot, "ee_target")
        self._actuation.set_ee_target(pose, robot)

    def set_ee_delta(self, values: Any, robot: str = "robot") -> None:
        self._claim(robot, "ee_target")
        self._actuation.set_ee_delta(values, robot)

    def hold(self, robot: str = "robot") -> None:
        self._claim(robot, "hold")
        self._actuation.hold(robot)

    def set_raw_action(self, values: Any, robot: str = "robot") -> None:
        self._claim(robot, "raw_action")
        self._actuation.set_raw_action(values, robot)

    def unwrap(self) -> Actuation:
        """The underlying actuation, so re-starting a run does not nest trackers."""
        return self._actuation

    def __getattr__(self, name: str) -> Any:
        # Anything the tracker does not model (e.g. adapter-specific helpers)
        # passes through untracked.
        return getattr(self._actuation, name)


@dataclass
class NodeState:
    """Per-node bookkeeping for one workflow run."""

    node: Node
    status: str = "pending"  # pending | active | succeeded | failed | skipped
    task: Task | None = None
    entered_at: int = 0
    ticks: int = 0
    attempts: int = 0
    outputs: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def is_done(self) -> bool:
        return self.status in ("succeeded", "failed", "skipped")


class Engine:
    """Advances a :class:`~i4h_engine.graph.TaskGraph`, one node-tick per sim step."""

    def __init__(
        self,
        graph: TaskGraph,
        registry: Any | None = None,
        *,
        workflow_name: str = "",
        on_event: EventSink | None = None,
        max_steps: int | None = None,
        autowire_inputs: bool = True,
    ) -> None:
        self.graph = graph
        self.workflow_name = workflow_name
        self.registry = registry
        self.on_event = on_event
        self.max_steps = max_steps if max_steps is not None else graph.max_steps
        if autowire_inputs:
            autowire(graph)
        self.states: dict[str, NodeState] = {n.id: NodeState(node=n) for n in graph.nodes}
        self.status = WorkflowStatus.PENDING
        self.step = 0
        self.detail = ""
        self._tracker: ActuationTracker | None = None
        self._segments: list[tuple[str, str, int, int]] = []
        self.advance_requested = True
        self.terminal_advance_requested = False

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.states = {n.id: NodeState(node=n) for n in self.graph.nodes}
        self.status = WorkflowStatus.PENDING
        self.step = 0
        self.detail = ""
        self._segments = []
        self.advance_requested = True
        self.terminal_advance_requested = False

    def start(self, ctx: TickContext) -> None:
        """Prepare for a rollout. Wraps ``ctx.act`` so writes can be attributed."""
        self.reset()
        # Re-running an episode reuses the same ctx, whose `act` is already a
        # tracker from the previous run. Unwrap first: nesting trackers would
        # leave the inner one holding last episode's writes and misattribute
        # this episode's first tick as a conflict.
        target = ctx.act
        while isinstance(target, ActuationTracker):
            target = target.unwrap()
        self._tracker = ActuationTracker(target)
        ctx.act = self._tracker  # type: ignore[assignment]
        self.status = WorkflowStatus.RUNNING
        self._emit(EventKind.WORKFLOW_STARTED, ctx)

    def replace_actuation(self, ctx: TickContext, actuation: Actuation) -> None:
        """Keep the active workflow while replacing a reset scene's adapter."""
        if self._tracker is None:
            raise RuntimeError("engine must be started before replacing actuation")
        self._tracker.replace_actuation(actuation)
        ctx.act = self._tracker  # type: ignore[assignment]

    @property
    def segments(self) -> tuple[tuple[str, str, int, int], ...]:
        """``(node, task_id, start_step, end_step)`` for every completed node.

        This is what the HDF5 recorder turns into per-skill frame ranges.
        """
        return tuple(self._segments)

    @property
    def active_nodes(self) -> tuple[str, ...]:
        return tuple(nid for nid, st in self.states.items() if st.status == "active")

    # -- the tick --------------------------------------------------------
    def tick(self, ctx: TickContext) -> WorkflowStatus:
        if self.status is not WorkflowStatus.RUNNING:
            return self.status

        self.terminal_advance_requested = False
        if self.max_steps is not None and self.step >= self.max_steps:
            timeout_success = self.graph.timeout_success
            if timeout_success is not None and satisfied(timeout_success(ctx)):
                self._finish(
                    WorkflowStatus.SUCCEEDED,
                    ctx,
                    detail=f"accepted timeout_success at max_steps={self.max_steps}",
                )
            else:
                self._finish(WorkflowStatus.FAILED, ctx, detail=f"workflow exceeded max_steps={self.max_steps}")
            return self.status

        if self._tracker is not None:
            self._tracker.begin_tick()

        self.advance_requested = True
        saw_waiting = False
        saw_nonwaiting = False
        ctx.step = self.step
        self._activate_ready(ctx)

        for node_id in list(self.active_nodes):
            state = self.states[node_id]
            ctx.node_step = state.ticks
            if self._tracker is not None:
                self._tracker.set_owner(node_id)
            outcome = self._tick_node(state, ctx)
            if outcome is Status.WAITING:
                saw_waiting = True
                continue
            saw_nonwaiting = True
            state.ticks += 1
            if outcome is Status.SUCCESS:
                if state.task is not None and state.task.advance_on_success:
                    self.terminal_advance_requested = True
                self._succeed(state, ctx)
            elif outcome is Status.FAILURE:
                self._fail(state, ctx, detail=state.detail or "task reported FAILURE")
                if self.status.is_terminal:
                    return self.status

        if self._tracker is not None:
            self._tracker.set_owner("")

        if saw_waiting and not saw_nonwaiting:
            self.advance_requested = False
            return self.status

        self.step += 1
        self._check_completion(ctx)
        return self.status

    def _tick_node(self, state: NodeState, ctx: TickContext) -> Status:
        assert state.task is not None
        timeout = getattr(type(state.task), "timeout_s", None)
        if timeout is not None and state.ticks * ctx.dt > timeout:
            state.detail = f"timed out after {timeout}s"
            return Status.FAILURE
        try:
            return Status.coerce(state.task.tick(ctx))
        except ActuationConflict:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad task must not lose the whole run silently
            state.detail = f"{type(exc).__name__}: {exc}"
            return Status.FAILURE

    # -- frontier --------------------------------------------------------
    def _activate_ready(self, ctx: TickContext) -> None:
        for node_id, state in self.states.items():
            if state.status != "pending":
                continue
            preds = self.graph.predecessors(node_id)
            if not all(self.states[p].status == "succeeded" for p in preds):
                continue
            self._enter(state, ctx)

    def _enter(self, state: NodeState, ctx: TickContext) -> None:
        state.status = "active"
        state.entered_at = self.step
        state.ticks = 0
        state.attempts += 1
        state.inputs = self._resolve_inputs(state.node)
        if state.task is None:
            state.task = self._build_task(state.node)
        inputs_obj = self._build_inputs_object(state)
        if self._tracker is not None:
            self._tracker.set_owner(state.node.id)
        state.task.on_enter(ctx, inputs_obj)
        self._emit(EventKind.NODE_ENTERED, ctx, node=state.node)

    def _build_task(self, node_obj: Node) -> Task:
        if isinstance(node_obj.ref, Task):
            return node_obj.ref
        if self.registry is None:
            from i4h_engine.registry import default_registry  # noqa: PLC0415

            self.registry = default_registry()
        return self.registry.instantiate(node_obj.task_id, node_obj.params)

    def _resolve_inputs(self, node_obj: Node) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for edge in self.graph.inputs_for(node_obj.id):
            source = self.states[edge.src.node_id]
            if edge.src.port not in source.outputs:
                raise KeyError(
                    f"{node_obj.id}.{edge.dst.port} expects {edge.src}, but {edge.src.node_id} "
                    f"produced {sorted(source.outputs) or '(nothing)'}"
                )
            declared = node_obj.input_ports.get(edge.dst.port, "")
            values[edge.dst.port] = coerce_value(source.outputs[edge.src.port], declared)
        # Constructor params fill any input the graph did not.
        for name, value in node_obj.params.items():
            values.setdefault(name, value)
        return values

    def _build_inputs_object(self, state: NodeState) -> Any:
        task = state.task
        inputs_cls = getattr(type(task), "Inputs", None)
        if inputs_cls is None:
            return state.inputs
        try:
            import dataclasses  # noqa: PLC0415

            if not dataclasses.is_dataclass(inputs_cls):
                return state.inputs
            known = {f.name for f in dataclasses.fields(inputs_cls)}
            return inputs_cls(**{k: v for k, v in state.inputs.items() if k in known})
        except TypeError as exc:
            missing = [n for n in state.node.required_inputs if n not in state.inputs]
            raise TypeError(
                f"{state.node.id}: cannot build {inputs_cls.__name__}; unwired required inputs: {missing}"
            ) from exc

    # -- transitions -----------------------------------------------------
    def _succeed(self, state: NodeState, ctx: TickContext) -> None:
        assert state.task is not None
        outputs_obj = state.task.on_exit(ctx)
        state.outputs = _as_mapping(outputs_obj)
        state.status = "succeeded"
        self._segments.append((state.node.id, state.node.task_id, state.entered_at, self.step + 1))
        self._emit(EventKind.NODE_SUCCEEDED, ctx, node=state.node, outputs=state.outputs)

    def _fail(self, state: NodeState, ctx: TickContext, *, detail: str) -> None:
        policy = state.node.failure_policy
        state.detail = detail

        if policy.action == "retry" and state.attempts <= policy.times:
            self._emit(
                EventKind.NODE_RETRYING,
                ctx,
                node=state.node,
                detail=f"{detail} (attempt {state.attempts}/{policy.times + 1})",
            )
            state.status = "pending"
            state.ticks = 0
            state.task = None  # rebuild so the task starts from a clean state
            return

        state.status = "failed"
        self._segments.append((state.node.id, state.node.task_id, state.entered_at, self.step + 1))
        self._emit(EventKind.NODE_FAILED, ctx, node=state.node, detail=detail)

        self._finish(WorkflowStatus.FAILED, ctx, detail=f"{state.node.id}: {detail}")

    def _check_completion(self, ctx: TickContext) -> None:
        terminals = self.graph.terminals()
        if not terminals:
            self._finish(WorkflowStatus.FAILED, ctx, detail="workflow has no terminal nodes")
            return
        if all(self.states[n.id].is_done for n in terminals):
            # A workflow succeeds only when every terminal actually ran and succeeded.
            # A skipped terminal means part of the goal was never attempted, which
            # is not success even if the failure that caused it was tolerated.
            unmet = [(n.id, self.states[n.id].status) for n in terminals if self.states[n.id].status != "succeeded"]
            if unmet:
                rendered = ", ".join(f"{node_id}={status}" for node_id, status in unmet)
                self._finish(WorkflowStatus.FAILED, ctx, detail=f"terminal nodes not satisfied: {rendered}")
            else:
                self._finish(WorkflowStatus.SUCCEEDED, ctx)

    def _finish(self, status: WorkflowStatus, ctx: TickContext, *, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        for state in self.states.values():
            if state.status == "active" and state.task is not None:
                state.task.on_abort(ctx)
                state.status = "skipped"
        self._emit(EventKind.WORKFLOW_FINISHED, ctx, detail=detail or status.value)

    # -- external control ------------------------------------------------
    def abort(self, ctx: TickContext, *, detail: str = "aborted externally") -> None:
        if not self.status.is_terminal:
            self._finish(WorkflowStatus.ABORTED, ctx, detail=detail)

    def _emit(
        self,
        kind: str,
        ctx: TickContext,
        *,
        node: Node | None = None,
        outputs: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        if self.on_event is None:
            return
        self.on_event(
            WorkflowEvent(
                kind=kind,
                workflow=self.workflow_name,
                step=self.step,
                node=node.id if node else "",
                task_id=node.task_id if node else "",
                episode_index=ctx.episode_index,
                run_id=ctx.run_id,
                outputs=_summarize(outputs or {}),
                detail=detail,
            )
        )


def _as_mapping(outputs: Any) -> dict[str, Any]:
    if outputs is None:
        return {}
    if isinstance(outputs, dict):
        return dict(outputs)
    import dataclasses  # noqa: PLC0415

    if dataclasses.is_dataclass(outputs) and not isinstance(outputs, type):
        return {f.name: getattr(outputs, f.name) for f in dataclasses.fields(outputs)}
    raise TypeError(f"on_exit must return a dataclass or dict, got {type(outputs).__name__}")


def _summarize(outputs: dict[str, Any]) -> dict[str, Any]:
    """Event payloads go over the wire; keep only cheaply-serializable scalars."""
    summary: dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, str | int | float | bool) or value is None:
            summary[key] = value
        else:
            summary[key] = f"<{type(value).__name__}>"
    return summary
