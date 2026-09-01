# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from conftest import Consumer, Counter, Exploding, Failing, Producer, Writer

from i4h_common.types import Pose
from i4h_common.world import ActuationConflict
from i4h_engine.events import EventKind
from i4h_engine.executor import Engine, autowire
from i4h_engine.graph import TaskGraph, node
from i4h_engine.status import Status, WorkflowStatus
from i4h_engine.task import Task


class Waiting(Task):
    def tick(self, ctx):
        ctx.act.hold()
        return Status.WAITING


class FinalAction(Task):
    advance_on_success = True

    def tick(self, ctx):
        ctx.act.hold()
        return Status.SUCCESS


def run(engine: Engine, ctx, *, limit: int = 500) -> WorkflowStatus:
    engine.start(ctx)
    for _ in range(limit):
        if engine.status.is_terminal:
            break
        engine.tick(ctx)
    return engine.status


def test_linear_workflow_succeeds_in_order(ctx):
    a, b, c = Counter(2, name="a"), Counter(3, name="b"), Counter(1, name="c")
    workflow = TaskGraph().flow(node(a) >> node(b) >> node(c))
    engine = Engine(workflow)
    assert run(engine, ctx) is WorkflowStatus.SUCCEEDED
    assert (a.seen, b.seen, c.seen) == (2, 3, 1)
    # 2 + 3 + 1 ticks, each node entered exactly once.
    assert engine.step == 6
    assert (a.entered, b.entered, c.entered) == (1, 1, 1)


def test_node_ticks_exactly_once_per_step(ctx):
    a = Counter(5, name="a")
    engine = Engine(TaskGraph().flow(node(a)))
    engine.start(ctx)
    engine.tick(ctx)
    assert a.seen == 1
    engine.tick(ctx)
    assert a.seen == 2


def test_waiting_task_does_not_consume_a_simulation_step(ctx):
    engine = Engine(TaskGraph(max_steps=1).flow(node(Waiting(name="waiting"))))
    engine.start(ctx)
    for _ in range(10):
        engine.tick(ctx)
    assert engine.step == 0
    assert engine.states["waiting"].ticks == 0
    assert engine.advance_requested is False
    assert engine.status is WorkflowStatus.RUNNING


def test_terminal_action_requests_one_final_simulator_advance(ctx):
    engine = Engine(TaskGraph(max_steps=1).flow(node(FinalAction(name="final"))))
    engine.start(ctx)

    assert engine.tick(ctx) is WorkflowStatus.SUCCEEDED
    assert engine.step == 1
    assert engine.terminal_advance_requested is True


def test_successor_does_not_start_early(ctx):
    a, b = Counter(3, name="a"), Counter(1, name="b")
    engine = Engine(TaskGraph().flow(node(a) >> node(b)))
    engine.start(ctx)
    engine.tick(ctx)
    engine.tick(ctx)
    assert b.entered == 0
    assert engine.active_nodes == ("a",)


def test_parallel_branches_run_concurrently(ctx):
    left = Writer("joint_targets", name="left")
    right = Writer("gripper", name="right")
    root, join = Counter(1, name="root"), Counter(1, name="join")
    workflow = TaskGraph().flow(node(root) >> (node(left), node(right)) >> node(join))
    engine = Engine(workflow)
    engine.start(ctx)
    engine.tick(ctx)  # root
    engine.tick(ctx)  # both branches active
    assert set(engine.active_nodes) == {"left", "right"}


def test_conflicting_writes_raise(ctx):
    # Two branches driving the same actuator is a workflow bug, not a race to win.
    left = Writer("joint_targets", name="left")
    right = Writer("joint_targets", name="right")
    workflow = TaskGraph().flow(node(Counter(1, name="root")) >> (node(left), node(right)))
    engine = Engine(workflow)
    engine.start(ctx)
    engine.tick(ctx)
    with pytest.raises(ActuationConflict, match="both wrote joint_targets"):
        engine.tick(ctx)


def test_different_actuators_do_not_conflict(ctx):
    left = Writer("joint_targets", name="left")
    right = Writer("gripper", name="right")
    workflow = TaskGraph().flow(node(Counter(1, name="root")) >> (node(left), node(right)))
    engine = Engine(workflow)
    engine.start(ctx)
    engine.tick(ctx)
    engine.tick(ctx)  # no raise


def test_same_node_may_rewrite_across_ticks(ctx):
    writer = Writer("joint_targets", name="solo")
    engine = Engine(TaskGraph().flow(node(writer)))
    engine.start(ctx)
    for _ in range(5):
        engine.tick(ctx)  # same owner each tick, so no conflict


def test_data_edge_routes_outputs(ctx):
    producer = Producer(Pose.from_xyz(3.0, 4.0, 0.0), name="p")
    consumer = Consumer(name="c")
    p_node, c_node = node(producer), node(consumer)
    workflow = TaskGraph().flow(p_node >> c_node).wire(p_node.out.pose, c_node.in_.target)
    engine = Engine(workflow)
    assert run(engine, ctx) is WorkflowStatus.SUCCEEDED
    assert np.allclose(consumer.received.target.pos, [[3.0, 4.0, 0.0]])
    assert engine.states["c"].outputs["distance"] == pytest.approx(5.0)


def test_optional_input_uses_its_default(ctx):
    producer, consumer = node(Producer(name="p")), node(Consumer(name="c"))
    workflow = TaskGraph().flow(producer >> consumer).wire(producer.out.pose, consumer.in_.target)
    engine = Engine(workflow)
    run(engine, ctx)
    assert workflow.node_by_id("c").ref.received.scale == 1.0


def test_constructor_param_fills_an_input(ctx):
    producer = node(Producer(name="p"))
    consumer = node(Consumer(name="c"), scale=2.5)
    workflow = TaskGraph().flow(producer >> consumer).wire(producer.out.pose, consumer.in_.target)
    run(Engine(workflow), ctx)
    assert consumer.ref.received.scale == 2.5


def test_autowire_single_unambiguous_input():
    producer, consumer = node(Producer(name="p")), node(Consumer(name="c"))
    workflow = TaskGraph().flow(producer >> consumer)
    added = autowire(workflow)
    assert len(added) == 1
    assert str(added[0].src) == "p.out.pose"
    assert str(added[0].dst) == "c.in.target"


def test_autowire_skips_when_already_wired():
    producer, consumer = node(Producer(name="p")), node(Consumer(name="c"))
    workflow = TaskGraph().flow(producer >> consumer).wire(producer.out.pose, consumer.in_.target)
    assert autowire(workflow) == ()


def test_autowire_skips_ambiguous_predecessors():
    # Two predecessors: which pose did you mean? Refuse rather than guess.
    p1, p2 = node(Producer(name="p1")), node(Producer(name="p2"))
    consumer = node(Consumer(name="c"))
    workflow = TaskGraph().flow(p1 >> consumer, p2 >> consumer)
    assert autowire(workflow) == ()


def test_failure_aborts_workflow_by_default(ctx):
    a, b = Failing(1, name="a"), Counter(1, name="b")
    engine = Engine(TaskGraph().flow(node(a) >> node(b)))
    assert run(engine, ctx) is WorkflowStatus.FAILED
    assert b.entered == 0
    assert "a" in engine.detail


def test_retry_policy_reenters_node(ctx):
    flaky = Failing(1, succeed_on_attempt=3, name="flaky")
    flaky_node = node(flaky)
    workflow = TaskGraph().flow(flaky_node >> node(Counter(1, name="after")))
    workflow.on_failure(flaky_node, "retry", times=2)
    engine = Engine(workflow)
    assert run(engine, ctx) is WorkflowStatus.SUCCEEDED
    assert flaky.attempts == 3


def test_retry_exhausted_fails(ctx):
    flaky = node(Failing(1, name="flaky"))
    workflow = TaskGraph().flow(flaky)
    workflow.on_failure(flaky, "retry", times=1)
    assert run(Engine(workflow), ctx) is WorkflowStatus.FAILED


def test_task_exception_becomes_failure_not_crash(ctx):
    engine = Engine(TaskGraph().flow(node(Exploding(name="boom"))))
    assert run(engine, ctx) is WorkflowStatus.FAILED
    assert "RuntimeError: boom" in engine.states["boom"].detail


def test_timeout_fails_node(ctx):
    class Slow(Counter):
        timeout_s = 0.05  # 3 ticks at 60 Hz

    engine = Engine(TaskGraph().flow(node(Slow(1000, name="slow"))))
    assert run(engine, ctx) is WorkflowStatus.FAILED
    assert "timed out" in engine.states["slow"].detail


def test_max_steps_fails_workflow(ctx):
    engine = Engine(TaskGraph().flow(node(Counter(1000, name="a"))), max_steps=10)
    assert run(engine, ctx) is WorkflowStatus.FAILED
    assert "max_steps" in engine.detail


def test_timeout_success_can_accept_a_legacy_terminal_fallback(ctx):
    workflow = TaskGraph(timeout_success=lambda _ctx: True).flow(node(Counter(1000, name="a")))
    engine = Engine(workflow, max_steps=3)
    engine.start(ctx)
    for _ in range(4):
        engine.tick(ctx)
    assert engine.status is WorkflowStatus.SUCCEEDED
    assert "timeout_success" in engine.detail


def test_abort_calls_on_abort(ctx):
    a = Counter(1000, name="a")
    engine = Engine(TaskGraph().flow(node(a)))
    engine.start(ctx)
    engine.tick(ctx)
    engine.abort(ctx)
    assert engine.status is WorkflowStatus.ABORTED
    assert a.aborted == 1


def test_events_sequence(ctx):
    events: list[str] = []
    workflow = TaskGraph().flow(node(Counter(1, name="a")) >> node(Counter(1, name="b")))
    engine = Engine(workflow, on_event=lambda e: events.append(f"{e.kind}:{e.node}"))
    run(engine, ctx)
    assert events[0] == f"{EventKind.WORKFLOW_STARTED}:"
    assert f"{EventKind.NODE_ENTERED}:a" in events
    assert f"{EventKind.NODE_SUCCEEDED}:a" in events
    assert events[-1] == f"{EventKind.WORKFLOW_FINISHED}:"


def test_event_outputs_are_wire_safe(ctx):
    captured: list[dict] = []
    producer = Producer(name="p")
    engine = Engine(
        TaskGraph().flow(node(producer)),
        on_event=lambda e: captured.append(e.outputs) if e.kind == EventKind.NODE_SUCCEEDED else None,
    )
    run(engine, ctx)
    # A Pose must be summarized, not embedded, since events go over msgpack.
    assert captured[0]["pose"] == "<Pose>"


def test_segments_cover_each_node(ctx):
    workflow = TaskGraph().flow(node(Counter(2, name="a")) >> node(Counter(3, name="b")))
    engine = Engine(workflow)
    run(engine, ctx)
    segments = {name: (start, end) for name, _task, start, end in engine.segments}
    assert segments["a"] == (0, 2)
    assert segments["b"] == (2, 5)


def test_reset_clears_state(ctx):
    a = Counter(1, name="a")
    engine = Engine(TaskGraph().flow(node(a)))
    run(engine, ctx)
    engine.reset()
    assert engine.status is WorkflowStatus.PENDING
    assert engine.step == 0
    assert engine.states["a"].status == "pending"


def test_second_episode_reenters_tasks(ctx):
    a = Counter(2, name="a")
    engine = Engine(TaskGraph().flow(node(a)))
    run(engine, ctx)
    run(engine, ctx)
    assert a.entered == 2
