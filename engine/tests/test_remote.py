# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end exercise of the two-halves task protocol.

The whole point of ``RemoteTask`` is that it never imports the backend, so a
fake backend on an in-process bus is a *complete* test of the contract — not a
stand-in for one. No zenoh, no torch, no policy stack.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from i4h_common.bus.inproc import InProcBus
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import ActionChunk, ObsFrame, TaskSpecMsg, TaskStatusMsg, decode, encode
from i4h_common.manifest import BackendSpec, TaskSpec
from i4h_engine.executor import Engine
from i4h_engine.graph import TaskGraph, node
from i4h_engine.remote import DEFAULT_READY_TIMEOUT_S, READY_TIMEOUT_ENV, RemoteTask, RemoteTaskError
from i4h_engine.status import Status, WorkflowStatus

SPEC = TaskSpec(
    project="gr00t_n15",
    name="scissor_pick_and_place",
    runtime="remote",
    summary="Grip the scissors and put them into the tray",
    prompt="Pick up the scissors and place them in the tray",
    backend=BackendSpec(project="tasks/gr00t_n15", entry="i4h_tasks.gr00t_n15.server:main"),
    outputs={"success": "bool"},
    requires={"embodiment": "so101", "dof": 6, "cameras": []},
    observation={"state_names": ["a", "b", "c", "d", "e", "f"]},
    model={"action_horizon": 2},
)


class FakeBackend:
    """Minimal stand-in for a policy server: ready on spec, actions on obs."""

    def __init__(self, bus: InProcBus, keys: Keys, uid: str, *, dof: int = 6, succeed_after: int | None = None) -> None:
        self.bus = bus
        self.keys = keys
        self.uid = uid
        self.dof = dof
        self.succeed_after = succeed_after
        self.obs_seen: list[ObsFrame] = []
        self.spec_seen: list[TaskSpecMsg] = []
        self.silent = False
        bus.subscribe(keys.task_spec(uid), self._on_spec)
        bus.subscribe(keys.task_obs(uid), self._on_obs)

    def _on_spec(self, _key: str, payload: bytes) -> None:
        self.spec_seen.append(decode(payload, TaskSpecMsg))
        self.bus.publish(self.keys.task_status(self.uid), encode(TaskStatusMsg(task_uid=self.uid, status="ready")))

    def _on_obs(self, _key: str, payload: bytes) -> None:
        frame = decode(payload, ObsFrame)
        self.obs_seen.append(frame)
        if self.silent:
            return
        if self.succeed_after is not None and len(self.obs_seen) >= self.succeed_after:
            self.bus.publish(
                self.keys.task_status(self.uid), encode(TaskStatusMsg(task_uid=self.uid, status="success"))
            )
            return
        horizon = 2
        actions = [float(len(self.obs_seen))] * self.dof * horizon
        self.bus.publish(
            self.keys.task_action(self.uid),
            encode(ActionChunk(task_uid=self.uid, horizon=horizon, dof=self.dof, actions=actions)),
        )


@pytest.fixture
def bus() -> InProcBus:
    return InProcBus()


def _wire(ctx, bus, **task_kwargs):
    keys = Keys("test-run")
    ctx.bus = bus
    ctx.run_id = "test-run"
    task = RemoteTask(SPEC, keys=keys, **task_kwargs)
    backend = FakeBackend(bus, keys, f"{task.name}-0", **{})
    return task, backend, keys


def test_backend_handshake_and_spec_contents(ctx, bus):
    keys = Keys("test-run")
    ctx.bus = bus
    ctx.run_id = "test-run"
    task = RemoteTask(SPEC, keys=keys, prompt="Grip the scissors")
    backend = FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    assert len(backend.spec_seen) == 1
    sent = backend.spec_seen[0]
    assert sent.task_id == "gr00t_n15/scissor_pick_and_place"
    assert sent.prompt == "Grip the scissors"
    # model/observation are not forwarded: the backend reads its own
    # catalog, so arena never carries config it does not look at.
    assert sent.model == {}
    assert sent.observation == {}


def test_retry_uses_a_distinct_task_uid(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id, ctx.attempt_index = bus, "test-run", 2
    task = RemoteTask(SPEC, keys=keys)
    backend = FakeBackend(bus, keys, f"{task.name}-0-2")
    task.on_enter(ctx, None)
    assert backend.spec_seen[0].task_uid == f"{task.name}-0-2"


def test_prompt_defaults_to_task_manifest(ctx, bus):
    task = RemoteTask(SPEC)
    assert task.prompt == SPEC.prompt


def test_actions_are_applied_to_actuation(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys)
    FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    assert task.tick(ctx) is Status.RUNNING
    assert np.allclose(ctx.act.joint_targets["robot"], 1.0)


def test_action_chunk_is_consumed_across_ticks(ctx, bus):
    # A horizon-2 chunk must feed two ticks; the backend only publishes on obs.
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys)
    backend = FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    task.tick(ctx)
    backend.silent = True  # no further chunks
    task.tick(ctx)
    assert np.allclose(ctx.act.joint_targets["robot"], 1.0)  # second half of chunk 1


def test_backend_success_ends_the_node(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys)
    FakeBackend(bus, keys, f"{task.name}-0", succeed_after=3)
    task.on_enter(ctx, None)
    statuses = [task.tick(ctx) for _ in range(6)]
    assert Status.SUCCESS in statuses
    assert task.on_exit(ctx).success is True


def test_local_until_predicate_ends_the_node(ctx, bus):
    # A policy usually cannot judge its own success; the scene can.
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    fired = {"n": 0}

    def until(_ctx):
        fired["n"] += 1
        return fired["n"] >= 3

    task = RemoteTask(SPEC, keys=keys, until=until)
    FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    assert [task.tick(ctx) for _ in range(3)][-1] is Status.SUCCESS


def test_until_accepts_per_env_mask(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys, until=lambda _c: np.array([False, True]))
    FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    assert task.tick(ctx) is Status.SUCCESS


def test_backend_failure_fails_the_node(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys)
    FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    bus.publish(
        keys.task_status(f"{task.name}-0"),
        encode(TaskStatusMsg(task_uid=f"{task.name}-0", status="failure", detail="model diverged")),
    )
    assert task.tick(ctx) is Status.FAILURE


def test_missing_backend_raises_actionable_error(ctx, bus):
    """Backend startup latency waits without consuming simulation steps."""
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=Keys("test-run"), ready_timeout_s=0.05)
    task.on_enter(ctx, None)
    assert task.tick(ctx) is Status.WAITING
    time.sleep(0.06)
    with pytest.raises(RemoteTaskError, match="did not become ready"):
        task.tick(ctx)


def test_ready_timeout_reads_the_environment(monkeypatch):
    """A first run downloads the checkpoint inside the readiness window."""
    monkeypatch.delenv(READY_TIMEOUT_ENV, raising=False)
    assert RemoteTask(SPEC).ready_timeout_s == DEFAULT_READY_TIMEOUT_S
    monkeypatch.setenv(READY_TIMEOUT_ENV, "3600")
    assert RemoteTask(SPEC).ready_timeout_s == 3600.0
    # An explicit argument still wins: the environment only moves the default.
    assert RemoteTask(SPEC, ready_timeout_s=0.05).ready_timeout_s == 0.05


@pytest.mark.parametrize("value", ["soon", "0", "-1"])
def test_ready_timeout_rejects_a_useless_value(monkeypatch, value):
    monkeypatch.setenv(READY_TIMEOUT_ENV, value)
    with pytest.raises(RuntimeError, match=READY_TIMEOUT_ENV):
        RemoteTask(SPEC)


def test_no_bus_raises_actionable_error(ctx):
    ctx.bus = None
    task = RemoteTask(SPEC)
    with pytest.raises(RemoteTaskError, match="no bus is available"):
        task.on_enter(ctx, None)


def test_action_starvation_times_out(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys, action_timeout_s=0.05)
    backend = FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    backend.silent = True
    assert task.tick(ctx) is Status.WAITING
    time.sleep(0.06)
    assert task.tick(ctx) is Status.FAILURE
    assert ctx.act.holds  # held the robot while starved rather than dropping it


def test_remote_task_runs_inside_a_workflow(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys)
    FakeBackend(bus, keys, f"{task.name}-0", succeed_after=2)
    engine = Engine(TaskGraph().flow(node(task)))
    engine.start(ctx)
    for _ in range(10):
        if engine.status.is_terminal:
            break
        engine.tick(ctx)
    assert engine.status is WorkflowStatus.SUCCEEDED
    assert engine.states[task.name].outputs["success"] is True


def test_observation_carries_state_names_from_the_scene(ctx, bus):
    keys = Keys("test-run")
    ctx.bus, ctx.run_id = bus, "test-run"
    task = RemoteTask(SPEC, keys=keys)
    backend = FakeBackend(bus, keys, f"{task.name}-0")
    task.on_enter(ctx, None)
    task.tick(ctx)
    # Joint names come from the live scene, not from a manifest copy of them.
    assert backend.obs_seen[0].state_names == list(ctx.scene.joints().names)
    assert len(backend.obs_seen[0].state) == 6
