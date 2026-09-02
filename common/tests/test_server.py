# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend half of the protocol, tested against the in-process bus.

Together with engine's test_remote.py this covers both ends of the wire
without zenoh, torch, or any policy stack installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.bus.inproc import InProcBus
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import ActionChunk, ObsFrame, TaskSpecMsg, TaskStatusMsg, decode, encode
from i4h_common.server import PolicyServer, Session


class EchoServer(PolicyServer):
    """Emits a constant chunk sized from the session's declared horizon."""

    def __init__(self, *, dof: int = 6, fail_load: bool = False, fail_infer: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dof = dof
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.loaded: list[Session] = []
        self.unloaded: list[Session] = []
        self.frames: list[ObsFrame] = []
        self.done_after: int | None = None

    def load(self, session: Session) -> None:
        if self.fail_load:
            raise RuntimeError("checkpoint not found")
        self.loaded.append(session)

    def infer(self, session: Session, frame: ObsFrame):
        if self.fail_infer:
            raise ValueError("nan in logits")
        self.frames.append(frame)
        return np.full((session.action_horizon, self.dof), float(session.steps), dtype=np.float32)

    def is_done(self, session: Session, frame: ObsFrame) -> bool:
        return self.done_after is not None and session.steps >= self.done_after

    def unload(self, session: Session) -> None:
        self.unloaded.append(session)


@pytest.fixture
def wiring():
    bus = InProcBus()
    keys = Keys("test-run")
    server = EchoServer(namespace="test-run", bus=bus, keys=keys)
    server.start()
    statuses: list[TaskStatusMsg] = []
    actions: list[ActionChunk] = []
    bus.subscribe(f"{keys.root}/task/*/status", lambda _k, p: statuses.append(decode(p, TaskStatusMsg)))
    bus.subscribe(f"{keys.root}/task/*/action", lambda _k, p: actions.append(decode(p, ActionChunk)))
    yield bus, keys, server, statuses, actions
    server.close()


def _spec(bus, keys, uid="n0", **kwargs):
    payload = {"task_uid": uid, "task_id": "gr00t_n15/scissor_pick_and_place", "model": {"action_horizon": 2}}
    payload.update(kwargs)
    bus.publish(keys.task_spec(uid), encode(TaskSpecMsg(**payload)))


def test_spec_opens_a_session_and_reports_ready(wiring):
    bus, keys, server, statuses, _ = wiring
    _spec(bus, keys, prompt="Grip the scissors")
    assert [s.status for s in statuses] == ["ready"]
    assert server.loaded[0].prompt == "Grip the scissors"
    assert server.loaded[0].action_horizon == 2


def test_preload_forwards_checkpoint_override():
    server = EchoServer(namespace="preload-test", bus=InProcBus())
    server._declaration = lambda _task_id: {"model": {"action_horizon": 2}}  # type: ignore[method-assign]
    server.preload("gr00t_n16/my_workflow_reach_table", checkpoint="/tmp/checkpoint-200")
    assert server.loaded[0].checkpoint == "/tmp/checkpoint-200"
    server.close()


def test_preload_only_loads_without_starting_server():
    server = EchoServer(namespace="preload-only-test", bus=InProcBus())
    server._declaration = lambda _task_id: {}  # type: ignore[method-assign]

    server.preload_only(("gr00t_n16/reach_table",), checkpoint="/tmp/checkpoint-200")

    assert [session.task_id for session in server.loaded] == ["gr00t_n16/reach_table"]
    assert server.loaded[0].checkpoint == "/tmp/checkpoint-200"
    assert server._subscriptions == []


def test_observation_produces_an_action_chunk(wiring):
    bus, keys, _server, _, actions = wiring
    _spec(bus, keys)
    bus.publish(keys.task_obs("n0"), encode(ObsFrame(task_uid="n0", step=0, state=[0.0] * 6)))
    assert len(actions) == 1
    assert actions[0].horizon == 2
    assert actions[0].dof == 6
    assert actions[0].reshape() == [[1.0] * 6, [1.0] * 6]


def test_observations_before_a_spec_are_ignored(wiring):
    bus, keys, _server, _statuses, actions = wiring
    bus.publish(keys.task_obs("ghost"), encode(ObsFrame(task_uid="ghost")))
    assert actions == []


def test_load_failure_is_reported_not_raised(wiring):
    bus, keys, server, statuses, _ = wiring
    server.fail_load = True
    _spec(bus, keys)
    assert statuses[-1].status == "error"
    assert "checkpoint not found" in statuses[-1].detail
    assert server.sessions == {}


def test_inference_failure_is_reported_and_server_survives(wiring):
    bus, keys, server, statuses, actions = wiring
    _spec(bus, keys)
    server.fail_infer = True
    bus.publish(keys.task_obs("n0"), encode(ObsFrame(task_uid="n0")))
    assert statuses[-1].status == "failure"
    assert "nan in logits" in statuses[-1].detail
    # Still alive: a later good frame must still work.
    server.fail_infer = False
    bus.publish(keys.task_obs("n0"), encode(ObsFrame(task_uid="n0")))
    assert len(actions) == 1


def test_is_done_publishes_success(wiring):
    bus, keys, server, statuses, _ = wiring
    _spec(bus, keys)
    server.done_after = 2
    bus.publish(keys.task_obs("n0"), encode(ObsFrame(task_uid="n0")))
    assert statuses[-1].status == "ready"
    bus.publish(keys.task_obs("n0"), encode(ObsFrame(task_uid="n0")))
    assert statuses[-1].status == "success"


def test_task_id_filter_ignores_other_stacks(wiring):
    bus, keys, server, statuses, _ = wiring
    server.task_ids = ("gr00t_n16/locomanip_push_cart",)
    _spec(bus, keys)
    assert statuses == []


def test_re_speccing_the_same_uid_unloads_the_previous_session(wiring):
    # Episode 2 of the same node reuses the uid; the old session must be released.
    bus, keys, server, _, _ = wiring
    _spec(bus, keys)
    _spec(bus, keys)
    assert len(server.loaded) == 2
    assert len(server.unloaded) == 1


def test_malformed_payloads_do_not_kill_the_server(wiring):
    bus, keys, _server, statuses, _ = wiring
    bus.publish(keys.task_spec("n0"), b"not-msgpack")
    assert statuses == []
    _spec(bus, keys)
    assert statuses[-1].status == "ready"


def test_session_decodes_images_with_declared_shapes():
    session = Session(task_uid="n0", task_id="t", run_id="r", episode_index=0, prompt="", checkpoint="")
    pixels = np.zeros((4, 5, 3), dtype=np.uint8)
    frame = ObsFrame(images={"room": pixels.tobytes()}, image_shapes={"room": [4, 5, 3]})
    assert session.images(frame)["room"].shape == (4, 5, 3)


def test_session_skips_images_without_a_shape():
    session = Session(task_uid="n0", task_id="t", run_id="r", episode_index=0, prompt="", checkpoint="")
    frame = ObsFrame(images={"room": b"\x00\x00\x00"}, image_shapes={})
    assert session.images(frame) == {}


def test_execution_steps_defaults_to_the_horizon():
    session = Session(
        task_uid="n0",
        task_id="t",
        run_id="r",
        episode_index=0,
        prompt="",
        checkpoint="",
        model={"action_horizon": 16},
    )
    assert session.execution_steps == 16


def test_one_dimensional_actions_are_promoted_to_a_chunk(wiring):
    bus, keys, server, _, actions = wiring
    _spec(bus, keys)
    server.infer = lambda session, frame: np.zeros(6, dtype=np.float32)  # type: ignore[method-assign]
    bus.publish(keys.task_obs("n0"), encode(ObsFrame(task_uid="n0")))
    assert actions[0].horizon == 1
    assert actions[0].dof == 6


def test_close_unloads_every_session(wiring):
    bus, keys, server, _, _ = wiring
    _spec(bus, keys, uid="n0")
    _spec(bus, keys, uid="n1")
    server.close()
    assert len(server.unloaded) == 2
