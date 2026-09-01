# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from i4h_common.bus.base import Latest
from i4h_common.bus.inproc import InProcBus, _matches
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import ActionChunk, ObsFrame, WorkflowEventMsg, decode, encode
from i4h_common.bus.zenoh_bus import _default_config


def test_keys_layout():
    keys = Keys("scissor_pick_and_place")
    assert keys.task_obs("n0") == "i4h/scissor_pick_and_place/task/n0/obs"
    assert keys.task_action("n0") == "i4h/scissor_pick_and_place/task/n0/action"
    assert keys.camera("room") == "i4h/scissor_pick_and_place/camera/room"
    assert keys.workflow_events == "i4h/scissor_pick_and_place/workflow/events"


def test_keys_sanitize_namespace():
    assert Keys("my workflow/v2").root == "i4h/my-workflow-v2"


def test_keys_reject_unknown_channel():
    with pytest.raises(ValueError, match="unknown task channel"):
        Keys("p").task("n0", "nope")


def test_zenoh_endpoint_environment(monkeypatch):
    import zenoh

    monkeypatch.delenv("ZENOH_CONFIG", raising=False)
    monkeypatch.setenv("I4H_ZENOH_CONNECT", "tcp/10.0.0.8:7447, tcp/10.0.0.9:7447")
    monkeypatch.setenv("I4H_ZENOH_LISTEN", "tcp/0.0.0.0:7447")
    config = _default_config(zenoh)
    assert json.loads(config.get_json("connect/endpoints")) == ["tcp/10.0.0.8:7447", "tcp/10.0.0.9:7447"]
    assert json.loads(config.get_json("listen/endpoints")) == ["tcp/0.0.0.0:7447"]


def test_encode_decode_roundtrip():
    chunk = ActionChunk(task_uid="n0", horizon=2, dof=3, actions=[1, 2, 3, 4, 5, 6])
    restored = decode(encode(chunk), ActionChunk)
    assert restored.task_uid == "n0"
    assert restored.reshape() == [[1, 2, 3], [4, 5, 6]]


def test_decode_infers_type_without_hint():
    event = WorkflowEventMsg(workflow="p", event="node_entered", node="grasp")
    restored = decode(encode(event))
    assert isinstance(restored, WorkflowEventMsg)
    assert restored.node == "grasp"


def test_decode_tolerates_unknown_fields():
    # A newer producer adding a field must not break an older consumer.
    import msgpack

    payload = msgpack.packb({"__type__": "WorkflowEventMsg", "node": "grasp", "brand_new": 42})
    restored = decode(payload)
    assert restored.node == "grasp"


def test_decode_rejects_type_mismatch():
    with pytest.raises(ValueError, match="expected ActionChunk"):
        decode(encode(WorkflowEventMsg(node="x")), ActionChunk)


def test_inproc_pubsub():
    bus = InProcBus()
    seen: list[bytes] = []
    bus.subscribe("a/b", lambda _k, payload: seen.append(payload))
    bus.publish("a/b", b"hello")
    bus.publish("a/c", b"ignored")
    assert seen == [b"hello"]


def test_inproc_wildcard():
    bus = InProcBus()
    seen: list[str] = []
    bus.subscribe("i4h/p/camera/*", lambda key, _p: seen.append(key))
    bus.publish("i4h/p/camera/room", b"")
    bus.publish("i4h/p/camera/wrist", b"")
    bus.publish("i4h/p/robot/state", b"")
    assert seen == ["i4h/p/camera/room", "i4h/p/camera/wrist"]


def test_wildcard_matcher():
    assert _matches("a/*/c", "a/b/c")
    assert not _matches("a/*/c", "a/b/d")
    assert not _matches("a/*", "a/b/c")  # * spans exactly one segment
    assert _matches("a/**", "a/b/c")


def test_unsubscribe_stops_delivery():
    bus = InProcBus()
    seen: list[bytes] = []
    subscription = bus.subscribe("k", lambda _k, p: seen.append(p))
    bus.publish("k", b"1")
    subscription.close()
    bus.publish("k", b"2")
    assert seen == [b"1"]


def test_latest_keeps_only_newest():
    bus = InProcBus()
    latest: Latest[ObsFrame] = Latest(bus, "obs", ObsFrame)
    bus.publish("obs", encode(ObsFrame(task_uid="n0", step=1)))
    bus.publish("obs", encode(ObsFrame(task_uid="n0", step=2)))
    assert latest.count == 2
    assert latest.get().step == 2


def test_latest_take_clears():
    bus = InProcBus()
    latest: Latest[ObsFrame] = Latest(bus, "obs", ObsFrame)
    bus.publish("obs", encode(ObsFrame(step=7)))
    assert latest.take().step == 7
    assert latest.take() is None


def test_latest_ignores_malformed_payload():
    # A bad frame must never propagate into the sim loop.
    bus = InProcBus()
    latest: Latest[ObsFrame] = Latest(bus, "obs", ObsFrame)
    bus.publish("obs", b"not-msgpack")
    assert latest.get() is None
    assert latest.count == 0
