# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The client speaks the pinned ``PolicyServer`` wire format without a socket.

A fake ``zmq`` module stands in for pyzmq, so the request bytes, the socket
options and the recovery after a timeout are pinned without a server. msgpack
is reached through ``importorskip``: the light venv does not promise it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.client import protocol  # noqa: E402

# -- codec ------------------------------------------------------------------


def test_arrays_round_trip_through_the_wire_dict() -> None:
    for array in (
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.full((4, 4), 200, dtype=np.uint8),
        np.array(1.5),
    ):
        wire = protocol.encode_ndarray(array)
        assert wire[protocol.NDARRAY_MARKER] is True and isinstance(wire[protocol.NDARRAY_PAYLOAD], bytes)
        back = protocol.decode_ndarray(wire)
        assert back.dtype == array.dtype and back.shape == array.shape
        np.testing.assert_array_equal(back, array)


def test_decode_accepts_bytes_keys_and_passes_other_values_through() -> None:
    wire = protocol.encode_ndarray(np.arange(3))
    as_bytes = {key.encode(): value for key, value in wire.items()}
    np.testing.assert_array_equal(protocol.decode_ndarray(as_bytes), np.arange(3))
    assert protocol.decode_ndarray({"status": "ok"}) == {"status": "ok"}
    assert protocol.decode_ndarray([1, 2]) == [1, 2]
    assert protocol.decode_ndarray("x") == "x"


def test_encode_rejects_anything_but_arrays() -> None:
    with pytest.raises(TypeError):
        protocol.encode_ndarray(object())


def test_pack_and_unpack_carry_nested_arrays() -> None:
    pytest.importorskip("msgpack")
    message = {"endpoint": "get_action", "data": {"observation": {"state": {"left_arm": np.ones((1, 1, 7))}}}}
    back = protocol.unpack(protocol.pack(message))
    assert back["endpoint"] == "get_action"
    np.testing.assert_array_equal(back["data"]["observation"]["state"]["left_arm"], np.ones((1, 1, 7)))


# -- the REQ client against a fake zmq ---------------------------------------


class _FakeSocket:
    def __init__(self, replies: list) -> None:
        self.replies = replies
        self.sent: list[bytes] = []
        self.options: dict[int, int] = {}
        self.connected: list[str] = []
        self.closed = False

    def setsockopt(self, option: int, value: int) -> None:
        self.options[option] = value

    def connect(self, address: str) -> None:
        self.connected.append(address)

    def send(self, raw: bytes) -> None:
        self.sent.append(raw)

    def recv(self) -> bytes:
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, linger: int | None = None) -> None:
        self.closed = True


class _FakeContext:
    def __init__(self) -> None:
        self.sockets: list[_FakeSocket] = []
        self.terminated = 0
        self.replies: list = []

    def socket(self, kind: int) -> _FakeSocket:
        assert kind == _REQ
        socket = _FakeSocket(self.replies)
        self.sockets.append(socket)
        return socket

    def term(self) -> None:
        self.terminated += 1


_REQ, _RCVTIMEO, _SNDTIMEO, _LINGER = 3, 27, 28, 17


@pytest.fixture
def fake_zmq(monkeypatch):
    pytest.importorskip("msgpack")
    module = types.ModuleType("zmq")
    context = _FakeContext()
    module.Context = lambda: context  # type: ignore[attr-defined]
    module.REQ, module.RCVTIMEO, module.SNDTIMEO, module.LINGER = _REQ, _RCVTIMEO, _SNDTIMEO, _LINGER  # type: ignore[attr-defined]
    module.Again = type("Again", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zmq", module)
    module.context = context  # type: ignore[attr-defined]
    return module


def _client(fake_zmq, **kwargs) -> protocol.GR00TClient:
    return protocol.GR00TClient(port=5599, **kwargs)


def test_the_socket_is_configured_and_connected_like_the_old_client(fake_zmq) -> None:
    client = _client(fake_zmq, timeout_ms=1234)
    socket = fake_zmq.context.sockets[-1]
    assert socket.options == {_RCVTIMEO: 1234, _SNDTIMEO: 1234, _LINGER: 0}
    assert socket.connected == ["tcp://localhost:5599"]
    client.close()


def test_get_action_sends_handler_kwargs_and_splits_the_two_list_reply(fake_zmq) -> None:
    chunk = {"left_arm": np.zeros((1, 32, 7), dtype=np.float32)}
    fake_zmq.context.replies.append(protocol.pack([chunk, {"took_ms": 3}]))
    client = _client(fake_zmq)
    observation = {"state": {"left_arm": np.ones((1, 1, 7), dtype=np.float32)}, "language": {"k": [["task"]]}}

    action, info = client.get_action(observation)

    request = protocol.unpack(fake_zmq.context.sockets[-1].sent[0])
    assert set(request) == {"endpoint", "data"} and request["endpoint"] == "get_action"
    assert set(request["data"]) == {"observation", "options"} and request["data"]["options"] is None
    np.testing.assert_array_equal(request["data"]["observation"]["state"]["left_arm"], observation["state"]["left_arm"])
    assert request["data"]["observation"]["language"] == {"k": [["task"]]}
    assert action["left_arm"].shape == (1, 32, 7) and action["left_arm"].dtype == np.float32
    assert info == {"took_ms": 3}


def test_get_action_forwards_options_and_wraps_a_bare_reply(fake_zmq) -> None:
    fake_zmq.context.replies.append(protocol.pack({"left_arm": np.zeros((1, 2, 7))}))
    client = _client(fake_zmq)
    action, info = client.get_action({}, options={"trex_manual": True, "trex_force_slow": False})
    request = protocol.unpack(fake_zmq.context.sockets[-1].sent[0])
    assert request["data"]["options"] == {"trex_manual": True, "trex_force_slow": False}
    assert set(action) == {"left_arm"} and info == {}


def test_input_free_endpoints_send_no_data_key(fake_zmq) -> None:
    fake_zmq.context.replies.append(protocol.pack({"status": "ok", "message": "Server is running"}))
    fake_zmq.context.replies.append(protocol.pack({"task": "t", "cascade": {"enabled": True}}))
    client = _client(fake_zmq)
    assert client.ping() is True
    assert client.server_info() == {"task": "t", "cascade": {"enabled": True}}
    requests = [protocol.unpack(raw) for raw in fake_zmq.context.sockets[-1].sent]
    assert requests == [{"endpoint": "ping"}, {"endpoint": "server_info"}]


def test_ping_is_false_on_a_bad_status_and_on_failure(fake_zmq) -> None:
    fake_zmq.context.replies.append(protocol.pack({"status": "busy"}))
    fake_zmq.context.replies.append(RuntimeError("socket gone"))
    client = _client(fake_zmq)
    assert client.ping() is False
    assert len(fake_zmq.context.sockets) == 1
    assert client.ping() is False
    assert len(fake_zmq.context.sockets) == 2 and fake_zmq.context.sockets[0].closed


def test_reset_sends_the_options_keyword(fake_zmq) -> None:
    fake_zmq.context.replies.append(protocol.pack({}))
    _client(fake_zmq).reset()
    assert protocol.unpack(fake_zmq.context.sockets[-1].sent[0]) == {"endpoint": "reset", "data": {"options": None}}


def test_error_replies_become_server_errors(fake_zmq) -> None:
    fake_zmq.context.replies.append(protocol.pack({"error": "joint_angles_right provided but no right URDF"}))
    client = _client(fake_zmq)
    with pytest.raises(protocol.ServerError, match="no right URDF") as excinfo:
        client.get_action({})
    assert not isinstance(excinfo.value, protocol.UnknownEndpointError)


def test_an_unknown_server_info_endpoint_means_none(fake_zmq) -> None:
    fake_zmq.context.replies.append(protocol.pack({"error": "Unknown endpoint: server_info"}))
    fake_zmq.context.replies.append(protocol.pack({"error": "Unknown endpoint: server_info"}))
    fake_zmq.context.replies.append(protocol.pack("not a dict"))
    client = _client(fake_zmq)
    with pytest.raises(protocol.UnknownEndpointError):
        client.call("server_info")
    assert client.server_info() is None
    assert client.server_info() is None


def test_a_timeout_reopens_the_socket_and_raises_timeout_error(fake_zmq) -> None:
    fake_zmq.context.replies.append(fake_zmq.Again())
    fake_zmq.context.replies.append(protocol.pack({"status": "ok"}))
    client = _client(fake_zmq, timeout_ms=15000)
    with pytest.raises(TimeoutError, match=r"get_action: no reply from tcp://localhost:5599 within 15000 ms"):
        client.get_action({})
    first, second = fake_zmq.context.sockets
    assert first.closed and not second.closed
    assert second.connected == ["tcp://localhost:5599"], "a fresh REQ socket must reconnect"
    assert client.ping() is True, "the next request goes out on the new socket"


def test_close_is_idempotent(fake_zmq) -> None:
    client = _client(fake_zmq)
    client.close()
    client.close()
    assert fake_zmq.context.sockets[-1].closed and fake_zmq.context.terminated == 1


# -- helpers -----------------------------------------------------------------


class _Pinger:
    def __init__(self, answers: list[bool]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def ping(self) -> bool:
        self.calls += 1
        return self.answers.pop(0)


def test_wait_for_server_sleeps_between_failed_pings() -> None:
    sleeps: list[float] = []
    pinger = _Pinger([False, False, True])
    protocol.wait_for_server(pinger, attempts=5, interval_s=0.5, sleep=sleeps.append)  # type: ignore[arg-type]
    assert pinger.calls == 3 and sleeps == [0.5, 0.5]


def test_wait_for_server_gives_up_after_the_total_seconds() -> None:
    sleeps: list[float] = []
    with pytest.raises(RuntimeError, match="after 6s"):
        protocol.wait_for_server(_Pinger([False] * 3), attempts=3, interval_s=2.0, sleep=sleeps.append)  # type: ignore[arg-type]
    assert sleeps == [2.0, 2.0, 2.0]


def test_describe_server_info_reads_the_served_schema_and_tolerates_gaps() -> None:
    info = {
        "task": "gr00t_n17_tactile/h2_sharpa_tactile",
        "embodiment": "h2_sharpa",
        "control_hz": 30.0,
        "action_horizon": 32,
        "execution_steps": 16,
        "tactile": {
            "source": "live",
            "fingertips": ["left_thumb", "left_index"],
            "force_window": 16,
            "refresh": "per-step",
        },
        "cascade": {"enabled": True, "fast_loops": 2},
        "checkpoint": {"class": "Gr00tN1d7Tactile", "use_tactile": True, "use_tactile_force": False, "delay_aug": True},
    }
    line = protocol.describe_server_info(info)
    for fragment in (
        "task=gr00t_n17_tactile/h2_sharpa_tactile",
        "embodiment=h2_sharpa",
        "checkpoint=Gr00tN1d7Tactile",
        "use_tactile=True",
        "use_tactile_force=False",
        "delay_aug=True",
        "fingertips=2",
        "force_window=16",
        "cascade=True/2",
        "tactile_source=live",
        "tactile_refresh=per-step",
        "control_hz=30.0",
        "action_horizon=32",
        "execution_steps=16",
    ):
        assert fragment in line
    assert "\n" not in line
    empty = protocol.describe_server_info({})
    assert "checkpoint=?" in empty and "fingertips=?" in empty and "cascade=?/?" in empty
