# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ZMQ request/reply client for Isaac-GR00T's ``PolicyServer``.

Speaks the wire format of ``gr00t/policy/server_client.py`` at the pinned
revision with pyzmq, msgpack and numpy alone, so the deployment machine needs
no gr00t install. A request is a msgpack map ``{"endpoint": str, "data": dict}``
whose ``data`` keys become the handler's keyword arguments (``data`` is left out
for endpoints that take no input: ``ping``, ``server_info``); the reply is the
msgpack of the handler's result; ndarrays travel both ways as ``np.save`` bytes
under a marker key; a failing handler answers ``{"error": str}``, and an unknown
endpoint is only recognizable by the ``"Unknown endpoint"`` text inside that
string. The i4h server adds ``server_info`` on top of ``ping``, ``reset`` and
``get_action``.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.client")

#: Key that marks a msgpack map as a serialized ndarray.
NDARRAY_MARKER = "__ndarray_class__"
#: Key holding the ``np.save`` bytes inside such a map.
NDARRAY_PAYLOAD = "as_npy"


def encode_ndarray(obj: Any) -> dict[str, Any]:
    """msgpack ``default`` hook: an ndarray to the GR00T wire dict; TypeError for anything else."""
    if isinstance(obj, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, obj, allow_pickle=False)
        return {NDARRAY_MARKER: True, NDARRAY_PAYLOAD: buffer.getvalue()}
    raise TypeError(f"cannot serialize {type(obj).__name__} for the GR00T server")


def decode_ndarray(obj: Any) -> Any:
    """msgpack ``object_hook``: the GR00T wire dict back to an ndarray (str or bytes keys), else ``obj``."""
    if not isinstance(obj, dict):
        return obj
    if NDARRAY_MARKER in obj or NDARRAY_MARKER.encode() in obj:
        key = NDARRAY_PAYLOAD if NDARRAY_PAYLOAD in obj else NDARRAY_PAYLOAD.encode()
        return np.load(io.BytesIO(obj[key]), allow_pickle=False)
    return obj


def pack(data: Any) -> bytes:
    """Serialize one request or reply the way the server does."""
    import msgpack

    return msgpack.packb(data, default=encode_ndarray)


def unpack(raw: bytes) -> Any:
    """Deserialize one request or reply the way the server does."""
    import msgpack

    return msgpack.unpackb(raw, object_hook=decode_ndarray)


class ServerError(RuntimeError):
    """The server answered ``{"error": ...}``; ``str(exc)`` is the server's message."""


class UnknownEndpointError(ServerError):
    """``ServerError`` whose message contains ``"Unknown endpoint"`` (the server predates the endpoint)."""


class GR00TClient:
    """Single-socket REQ client.

    Not thread-safe: exactly one owner thread at a time. In synchronous runs
    that is the control thread; in asynchronous runs the policy worker owns it
    from ``start()`` to ``stop()`` (see ``execution.py``).
    """

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000) -> None:
        import zmq

        self._zmq = zmq
        self.host = host
        self.port = port
        self.timeout_ms = int(timeout_ms)
        self._context: Any = zmq.Context()
        self._socket: Any = None
        self._open_socket()

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def _open_socket(self) -> None:
        """Replace the REQ socket. A fresh REQ socket is unbound, so it must reconnect,
        otherwise every later request times out and the ping-retry loop never recovers."""
        if self._context is None:
            raise RuntimeError("GR00TClient is closed")
        zmq = self._zmq
        if self._socket is not None:
            self._socket.close(linger=0)
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        self._socket = socket

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """One request/reply. ``data`` None sends no ``"data"`` key.

        A timeout leaves a REQ socket waiting for a reply that never comes, so
        the socket is replaced before ``TimeoutError`` is raised and the next
        call can send again. An error reply raises ``UnknownEndpointError`` when
        the server does not know the endpoint, ``ServerError`` otherwise.
        """
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(pack(request))
            raw = self._socket.recv()
        except self._zmq.Again:
            self._open_socket()
            raise TimeoutError(f"{endpoint}: no reply from {self.endpoint} within {self.timeout_ms} ms") from None
        reply = unpack(raw)
        if isinstance(reply, dict) and "error" in reply:
            message = str(reply["error"])
            if "Unknown endpoint" in message:
                raise UnknownEndpointError(message)
            raise ServerError(message)
        return reply

    def ping(self) -> bool:
        """True iff the server answers ``{"status": "ok"}``; any failure reopens the socket and is False."""
        try:
            reply = self.call("ping")
            return isinstance(reply, dict) and reply.get("status") == "ok"
        except Exception:
            self._open_socket()
            return False

    def reset(self, options: dict[str, Any] | None = None) -> Any:
        return self.call("reset", {"options": options})

    def server_info(self) -> dict[str, Any] | None:
        """What the server says about the checkpoint it serves, or None when it has no such endpoint."""
        try:
            reply = self.call("server_info")
        except UnknownEndpointError:
            return None
        return reply if isinstance(reply, dict) else None

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(action_chunk, info)``; msgpack turns the server's tuple into a 2-list, anything else gets ``{}`` info."""
        reply = self.call("get_action", {"observation": observation, "options": options})
        if isinstance(reply, list | tuple) and len(reply) == 2:
            return reply[0], reply[1]
        return reply, {}

    def close(self) -> None:
        """Release the socket and the context; safe to call more than once."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            finally:
                self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            finally:
                self._context = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def wait_for_server(
    client: GR00TClient, attempts: int = 30, interval_s: float = 2.0, sleep: Callable[[float], None] = time.sleep
) -> None:
    """Ping until the server answers, or raise after ``attempts * interval_s`` seconds."""
    for attempt in range(attempts):
        if client.ping():
            logger.info("GR00T server: connected.")
            return
        logger.info("waiting for server... (%d/%d)", attempt + 1, attempts)
        sleep(interval_s)
    raise RuntimeError(f"Could not reach GR00T server after {attempts * interval_s:.0f}s. Is it running?")


def describe_server_info(info: dict[str, Any]) -> str:
    """One log line with the facts the client acts on; a missing key prints as ``?``."""

    def get(*path: str) -> Any:
        node: Any = info
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return "?"
            node = node[key]
        return node

    fingertips = get("tactile", "fingertips")
    count = len(fingertips) if isinstance(fingertips, list | tuple) else "?"
    return (
        f"server_info: task={get('task')} embodiment={get('embodiment')} checkpoint={get('checkpoint', 'class')} "
        f"use_tactile={get('checkpoint', 'use_tactile')} use_tactile_force={get('checkpoint', 'use_tactile_force')} "
        f"delay_aug={get('checkpoint', 'delay_aug')} fingertips={count} force_window={get('tactile', 'force_window')} "
        f"cascade={get('cascade', 'enabled')}/{get('cascade', 'fast_loops')} "
        f"tactile_source={get('tactile', 'source')} tactile_refresh={get('tactile', 'refresh')} "
        f"control_hz={get('control_hz')} action_horizon={get('action_horizon')} "
        f"execution_steps={get('execution_steps')}"
    )
