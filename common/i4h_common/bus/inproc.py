# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process bus: same contract as zenoh, no broker.

This is what lets ``RemoteTask`` — the trickiest piece of the runtime — be
tested end-to-end against a fake backend in a unit test, with no network and no
policy stack installed.

Delivery is synchronous on the publishing thread, which makes tests
deterministic: after ``publish`` returns, every handler has run.
"""

from __future__ import annotations

import fnmatch
import threading
from collections.abc import Callable


class _Subscription:
    def __init__(self, bus: InProcBus, key: str, handler: Callable[[str, bytes], None]) -> None:
        self._bus = bus
        self._key = key
        self._handler = handler
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._bus._unsubscribe(self._key, self._handler)
            self._closed = True


class InProcBus:
    """A :class:`~i4h_common.bus.base.Bus` backed by a dict of handlers.

    Supports the same ``*`` wildcard segment matching that zenoh key
    expressions use, so subscriber code is identical against either transport.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, list[Callable[[str, bytes], None]]] = {}
        self._closed = False
        #: Every (key, payload) ever published — handy for asserting in tests.
        self.published: list[tuple[str, bytes]] = []

    def publish(self, key: str, payload: bytes) -> None:
        if self._closed:
            raise RuntimeError("bus is closed")
        with self._lock:
            self.published.append((key, bytes(payload)))
            targets = [
                handler
                for pattern, handlers in self._handlers.items()
                if _matches(pattern, key)
                for handler in handlers
            ]
        for handler in targets:
            handler(key, payload)

    def subscribe(self, key: str, handler: Callable[[str, bytes], None]) -> _Subscription:
        if self._closed:
            raise RuntimeError("bus is closed")
        with self._lock:
            self._handlers.setdefault(key, []).append(handler)
        return _Subscription(self, key, handler)

    def _unsubscribe(self, key: str, handler: Callable[[str, bytes], None]) -> None:
        with self._lock:
            handlers = self._handlers.get(key)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return
            if not handlers:
                del self._handlers[key]

    def close(self) -> None:
        with self._lock:
            self._handlers.clear()
            self._closed = True

    def __enter__(self) -> InProcBus:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _matches(pattern: str, key: str) -> bool:
    """Zenoh-style match: ``*`` spans one segment, ``**`` spans many."""
    if pattern == key:
        return True
    if "**" in pattern:
        return fnmatch.fnmatchcase(key, pattern.replace("**", "*"))
    pattern_parts = pattern.split("/")
    key_parts = key.split("/")
    if len(pattern_parts) != len(key_parts):
        return False
    return all(p == "*" or p == k for p, k in zip(pattern_parts, key_parts, strict=True))
