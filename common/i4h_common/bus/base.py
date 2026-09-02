# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The transport contract, plus the one consumption pattern the runtime needs.

A ``Bus`` moves bytes between processes. Two implementations exist: zenoh for
real runs and an in-process one for tests, so nothing that depends on the bus —
including ``RemoteTask`` — needs a broker to be exercised.

:class:`Latest` is the only subscriber shape the tick loop uses. A 60 Hz loop
must never block waiting for a message; it takes whatever arrived since the last
tick and moves on.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from i4h_common.bus.messages import decode

T = TypeVar("T")


@runtime_checkable
class Subscription(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class Bus(Protocol):
    """Byte transport. Implementations must be safe to call from any thread."""

    def publish(self, key: str, payload: bytes) -> None: ...

    def subscribe(self, key: str, handler: Callable[[str, bytes], None]) -> Subscription: ...

    def close(self) -> None: ...


class Latest(Generic[T]):
    """Keeps only the most recent decoded message on a key.

    Deliberately lossy. For observation/action streams the newest value is the
    only one that matters, and queueing would let a slow backend build unbounded
    lag into the control loop.
    """

    def __init__(self, bus: Bus, key: str, message_type: type[T]) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None
        self._count = 0
        self._key = key
        self._message_type = message_type
        self._subscription = bus.subscribe(key, self._on_message)

    def _on_message(self, _key: str, payload: bytes) -> None:
        try:
            message = decode(payload, self._message_type)
        except Exception:  # noqa: BLE001 - a malformed frame must not kill the sim loop
            return
        with self._lock:
            self._value = message
            self._count += 1

    @property
    def key(self) -> str:
        return self._key

    @property
    def count(self) -> int:
        """Total messages accepted. Useful for 'has anything arrived yet?'."""
        with self._lock:
            return self._count

    def get(self) -> T | None:
        """Most recent message, or ``None``. Never blocks."""
        with self._lock:
            return self._value

    def take(self) -> T | None:
        """Most recent message, clearing it so the next ``take`` returns ``None``."""
        with self._lock:
            value, self._value = self._value, None
            return value

    def wait(self, timeout: float, *, poll: float = 0.01) -> T | None:
        """Block up to ``timeout`` seconds for a message.

        Only for setup paths — backend handshake, teleop calibration. Never call
        this from ``tick``.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.get()
            if value is not None:
                return value
            time.sleep(poll)
        return self.get()

    def close(self) -> None:
        close = getattr(self._subscription, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Latest[T]:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
