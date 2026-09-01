# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zenoh transport.

``zenoh`` is imported lazily so that merely importing ``i4h_common.bus`` — which
``engine`` does — never requires the native extension. Tests and lint run
without it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("i4h.bus")


def _default_config(zenoh: Any) -> Any:
    """Build transport config from Zenoh's file env plus simple endpoint envs."""
    config = zenoh.Config.from_env() if os.environ.get(zenoh.Config.DEFAULT_CONFIG_PATH_ENV) else zenoh.Config()
    for env_name, key in (
        ("I4H_ZENOH_CONNECT", "connect/endpoints"),
        ("I4H_ZENOH_LISTEN", "listen/endpoints"),
    ):
        endpoints = [value.strip() for value in os.environ.get(env_name, "").split(",") if value.strip()]
        if endpoints:
            config.insert_json5(key, json.dumps(endpoints))
    return config


def payload_to_bytes(payload: Any) -> bytes:
    """Normalise the several shapes zenoh has used for sample payloads."""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray | memoryview):
        return bytes(payload)
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return bytes(payload)


class _Subscription:
    def __init__(self, subscriber: Any) -> None:
        self._subscriber = subscriber

    def close(self) -> None:
        for method in ("undeclare", "close"):
            fn = getattr(self._subscriber, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    logger.debug("zenoh subscriber %s failed", method, exc_info=True)
                return


class ZenohBus:
    """A :class:`~i4h_common.bus.base.Bus` over an ``eclipse-zenoh`` session."""

    def __init__(self, session: Any | None = None, *, config: Any | None = None) -> None:
        if session is None:
            import zenoh  # noqa: PLC0415 - lazy: keeps the native dep off the import path

            session = zenoh.open(config if config is not None else _default_config(zenoh))
            self._owns_session = True
        else:
            self._owns_session = False
        self._session = session
        self._publishers: dict[str, Any] = {}
        self._subscriptions: list[_Subscription] = []

    def publish(self, key: str, payload: bytes) -> None:
        publisher = self._publishers.get(key)
        if publisher is None:
            publisher = self._session.declare_publisher(key)
            self._publishers[key] = publisher
        publisher.put(payload)

    def subscribe(self, key: str, handler: Callable[[str, bytes], None]) -> _Subscription:
        def _on_sample(sample: Any) -> None:
            try:
                handler(str(sample.key_expr), payload_to_bytes(sample.payload))
            except Exception:  # noqa: BLE001 - a bad frame must not kill the zenoh thread
                logger.warning("bus handler for %s raised", key, exc_info=True)

        subscription = _Subscription(self._session.declare_subscriber(key, _on_sample))
        self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.close()
        self._subscriptions.clear()
        for publisher in self._publishers.values():
            for method in ("undeclare", "close"):
                fn = getattr(publisher, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        logger.debug("zenoh publisher %s failed", method, exc_info=True)
                    break
        self._publishers.clear()
        if self._owns_session:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                logger.debug("zenoh session close failed", exc_info=True)

    def __enter__(self) -> ZenohBus:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_zenoh_bus(config: object | None = None) -> ZenohBus:
    """Open a zenoh-backed bus.

    Importing this module is what pulls in the native extension, so callers that
    may run without a broker should import it here rather than at module scope.
    """
    return ZenohBus(config=config)
