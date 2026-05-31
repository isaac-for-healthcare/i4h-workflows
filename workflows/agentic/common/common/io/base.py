# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

from common.messages import Message, encode_message
from common.zenoh_utils import close_quietly, payload_to_bytes

MessageT = TypeVar("MessageT", bound=Message)


class MessagePublisher(Generic[MessageT]):
    key_expr: ClassVar[str | None] = None

    def __init__(self, session: Any, key_expr: str | None = None) -> None:
        key_expr = key_expr or self.key_expr
        if key_expr is None:
            raise ValueError("key_expr is required")
        self._publisher = session.declare_publisher(key_expr)

    def close(self) -> None:
        close_quietly(self._publisher)

    def publish(self, message: MessageT) -> None:
        self._publisher.put(encode_message(message))


class MessageSubscriber(Generic[MessageT]):
    key_expr: ClassVar[str | None] = None
    decoder: ClassVar[Callable[[bytes], Any] | None] = None

    def __init__(
        self,
        session: Any,
        callback: Callable[[MessageT], None],
        key_expr: str | None = None,
        decoder: Callable[[bytes], MessageT] | None = None,
    ) -> None:
        key_expr = key_expr or self.key_expr
        decoder = decoder or type(self).decoder
        if key_expr is None:
            raise ValueError("key_expr is required")
        if decoder is None:
            raise ValueError("decoder is required")
        self._decoder = decoder
        self._callback = callback
        self._subscriber = session.declare_subscriber(key_expr, self._on_message)

    def close(self) -> None:
        close_quietly(self._subscriber)

    def _on_message(self, sample: Any) -> None:
        try:
            message = self._decoder(payload_to_bytes(sample.payload))
        except Exception:
            return
        self._callback(message)
