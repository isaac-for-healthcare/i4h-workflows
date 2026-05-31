# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def open_zenoh_session() -> Any:
    import zenoh

    return zenoh.open(zenoh.Config())


def payload_to_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray | memoryview):
        return bytes(payload)
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return bytes(payload)


def close_quietly(resource: Any) -> None:
    for method_name in ("undeclare", "close"):
        method = getattr(resource, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            return
