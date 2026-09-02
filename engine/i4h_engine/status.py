# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task and workflow completion states."""

from __future__ import annotations

from enum import Enum


class Status(Enum):
    """What a task reports from ``tick``."""

    WAITING = "waiting"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"

    @property
    def is_terminal(self) -> bool:
        return self in (Status.SUCCESS, Status.FAILURE)

    @classmethod
    def coerce(cls, value: object) -> Status:
        """Accept a ``Status``, its string name, or a bool (``True`` → SUCCESS)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.SUCCESS if value else cls.FAILURE
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError as exc:
                raise ValueError(f"{value!r} is not a Status") from exc
        raise TypeError(f"cannot interpret {value!r} as a Status")


class WorkflowStatus(Enum):
    """What the engine reports from ``tick``."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"

    @property
    def is_terminal(self) -> bool:
        return self in (WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.ABORTED)
