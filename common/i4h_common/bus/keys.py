# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key expressions for the zenoh bus.

Every key is built here so the arena side and the backend side cannot drift.
Layout::

    i4h/{namespace}/task/{task_uid}/spec      arena   -> backend
    i4h/{namespace}/task/{task_uid}/obs       arena   -> backend
    i4h/{namespace}/task/{task_uid}/action    backend -> arena
    i4h/{namespace}/task/{task_uid}/status    backend -> arena
    i4h/{namespace}/camera/{name}             arena   -> out
    i4h/{namespace}/robot/state               arena   -> out
    i4h/{namespace}/robot/command             in      -> arena
    i4h/{namespace}/workflow/events               arena   -> out
    i4h/{namespace}/workflow/control              in      -> arena

``namespace`` defaults to the workflow name so two concurrent runs of different
workflows never collide; ``--namespace`` overrides it when two runs of the *same*
workflow must coexist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ILLEGAL = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize(value: str) -> str:
    """Make ``value`` safe for a single key-expression segment."""
    cleaned = _ILLEGAL.sub("-", value).strip("-")
    if not cleaned:
        raise ValueError(f"{value!r} has no usable characters for a key segment")
    return cleaned


@dataclass(frozen=True, slots=True)
class Keys:
    """Key builder for one run."""

    namespace: str
    prefix: str = "i4h"

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", sanitize(self.namespace))

    @property
    def root(self) -> str:
        return f"{self.prefix}/{self.namespace}"

    # -- task channels ---------------------------------------------------
    def task(self, task_uid: str, channel: str) -> str:
        if channel not in ("spec", "obs", "action", "status"):
            raise ValueError(f"unknown task channel {channel!r}")
        return f"{self.root}/task/{sanitize(task_uid)}/{channel}"

    def task_spec(self, task_uid: str) -> str:
        return self.task(task_uid, "spec")

    def task_obs(self, task_uid: str) -> str:
        return self.task(task_uid, "obs")

    def task_action(self, task_uid: str) -> str:
        return self.task(task_uid, "action")

    def task_status(self, task_uid: str) -> str:
        return self.task(task_uid, "status")

    # -- streams ---------------------------------------------------------
    def camera(self, name: str) -> str:
        return f"{self.root}/camera/{sanitize(name)}"

    @property
    def camera_wildcard(self) -> str:
        return f"{self.root}/camera/*"

    @property
    def robot_state(self) -> str:
        return f"{self.root}/robot/state"

    @property
    def robot_command(self) -> str:
        return f"{self.root}/robot/command"

    # -- workflow control plane ----------------------------------------------
    @property
    def workflow_events(self) -> str:
        return f"{self.root}/workflow/events"

    @property
    def workflow_control(self) -> str:
        return f"{self.root}/workflow/control"
