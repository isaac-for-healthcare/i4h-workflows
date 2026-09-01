# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow telemetry.

The engine emits one of these at every state change. Three consumers exist:
the log, the HDF5 recorder (which turns ``node_entered``/``node_succeeded``
pairs into frame segments), and the zenoh ``workflow/events`` topic that makes a
rollout observable from outside the process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class EventKind:
    """String constants; kept plain so they survive the msgpack round-trip."""

    WORKFLOW_STARTED = "workflow_started"
    NODE_ENTERED = "node_entered"
    NODE_SUCCEEDED = "node_succeeded"
    NODE_FAILED = "node_failed"
    NODE_RETRYING = "node_retrying"
    NODE_ABORTED = "node_aborted"
    WORKFLOW_FINISHED = "workflow_finished"

    ALL = (
        WORKFLOW_STARTED,
        NODE_ENTERED,
        NODE_SUCCEEDED,
        NODE_FAILED,
        NODE_RETRYING,
        NODE_ABORTED,
        WORKFLOW_FINISHED,
    )


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    kind: str
    workflow: str
    step: int = 0
    node: str = ""
    task_id: str = ""
    episode_index: int = 0
    run_id: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def __str__(self) -> str:
        parts = [f"step={self.step}", self.kind]
        if self.node:
            parts.append(f"node={self.node}")
        if self.detail:
            parts.append(f"({self.detail})")
        return " ".join(parts)


EventSink = Callable[[WorkflowEvent], None]
