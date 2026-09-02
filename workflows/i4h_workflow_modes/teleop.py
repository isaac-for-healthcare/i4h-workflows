# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``teleop`` mode: a human drives, but it is still a workflow.

Recording through a workflow is what keeps the demo node-tagged, so mimic and the
annotator can later work on one skill rather than a whole episode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from i4h_engine.graph import TaskGraph, task


def teleop(
    device: str = "keyboard",
    *,
    until: Callable[..., Any] | None = None,
    **device_kwargs: Any,
) -> TaskGraph:
    return TaskGraph(description=f"Human demonstration via {device}.").flow(
        task("teleop/drive", device=device, until=until, **device_kwargs)
    )
