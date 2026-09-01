# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``idle`` mode: open the scene and render. Nothing drives the robot."""

from __future__ import annotations

from i4h_engine.graph import TaskGraph, node
from i4h_tasks.basic.control.wait import Wait


def idle(seconds: float = 60.0) -> TaskGraph:
    return TaskGraph(description="Idle scene inspection.").flow(node(Wait(seconds, name="idle")))
