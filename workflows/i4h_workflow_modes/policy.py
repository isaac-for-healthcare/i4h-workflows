# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``policy`` mode: one remote node drives the whole task.

The language instruction is not written here or in the workflow module. It is
part of the remote task declaration; a runtime ``--prompt`` override can replace
that task-owned default.

This is the whole-workflow case, and only the common one: a policy is a task node
like any other, so a workflow that mixes rule-based stages with a policy-driven one
just puts ``task("<stack>/<name>")`` in whatever node position it belongs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from i4h_engine.graph import TaskGraph, task


def policy(
    task_id: str,
    *,
    until: Callable[..., Any] | None = None,
    timeout_success: Callable[..., Any] | None = None,
) -> TaskGraph:
    return TaskGraph(
        timeout_success=timeout_success,
        description=f"Policy rollout via {task_id}.",
    ).flow(task(task_id, until=until))
