# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``replay`` mode: play a recording back through the same scene."""

from __future__ import annotations

from typing import Any

from i4h_engine.graph import TaskGraph, task


def replay(dataset: str = "", episode: int = 0, node: str = "") -> TaskGraph:
    """``node`` replays one skill's frames; needs a segmented recording."""
    params: dict[str, Any] = {"dataset": dataset, "episode": episode}
    if node:
        params["node"] = node
    return TaskGraph(description="Replay a recorded episode.").flow(task("basic/replay", **params))
