# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The small, author-facing workflow contract.

An authored module exports one :data:`WORKFLOW` value. Each named mode builds a
task graph; graph construction and execution remain separate engine concerns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from i4h_engine.graph import TaskGraph

TaskGraphBuilder = Callable[..., "TaskGraph"]
GoalPredicate = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class Workflow:
    """One authored capability: its Scene, modes, and optional goal predicate."""

    scene: str
    modes: Mapping[str, TaskGraphBuilder]
    default_mode: str = "idle"
    success: GoalPredicate | None = None

    def __post_init__(self) -> None:
        if not self.scene:
            raise ValueError("workflow scene must not be empty")
        if not self.modes:
            raise ValueError("workflow must expose at least one mode")
        normalized = dict(self.modes)
        if any(not name for name in normalized):
            raise ValueError("workflow mode names must not be empty")
        if self.default_mode not in normalized:
            raise ValueError(f"default mode {self.default_mode!r} is not exposed; available: {sorted(normalized)}")
        if any(not callable(builder) for builder in normalized.values()):
            raise TypeError("every workflow mode must be callable")
        object.__setattr__(self, "modes", MappingProxyType(normalized))
