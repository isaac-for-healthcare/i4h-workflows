# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block until a predicate holds, or fail on timeout."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from i4h_common.types import satisfied
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class WaitUntil(Task):
    """Block until ``predicate(ctx)`` is true, or fail on timeout."""

    @dataclass
    class Outputs:
        satisfied: bool = False
        elapsed_s: float = 0.0

    def __init__(
        self,
        predicate: Callable[[TickContext], Any],
        *,
        timeout_s: float = 5.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.predicate = predicate
        self.timeout_s = timeout_s
        self._ticks = 0
        self._satisfied = False

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._ticks = 0
        self._satisfied = False

    def tick(self, ctx: TickContext) -> Status:
        self._ticks += 1
        ctx.act.hold()
        if satisfied(self.predicate(ctx), across="all"):  # every env must reach it
            self._satisfied = True
            return Status.SUCCESS
        return Status.RUNNING if self._ticks * ctx.dt < self.timeout_s else Status.FAILURE

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(satisfied=self._satisfied, elapsed_s=self._ticks * ctx.dt)
