# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the jaw to a width and hold there."""

from __future__ import annotations

from dataclasses import dataclass

from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class SetGripper(Task):
    """Drive the jaw to ``width`` over ``duration_s`` and hold there."""

    requires = {"gripper": True}

    @dataclass
    class Outputs:
        width: float = 0.0

    def __init__(self, width: float, *, duration_s: float = 0.3, name: str | None = None) -> None:
        super().__init__(name=name)
        self.width = width
        self.duration_s = duration_s
        self._steps = 1
        self._tick = 0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._steps = max(1, round(self.duration_s / ctx.dt))
        self._tick = 0

    def tick(self, ctx: TickContext) -> Status:
        self._tick += 1
        ctx.act.set_gripper(self.width)
        return Status.SUCCESS if self._tick >= self._steps else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(width=self.width)
