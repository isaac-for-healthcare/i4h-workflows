# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hold the measured end-effector pose for a fixed time."""

from __future__ import annotations

from dataclasses import dataclass

from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class HoldPose(Task):
    """Keep commanding the measured TCP while a Cartesian controller settles."""

    requires = {"action_space": "ee_pose"}

    @dataclass
    class Outputs:
        waited_s: float = 0.0

    def __init__(
        self,
        seconds: float = 0.5,
        *,
        robot: str = "robot",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.seconds = seconds
        self.robot = robot
        self._ticks = 0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._ticks = 0

    def tick(self, ctx: TickContext) -> Status:
        self._ticks += 1
        ctx.act.set_ee_target(ctx.scene.tcp(self.robot), self.robot)
        return Status.SUCCESS if self._ticks * ctx.dt >= self.seconds else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(waited_s=self._ticks * ctx.dt)
