# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read a scene object's pose and publish it as a workflow value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_common.types import Pose
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class Locate(Task):
    """Snapshot an object's pose, optionally waiting for it to settle first.

    This is the node that turns scene randomization into a workflow value: every
    downstream task consumes the located pose rather than a hard-coded constant,
    which lets one rule-based workflow survive a randomized scene.
    """

    postcondition = {"located": "$object"}

    @dataclass
    class Outputs:
        pose: Pose
        settled: bool = True

    def __init__(
        self,
        object: str,  # noqa: A002 - matches the manifest port name
        *,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        wait_for_settle: bool = False,
        settle_timeout_s: float = 2.0,
        robot: str = "robot",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"locate_{object}")
        self.object = object
        self.offset = offset
        self.wait_for_settle = wait_for_settle
        self.settle_timeout_s = settle_timeout_s
        self.robot = robot
        self._pose: Pose | None = None
        self._settled = True
        self._ticks = 0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._pose = None
        self._settled = True
        self._ticks = 0

    def tick(self, ctx: TickContext) -> Status:
        self._ticks += 1
        state = ctx.scene.object(self.object)
        # Locate does not command the robot; say so explicitly rather than
        # leaving the actuator unwritten, which would read as "nobody is driving".
        ctx.act.hold(self.robot)

        if self.wait_for_settle and not bool(np.asarray(state.is_settled).all()):
            if self._ticks * ctx.dt < self.settle_timeout_s:
                return Status.RUNNING
            self._settled = False

        self._pose = state.pose.translated(self.offset) if any(self.offset) else state.pose
        return Status.SUCCESS

    def on_exit(self, ctx: TickContext) -> Outputs:
        assert self._pose is not None
        return self.Outputs(pose=self._pose, settled=self._settled)

    def describe(self) -> str:
        return f"locate {self.object!r}"
