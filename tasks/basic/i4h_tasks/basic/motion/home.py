# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return the arm to its home pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext
from i4h_tasks.basic.motion.keyframes import smoothstep


class Home(Task):
    """Drive every joint back to the scene's home pose and settle there.

    Success requires actually arriving within ``tolerance_rad`` — several tasks
    in the legacy workflow implementation treated "commanded home" as "is home", which
    silently passed when the arm was still obstructed.
    """

    requires = {"action_space": "joint_position"}

    @dataclass
    class Outputs:
        at_home: bool = False

    def __init__(
        self,
        *,
        duration_s: float = 1.0,
        tolerance_rad: float = 0.08,
        gripper: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.duration_s = duration_s
        self.tolerance_rad = tolerance_rad
        self.gripper = gripper
        self._start: np.ndarray | None = None
        self._steps = 1
        self._tick = 0
        self._arrived = False

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._start = np.array(ctx.scene.joints().pos, dtype=np.float32, copy=True)
        self._steps = max(1, round(self.duration_s / ctx.dt))
        self._tick = 0
        self._arrived = False

    def tick(self, ctx: TickContext) -> Status:
        home = ctx.scene.home_joints()
        self._tick += 1
        alpha = smoothstep(self._tick / self._steps)
        assert self._start is not None
        ctx.act.set_joint_targets(self._start + (home - self._start) * alpha)
        if self.gripper is not None:
            ctx.act.set_gripper(self.gripper)

        if self._tick < self._steps:
            return Status.RUNNING
        error = np.abs(ctx.scene.joints().pos - home)
        if self.gripper is not None:
            # The jaw was commanded somewhere other than home on purpose, so
            # holding it to the home pose would keep this task running forever.
            error = error[..., :-1]
        self._arrived = bool((error.max(axis=-1) < self.tolerance_rad).all())
        return Status.SUCCESS if self._arrived else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(at_home=self._arrived)
