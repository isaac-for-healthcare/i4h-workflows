# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn a Unitree G1 in place to face a named scene object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext
from i4h_tasks.basic.motion.g1_locomotion import G1_WBC_WIDTH, heading_error_to_position, set_g1_wbc_command


class G1FaceObject(Task):
    """Rotate G1 until its local forward axis points at an object's origin."""

    advance_on_success = True
    requires: ClassVar[dict[str, object]] = {
        "embodiment": "g1",
        "action_space": "joint_position",
        "dof": G1_WBC_WIDTH,
        "robots": ["robot"],
    }

    @dataclass
    class Outputs:
        heading_error_deg: float = math.inf

    def __init__(
        self,
        *,
        object: str,
        robot: str = "robot",
        heading_tolerance_deg: float = 2.0,
        stable_s: float = 0.25,
        max_yaw_rate_rad_s: float = 0.5,
        heading_gain: float = 1.0,
        base_height_m: float = 0.75,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if heading_tolerance_deg <= 0.0:
            raise ValueError("heading_tolerance_deg must be positive")
        self.object = object
        self.robot = robot
        self.heading_tolerance_rad = math.radians(heading_tolerance_deg)
        self.stable_s = stable_s
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self.heading_gain = heading_gain
        self.base_height_m = base_height_m
        self._stable_ticks = 0
        self._heading_error = np.array([math.inf], dtype=np.float32)

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._stable_ticks = 0

    def tick(self, ctx: TickContext) -> Status:
        root = ctx.scene.robot_root(self.robot)
        target = ctx.scene.object(self.object)
        self._heading_error = heading_error_to_position(
            root.pose.pos,
            root.pose.quat,
            target.pose.pos,
        )
        aligned = np.abs(self._heading_error) <= self.heading_tolerance_rad
        if bool(aligned.all()):
            self._stable_ticks += 1
        else:
            self._stable_ticks = 0

        yaw_rate = np.clip(
            self.heading_gain * self._heading_error,
            -self.max_yaw_rate_rad_s,
            self.max_yaw_rate_rad_s,
        ).astype(np.float32)
        yaw_rate[aligned] = 0.0
        navigation = np.column_stack([np.zeros(ctx.num_envs), np.zeros(ctx.num_envs), yaw_rate]).astype(np.float32)
        set_g1_wbc_command(
            ctx,
            navigation=navigation,
            robot=self.robot,
            base_height_m=self.base_height_m,
        )

        required_ticks = max(1, math.ceil(self.stable_s / ctx.dt))
        return Status.SUCCESS if self._stable_ticks >= required_ticks else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(
            heading_error_deg=float(np.degrees(np.max(np.abs(self._heading_error)))),
        )
