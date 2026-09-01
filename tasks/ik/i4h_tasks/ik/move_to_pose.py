# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Servo the end effector to a Cartesian target and hold until it arrives.

Success requires the measured TCP to be within tolerance: a commanded pose
is not an achieved pose, and treating the two as equal is how a workflow
continues with the tool nowhere near where it believes it is."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from i4h_common.types import Pose, quat_mul
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext

logger = logging.getLogger("i4h_tasks.ik.move_to_pose")


def slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc interpolation between batched ``wxyz`` quaternions.

    Nlerp with a hemisphere fix rather than true slerp: for the small
    orientation deltas these nodes command, the difference is below joint
    resolution and this has no trig and no near-parallel singularity.
    """
    b = np.where((np.sum(a * b, axis=-1, keepdims=True) < 0.0), -b, b)
    out = a * (1.0 - alpha) + b * alpha
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.divide(out, norm, out=np.tile(np.array([1.0, 0, 0, 0], np.float32), (len(out), 1)), where=norm > 1e-8)


class MoveToPose(Task):
    """Servo the end effector to a Cartesian target and hold until it arrives.

    Success requires the measured TCP to be within ``position_tolerance`` — a
    commanded pose is not an achieved pose, and treating the two as equal is how
    a workflow silently continues with the tool nowhere near where it thinks it is.
    """

    requires = {"action_space": "ee_pose"}

    @dataclass
    class Inputs:
        target: Pose
        offset: Pose | None = None

    @dataclass
    class Outputs:
        reached: bool = False
        tcp: Pose | None = None

    def __init__(
        self,
        *,
        duration_s: float = 1.0,
        position_tolerance: float = 0.005,
        settle_timeout_s: float = 2.0,
        robot: str = "robot",
        gripper: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.duration_s = duration_s
        self.position_tolerance = position_tolerance
        self.settle_timeout_s = settle_timeout_s
        self.robot = robot
        self.gripper = gripper
        self._start: Pose | None = None
        self._goal: Pose | None = None
        self._steps = 1
        self._tick = 0
        self._reached = False

    def on_enter(self, ctx: TickContext, inputs: Inputs) -> None:
        target = getattr(inputs, "target", None)
        if target is None:
            raise ValueError(f"{self.name}: no target pose; wire it from a locate node")
        offset = getattr(inputs, "offset", None)
        if offset is not None:
            target = Pose(pos=target.pos + offset.pos, quat=quat_mul(target.quat, offset.quat))
        self._start = ctx.scene.tcp(self.robot)
        self._goal = self._resolve_goal(target)
        self._steps = max(1, round(self.duration_s / ctx.dt))
        self._tick = 0
        self._reached = False
        logger.debug(
            "%s target: start=%s goal=%s",
            self.name,
            self._start.pos.round(5).tolist(),
            self._goal.pos.round(5).tolist(),
        )

    def _resolve_goal(self, target: Pose) -> Pose:
        return target

    def tick(self, ctx: TickContext) -> Status:
        assert self._start is not None and self._goal is not None
        self._tick += 1
        alpha = min(1.0, self._tick / self._steps)
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        quat = self._goal.quat if eased >= 1.0 else slerp(self._start.quat, self._goal.quat, eased)
        command = Pose(
            pos=self._start.pos + (self._goal.pos - self._start.pos) * eased,
            # At the endpoint preserve the target's exact quaternion sign.
            # q and -q are the same rotation, but IsaacLab's DLS orientation
            # error follows the supplied sign and the controller sends
            # the command-manager quaternion unchanged.
            quat=quat,
        )
        ctx.act.set_ee_target(command, self.robot)
        if self.gripper is not None:
            ctx.act.set_gripper(self.gripper, self.robot)

        if self._tick < self._steps:
            return Status.RUNNING

        error = ctx.scene.tcp(self.robot).distance_to(self._goal)
        if bool((error < self.position_tolerance).all()):
            self._reached = True
            return Status.SUCCESS
        # Keep commanding the goal while the controller closes the gap; give up
        # only after the settle budget so a stuck arm fails rather than hangs.
        overrun = (self._tick - self._steps) * ctx.dt
        if overrun < self.settle_timeout_s:
            return Status.RUNNING
        logger.warning(
            "%s failed to reach target: error_m=%s tolerance_m=%.4f current=%s target=%s",
            self.name,
            np.asarray(error).round(5).tolist(),
            self.position_tolerance,
            ctx.scene.tcp(self.robot).pos.round(5).tolist(),
            self._goal.pos.round(5).tolist(),
        )
        return Status.FAILURE

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(reached=self._reached, tcp=ctx.scene.tcp(self.robot))
