# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint-space keyframe playback for rule-based controllers without IK.

One node holds a semantic group of keyframes ("descend to the scissors"),
not a single frame: a 25-frame trajectory as 25 nodes is unreadable, while
grouping keeps each stage independently retryable and taggable."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from i4h_common.types import JointState, Pose, as_batch
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


def smoothstep(alpha: float) -> float:
    """Hermite ease-in/ease-out. Zero velocity at both ends, so no joint snap."""
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


@dataclass(frozen=True, slots=True)
class Frame:
    """One joint-space target, held for ``duration_s``."""

    name: str
    target: tuple[float, ...]
    duration_s: float = 0.33


class Keyframes(Task):
    """Play a sequence of joint targets, smoothstepping between them.

    Targets are deltas from the robot's home pose by default, which is what
    makes a trajectory portable across scenes that mount the same arm at
    different heights.

    ``offset_from`` applies a closed-loop correction: given an input pose (say
    the randomized scissors), shift one joint proportionally to that pose's
    displacement from a reference. That single adaptation is what lets an
    otherwise open-loop trajectory survive scene randomization.
    """

    requires = {"action_space": "joint_position"}

    @dataclass
    class Inputs:
        reference: Pose | None = None

    @dataclass
    class Outputs:
        joints: JointState | None = None

    def __init__(
        self,
        frames: Sequence[Frame | tuple[str, Sequence[float], float]],
        *,
        relative_to_home: bool = True,
        offset_joint: int | None = None,
        offset_axis: int = 1,
        offset_gain: float = 0.0,
        offset_reference: float = 0.0,
        offset_limits: tuple[float, float] = (0.0, 0.0),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.frames = tuple(f if isinstance(f, Frame) else Frame(f[0], tuple(f[1]), f[2]) for f in frames)
        if not self.frames:
            raise ValueError("Keyframes needs at least one keyframe")
        self.relative_to_home = relative_to_home
        self.offset_joint = offset_joint
        self.offset_axis = offset_axis
        self.offset_gain = offset_gain
        self.offset_reference = offset_reference
        self.offset_limits = offset_limits
        self._offset: np.ndarray | float = 0.0
        self._start: np.ndarray | None = None
        self._frame_index = 0
        self._frame_tick = 0
        self._steps: list[int] = []

    def on_enter(self, ctx: TickContext, inputs: Inputs) -> None:
        self._start = np.array(ctx.scene.joints().pos, dtype=np.float32, copy=True)
        self._frame_index = 0
        self._frame_tick = 0
        self._steps = [max(1, round(frame.duration_s / ctx.dt)) for frame in self.frames]
        self._offset = self._compute_offset(getattr(inputs, "reference", None), ctx)

    def _compute_offset(self, reference: Pose | None, ctx: TickContext) -> np.ndarray | float:
        if reference is None or self.offset_joint is None or self.offset_gain == 0.0:
            return 0.0
        measured = reference.pos[:, self.offset_axis]
        low, high = self.offset_limits
        return np.clip((measured - self.offset_reference) * self.offset_gain, low, high).astype(np.float32)

    def _absolute_target(self, ctx: TickContext, frame: Frame) -> np.ndarray:
        dof = ctx.scene.joints().dof
        target = as_batch(frame.target, ctx.num_envs, dof)
        if self.relative_to_home:
            target = target + ctx.scene.home_joints()
        if self.offset_joint is not None and not np.isscalar(self._offset):
            target[:, self.offset_joint] += self._offset
        return target

    def tick(self, ctx: TickContext) -> Status:
        if self._frame_index >= len(self.frames):
            return Status.SUCCESS
        frame = self.frames[self._frame_index]
        steps = self._steps[self._frame_index]
        self._frame_tick += 1

        goal = self._absolute_target(ctx, frame)
        alpha = smoothstep(self._frame_tick / steps)
        assert self._start is not None
        command = self._start + (goal - self._start) * alpha
        ctx.act.set_joint_targets(command)

        if self._frame_tick >= steps:
            self._start = goal
            self._frame_index += 1
            self._frame_tick = 0
        return Status.SUCCESS if self._frame_index >= len(self.frames) else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        # The whole JointState, not just positions: a downstream node usually
        # wants the names and velocities too, and re-reading the scene later
        # would give it a different tick's values.
        return self.Outputs(joints=ctx.scene.joints())

    def describe(self) -> str:
        return f"{self.name}: {len(self.frames)} keyframes, {sum(f.duration_s for f in self.frames):.2f}s"
