# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand control to a human until they stop, a predicate fires, or time runs out.

Teleop as a task: a human-driven segment can be spliced
into an otherwise rule-based or policy-driven workflow, and the recording is
node-tagged either way."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from i4h_common.types import satisfied
from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext
from i4h_tasks.teleop.devices import InputDevice, make_device

logger = logging.getLogger("i4h_tasks.teleop")


class Drive(Task):
    """Hand control to a human until they stop, a predicate fires, or time runs out."""

    @dataclass
    class Inputs:
        device: str = "keyboard"

    @dataclass
    class Outputs:
        frames: int = 0
        completed: bool = False

    def __init__(
        self,
        *,
        device: str = "keyboard",
        until: Callable[[TickContext], Any] | None = None,
        max_seconds: float = 300.0,
        robot: str = "robot",
        name: str | None = None,
        **device_kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        self.device_name = device
        self.device_kwargs = device_kwargs
        self.until = until
        self.max_seconds = max_seconds
        self.robot = robot
        self._device: InputDevice | None = None
        self._frames = 0
        self._ticks = 0
        self._completed = False

    def on_enter(self, ctx: TickContext, inputs: Inputs) -> None:
        name = getattr(inputs, "device", "") or self.device_name
        self._device = make_device(name, **self.device_kwargs)
        self._device.open(ctx)
        self._frames = 0
        self._ticks = 0
        self._completed = False
        logger.info("teleop started on %s", name)

    def tick(self, ctx: TickContext) -> Status:
        assert self._device is not None
        self._ticks += 1

        if self.until is not None and satisfied(self.until(ctx)):
            self._completed = True
            return Status.SUCCESS
        if self._device.done:
            self._completed = True
            return Status.SUCCESS
        if self._ticks * ctx.dt > self.max_seconds:
            logger.warning("teleop hit its %ss budget", self.max_seconds)
            return Status.FAILURE

        command = self._device.read(ctx)
        if command is None:
            # No new sample this tick is normal for a polled device; holding is
            # the honest behaviour, and it keeps the arm where the human left it.
            ctx.act.hold(self.robot)
        else:
            apply_action(ctx.act, command, self.robot)
            self._frames += 1
        return Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        self._teardown()
        return self.Outputs(frames=self._frames, completed=self._completed)

    def on_abort(self, ctx: TickContext) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
