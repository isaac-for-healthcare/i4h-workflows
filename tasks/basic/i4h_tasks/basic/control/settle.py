# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wait for an object to stop moving.

Checking success immediately after a release reads a scene still in motion;
this is what makes "it landed in the tray" mean what it says."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class Settle(Task):
    """Wait for an object to stop moving.

    Placing a task's success check immediately after a release reads a scene
    that is still in motion; this is the node that makes "it landed in the tray"
    mean what it says.
    """

    @dataclass
    class Outputs:
        settled: bool = False

    def __init__(self, object: str, *, timeout_s: float = 2.0, name: str | None = None) -> None:  # noqa: A002
        super().__init__(name=name or f"settle_{object}")
        self.object = object
        self.timeout_s = timeout_s
        self._ticks = 0
        self._settled = False

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._ticks = 0
        self._settled = False

    def tick(self, ctx: TickContext) -> Status:
        self._ticks += 1
        ctx.act.hold()
        if bool(np.asarray(ctx.scene.object(self.object).is_settled).all()):
            self._settled = True
            return Status.SUCCESS
        # Timing out is not a failure: the workflow still wants to finish, it just
        # records that the scene never came to rest.
        return Status.RUNNING if self._ticks * ctx.dt < self.timeout_s else Status.SUCCESS

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(settled=self._settled)
