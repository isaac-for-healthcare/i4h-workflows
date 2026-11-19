# SPDX-License-Identifier: Apache-2.0
"""The approved 115-command normalized static-aorta demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class AortaFluoroscopySweep(Task):
    """Settle, insert, rotate, bend, unbend, retract, and hold at 30 Hz."""

    requires = {"action_space": "catheter_normalized_velocity", "dof": 3}
    advance_on_success = True
    _phases = (
        (15, (0.0, 0.0, 0.0)),
        (30, (1.0, 0.0, 0.0)),
        (15, (0.0, 0.5, 0.0)),
        (15, (0.0, 0.0, 0.5)),
        (15, (0.0, 0.0, -0.5)),
        (10, (-1.0, 0.0, 0.0)),
        (15, (0.0, 0.0, 0.0)),
    )

    @dataclass
    class Outputs:
        commands: int = 0

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._index = 0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._index = 0

    def tick(self, ctx: TickContext) -> Status:
        remaining = self._index
        for count, command in self._phases:
            if remaining < count:
                apply_action(ctx.act, np.tile(np.asarray(command, dtype=np.float32), (ctx.num_envs, 1)))
                self._index += 1
                return Status.SUCCESS if self._index == 115 else Status.RUNNING
            remaining -= count
        return Status.SUCCESS

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(commands=self._index)
