# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic catheter command used for simulation smoke validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class CatheterSweep(Task):
    """Exercise catheter controls and an autonomous C-arm orbital sweep."""

    requires = {"action_space": "catheter_carm_velocity", "dof": 3}

    @dataclass
    class Outputs:
        elapsed_s: float = 0.0

    def __init__(
        self,
        *,
        insertion_speed_mps: float = 0.012,
        rotation_rate_radps: float = 0.8,
        advance_s: float = 0.6,
        rotate_s: float = 0.6,
        retract_s: float = 0.3,
        orbit_s: float = 1.2,
        orbit_rate_radps: float = 0.45,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.insertion_speed_mps = float(insertion_speed_mps)
        self.rotation_rate_radps = float(rotation_rate_radps)
        self.advance_s = float(advance_s)
        self.rotate_s = float(rotate_s)
        self.retract_s = float(retract_s)
        self.orbit_s = float(orbit_s)
        self.orbit_rate_radps = float(orbit_rate_radps)
        self._ticks = 0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._ticks = 0

    def tick(self, ctx: TickContext) -> Status:
        elapsed = self._ticks * ctx.dt
        advance_end = self.advance_s
        rotate_end = advance_end + self.rotate_s
        retract_end = rotate_end + self.retract_s
        orbit_positive_end = retract_end + self.orbit_s
        orbit_return_end = orbit_positive_end + self.orbit_s
        if elapsed < advance_end:
            command = (self.insertion_speed_mps, 0.0, 0.0)
        elif elapsed < rotate_end:
            command = (0.0, self.rotation_rate_radps, 0.0)
        elif elapsed < retract_end:
            command = (-self.insertion_speed_mps, 0.0, 0.0)
        elif elapsed < orbit_positive_end:
            command = (0.0, 0.0, self.orbit_rate_radps)
        elif elapsed < orbit_return_end:
            command = (0.0, 0.0, -self.orbit_rate_radps)
        else:
            command = (0.0, 0.0, 0.0)
        apply_action(ctx.act, np.tile(np.asarray(command, dtype=np.float32), (ctx.num_envs, 1)))
        self._ticks += 1
        return Status.SUCCESS if elapsed >= orbit_return_end else Status.RUNNING

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(elapsed_s=self._ticks * ctx.dt)
