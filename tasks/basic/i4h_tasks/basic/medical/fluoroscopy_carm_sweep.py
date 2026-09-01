# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Autonomous image-coupling validation for the fluoroscopy C-arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext


class FluoroscopyCArmSweep(Task):
    """Orbit the C-arm and require the fluoroscopy sensor image to change."""

    requires = {"action_space": "catheter_carm_velocity", "dof": 3}

    @dataclass
    class Outputs:
        elapsed_s: float = 0.0
        max_frame_delta: float = 0.0

    def __init__(
        self,
        *,
        orbit_s: float = 1.2,
        orbit_rate_radps: float = 0.45,
        min_frame_delta: float = 2.0,
        sensor_name: str = "fluoroscopy",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.orbit_s = float(orbit_s)
        self.orbit_rate_radps = float(orbit_rate_radps)
        self.min_frame_delta = float(min_frame_delta)
        self.sensor_name = sensor_name
        self._ticks = 0
        self._baseline: np.ndarray | None = None
        self._max_frame_delta = 0.0

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        del ctx, inputs
        self._ticks = 0
        self._baseline = None
        self._max_frame_delta = 0.0

    def tick(self, ctx: TickContext) -> Status:
        frame = ctx.scene.camera(self.sensor_name)
        if frame is not None:
            image = frame.to_array().astype(np.int16, copy=False)
            if self._baseline is None:
                self._baseline = image.copy()
            else:
                delta = float(np.abs(image - self._baseline).mean())
                self._max_frame_delta = max(self._max_frame_delta, delta)

        elapsed = self._ticks * ctx.dt
        if elapsed < self.orbit_s:
            command = (0.0, 0.0, self.orbit_rate_radps)
        elif elapsed < 2.0 * self.orbit_s:
            command = (0.0, 0.0, -self.orbit_rate_radps)
        else:
            command = (0.0, 0.0, 0.0)
        apply_action(ctx.act, np.tile(np.asarray(command, dtype=np.float32), (ctx.num_envs, 1)))
        self._ticks += 1

        if elapsed < 2.0 * self.orbit_s:
            return Status.RUNNING
        return Status.SUCCESS if self._max_frame_delta >= self.min_frame_delta else Status.FAILURE

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(elapsed_s=self._ticks * ctx.dt, max_frame_delta=self._max_frame_delta)
