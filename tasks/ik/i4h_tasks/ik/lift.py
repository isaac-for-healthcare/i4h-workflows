# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raise the end effector vertically from wherever it currently is."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i4h_common.types import Pose
from i4h_engine.task import TickContext
from i4h_tasks.ik.move_to_pose import MoveToPose


class Lift(MoveToPose):
    """Raise the end effector vertically from wherever it currently is."""

    requires = {"action_space": "ee_pose"}

    @dataclass
    class Inputs:
        from_pose: Pose | None = None

    def __init__(
        self,
        *,
        height: float = 0.15,
        duration_s: float = 1.0,
        name: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(duration_s=duration_s, name=name, **kwargs)  # type: ignore[arg-type]
        self.height = height

    def on_enter(self, ctx: TickContext, inputs: Inputs) -> None:
        origin = getattr(inputs, "from_pose", None) or ctx.scene.tcp(self.robot)
        # Reuse the parent's machinery by handing it an already-resolved target.
        super().on_enter(
            ctx,
            MoveToPose.Inputs(
                target=Pose(pos=origin.pos + np.array([0.0, 0.0, self.height], np.float32), quat=origin.quat)
            ),
        )
