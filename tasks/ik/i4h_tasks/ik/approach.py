# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move to a standoff pose above the target rather than onto it.

Separating approach from contact is what makes a grasp retryable: on
failure the workflow re-enters from a known standoff."""

from __future__ import annotations

import numpy as np

from i4h_common.types import Pose, quat_mul, quat_rotate
from i4h_tasks.ik.move_to_pose import MoveToPose


class Approach(MoveToPose):
    """Move to a standoff pose above the target rather than onto it.

    Separating approach from contact is what makes a grasp retryable: if the
    grasp fails you re-run it from a known standoff, not from wherever the
    gripper ended up.
    """

    requires = {"action_space": "ee_pose"}

    def __init__(
        self,
        *,
        standoff: tuple[float, float, float] = (0.0, 0.0, 0.10),
        local_standoff: bool = False,
        orientation: tuple[float, float, float, float] | None = None,
        local_orientation: bool = False,
        duration_s: float = 1.0,
        name: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(duration_s=duration_s, name=name, **kwargs)  # type: ignore[arg-type]
        self.standoff = np.asarray(standoff, dtype=np.float32)
        self.local_standoff = local_standoff
        self.orientation = None if orientation is None else np.asarray(orientation, dtype=np.float32)
        self.local_orientation = local_orientation

    def _resolve_goal(self, target: Pose) -> Pose:
        offset = np.broadcast_to(self.standoff, target.pos.shape)
        if self.local_standoff:
            offset = quat_rotate(target.quat, offset)
        quat = target.quat
        if self.orientation is not None:
            orientation = np.broadcast_to(self.orientation, target.quat.shape).astype(np.float32, copy=True)
            quat = quat_mul(target.quat, orientation) if self.local_orientation else orientation
        return Pose(pos=target.pos + offset, quat=quat)
