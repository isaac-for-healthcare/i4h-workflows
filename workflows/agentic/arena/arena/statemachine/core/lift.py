# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable pick-and-lift behavior for the surgical lift envs.

Approach above the object, descend, close the gripper, settle, then lift straight
up from the grasp point. Success is an explicit predicate -- the object rising a
set distance above where it started -- since the lift task itself only terminates
on timeout or a dropped object.

The lift servos to a *frozen grasp anchor* + a fixed rise rather than the task's
``object_pose`` command: that command only implies a ~5 mm rise and resamples
mid-episode, which would drag a held object back down.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from arena.statemachine.core.common import frame_pose_b, object_pose_for_lift, step_dt, zero_pose_action
from arena.statemachine.core.machine import Stage, StateMachine

_LIFT_SUCCESS_RISE_M = 0.03  # object must clear this rise above its reset height to count as lifted
_LIFT_HEIGHT_M = 0.08  # commanded tool-tip rise above the grasp pose (clears the success threshold with margin)


class LiftStateMachine(StateMachine):
    """Grasp the ``object`` and lift it straight up from a frozen grasp anchor.

    (See the module docstring for why we lift by a fixed rise instead of tracking
    the resampling ``object_pose`` command.)
    """

    default_max_steps = 250
    hold_last = True

    def initial_action(self, env: Any) -> torch.Tensor:
        return zero_pose_action(env)

    def on_reset(self, env: Any) -> None:
        self._start_z = self._object_z(env).clone()
        self._lift_anchor: tuple[torch.Tensor, torch.Tensor] | None = None

    def build_stages(self, env: Any) -> Sequence[Stage]:
        dt = step_dt(env)

        def steps(seconds: float) -> int:
            return max(1, round(seconds / dt))

        return [
            Stage("rest", self._hold_open, steps(0.5)),
            Stage("above_object", self._above_object, steps(0.7)),
            Stage("approach_object", self._approach_object, steps(0.7)),
            Stage("grasp_object", self._grasp_object, steps(0.5)),
            Stage("grasp_settle", self._grasp_object, steps(0.4)),  # hold the grasp so the position-driven jaws seat
            Stage("lift_object", self._lift, steps(2.0)),
        ]

    def succeeded(self, env: Any) -> torch.Tensor:
        return self._object_z(env) > (self._start_z + _LIFT_SUCCESS_RISE_M)

    def log_fields(self, env: Any) -> dict[str, Any]:
        return {"rise_m": f"{float((self._object_z(env) - self._start_z).max()):.3f}"}

    def _object_z(self, env: Any) -> torch.Tensor:
        return env.unwrapped.scene["object"].data.root_pos_w[:, 2]

    def _hold_open(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        pos_b, quat_w = frame_pose_b(env, "robot", "ee_frame")
        action = zero_pose_action(env)
        action[:, :3] = pos_b
        action[:, 3:7] = quat_w
        action[:, 7] = 1.0
        return action

    def _above_object(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        pos_b, quat_w = object_pose_for_lift(env)
        action = zero_pose_action(env)
        action[:, :3] = pos_b
        action[:, 2] += 0.05
        action[:, 3:7] = quat_w
        action[:, 7] = 1.0
        return action

    def _approach_object(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        pos_b, quat_w = object_pose_for_lift(env)
        action = zero_pose_action(env)
        action[:, :3] = pos_b
        action[:, 3:7] = quat_w
        action[:, 7] = 1.0
        return action

    def _grasp_object(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        pos_b, quat_w = object_pose_for_lift(env)
        action = zero_pose_action(env)
        action[:, :3] = pos_b
        action[:, 3:7] = quat_w
        action[:, 7] = -1.0
        return action

    def _lift(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        # Snapshot the tool-tip pose where the grasp settled, then servo straight up from it.
        if self._lift_anchor is None:
            self._lift_anchor = frame_pose_b(env, "robot", "ee_frame")
        pos_b, quat_w = self._lift_anchor
        action = zero_pose_action(env)
        action[:, :3] = pos_b
        action[:, 2] += _LIFT_HEIGHT_M
        action[:, 3:7] = quat_w
        action[:, 7] = -1.0
        return action
