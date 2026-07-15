# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted Franka ultrasound liver-scan state machine (relative-IK EE servo).

Servo the probe along an organ-local scan line -- approach above the phantom,
descend to contact, twist to the goal orientation, sweep, hold -- emitting a
clamped relative EE-pose delta each step. All targets are read from the live
(randomized) organ frame, so no world coordinates are baked in. Success reuses
the task's own ``ultrasound_scan_success``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from arena.arena_config import get_arena_config
from arena.statemachine.core.common import step_dt
from arena.statemachine.core.machine import Stage, StateMachine
from arena.tasks.ultrasound_liver_scan import reset_ultrasound_success_state, ultrasound_scan_success

ENV_ID = "ultrasound_liver_scan"
_MAX_POS_STEP = 0.025
_MAX_ROT_STEP = 0.20
_SETUP_POS = (0.3229, -0.0110, 0.3000)
_DOWN_EULER_DEG = (180.0, 0.0, 180.0)
_RESET_STEPS = 40

# Organ-local scan line (offsets transformed by the organ's yaw each step).
_SCAN_START = (0.0030, 0.0500, 0.2000)
_SCAN_CONTACT = (0.0030, 0.0500, 0.0983)
_SCAN_END = (-0.0650, -0.0715, 0.0857)


class UltrasoundLiverScan(StateMachine):
    env_id = ENV_ID
    default_max_steps = get_arena_config(ENV_ID).max_timesteps or 250
    # Hold the final scan pose to fill the budget so the scan-success step gate is always reachable.
    hold_last = True

    def initial_action(self, env: Any) -> torch.Tensor:
        return torch.zeros(env.num_envs, 6, dtype=torch.float32, device=env.device)

    def on_reset(self, env: Any) -> None:
        # Pre-roll: servo to the fixed setup pose (probe down), then reset the scan-success state.
        target = self._batch(env, _SETUP_POS, 3) + env.unwrapped.scene.env_origins
        quat = self._down_quat(env)
        for _ in range(_RESET_STEPS):
            env.step(self._servo(env, target, quat))
        reset_ultrasound_success_state(env.unwrapped, None)

    def build_stages(self, env: Any) -> Sequence[Stage]:
        dt = step_dt(env)

        def steps(seconds: float) -> int:
            return max(1, round(seconds / dt))

        return [
            Stage("approach_start", self._servo_to(_SCAN_START, self._down_quat), steps(0.60)),
            Stage("descend_to_contact", self._servo_to(_SCAN_CONTACT, self._down_quat), steps(0.60)),
            Stage("orient_for_scan", self._servo_to(_SCAN_CONTACT, self._goal_quat), steps(0.80)),
            Stage("scan_sweep", self._sweep(steps(2.0)), steps(2.0)),
            Stage("hold_scan_end", self._servo_to(_SCAN_END, self._goal_quat), steps(1.0)),
        ]

    def succeeded(self, env: Any) -> torch.Tensor:
        return ultrasound_scan_success(env.unwrapped)

    # -- action builders -------------------------------------------------
    def _servo_to(self, offset: Any, quat_fn):
        return lambda env, i, start: self._servo(env, self._organ_target(env, offset), quat_fn(env))

    def _sweep(self, total_steps: int):
        def action(env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
            alpha = (i + 1) / total_steps
            offset = torch.lerp(self._batch(env, _SCAN_CONTACT, 3), self._batch(env, _SCAN_END, 3), alpha)
            return self._servo(env, self._organ_target(env, offset), self._goal_quat(env))

        return action

    # -- kinematics helpers ----------------------------------------------
    def _batch(self, env: Any, values: Any, width: int) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values.to(device=env.device, dtype=torch.float32)
        else:
            tensor = torch.tensor(tuple(values), dtype=torch.float32, device=env.device)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        if tensor.shape[0] == 1 and env.num_envs > 1:
            tensor = tensor.repeat(env.num_envs, 1)
        return tensor[:, :width]

    def _ee_pose(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        frame = env.unwrapped.scene["ee_frame"].data
        return frame.target_pos_w[:, 0, :].clone(), frame.target_quat_w[:, 0, :].clone()

    def _down_quat(self, env: Any) -> torch.Tensor:
        from isaaclab.utils.math import quat_from_euler_xyz

        r, p, y = (torch.tensor(math.radians(v), device=env.device) for v in _DOWN_EULER_DEG)
        return self._batch(env, quat_from_euler_xyz(r, p, y).reshape(1, 4), 4)

    def _goal_quat(self, env: Any) -> torch.Tensor:
        return env.unwrapped.scene["goal_frame"].data.target_quat_w[:, 0, :].clone()

    def _organ_target(self, env: Any, offset: Any) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_yaw

        organs = env.unwrapped.scene["organs"].data
        return organs.root_pos_w + quat_apply_yaw(organs.root_quat_w, self._batch(env, offset, 3))

    def _servo(self, env: Any, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Clamped relative-IK action moving the EE toward the target pose."""
        from isaaclab.utils.math import compute_pose_error

        cur_pos, cur_quat = self._ee_pose(env)
        pos_err, rot_err = compute_pose_error(
            cur_pos, cur_quat, self._batch(env, target_pos, 3), target_quat, rot_error_type="axis_angle"
        )
        action = torch.cat([pos_err, rot_err], dim=-1)
        action[:, :3] = action[:, :3].clamp(-_MAX_POS_STEP, _MAX_POS_STEP)
        action[:, 3:] = action[:, 3:].clamp(-_MAX_ROT_STEP, _MAX_ROT_STEP)
        return action


def run_state_machine(*, args: Any, env: Any, app: Any, controller: Any) -> None:
    UltrasoundLiverScan().run(args=args, env=env, app=app, controller=controller)
