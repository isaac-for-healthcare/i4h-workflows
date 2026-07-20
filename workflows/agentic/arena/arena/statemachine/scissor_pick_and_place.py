# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted SO-ARM scissor pick-and-place state machine.

The SO-ARM exposes only joint-position control (no IK), so the path is a sequence
of joint-space keyframes (deltas from the home pose) smoothstepped between. The
one closed-loop adaptation is a shoulder-pan offset proportional to the
randomized scissors Y, read from the live scene at reset. Success reuses the
task's own ``success_scissors_placed`` (settled in tray + arm back home).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from arena.arena_config import get_arena_config
from arena.statemachine.core.common import step_dt
from arena.statemachine.core.machine import Stage, StateMachine
from arena.tasks.scissor_pick_and_place import success_scissors_in_tray, success_scissors_placed

ENV_ID = "scissor_pick_and_place"
_OPEN = 0.35
_CLOSED = -0.16

# Shoulder-pan correction for the randomized scissors Y (applied to grasp/lift stages).
_REF_SCISSORS_Y = -0.023
_PAN_GAIN = 10.0
_MAX_PAN = 0.12
_PAN_STAGES = frozenset(
    {
        "align_left",
        "align_scissors",
        "descend_high",
        "descend_low",
        "open_pregrasp",
        "open_on_scissors",
        "close_on_scissors",
        "close_settle",
        "seat_grip",
        "lift_scissors",
    }
)

_Target = tuple[float, float, float, float, float, float]
_HOME: _Target = (0.0, 0.0, 0.0, 0.0, 0.0, _OPEN)

# (name, joint-delta-from-home 6-vector, duration_s); index 5 is the jaw (+open / -close).
_KEYFRAMES: tuple[tuple[str, _Target, float], ...] = (
    ("settle_home", _HOME, 0.25),
    ("leave_home", (0.03, 0.34, -0.19, -0.33, 0.25, -0.08), 0.35),
    ("clear_front", (0.09, 0.37, -0.20, -0.48, 0.42, -0.12), 0.33),
    ("orient_front", (0.16, 0.25, -0.14, -0.71, 0.51, -0.12), 0.33),
    ("orient_scissors", (0.25, 0.06, 0.03, -0.97, 0.50, -0.15), 0.33),
    ("align_left", (-0.14, -0.03, 0.11, -1.08, 0.51, -0.15), 0.33),
    ("align_scissors", (-0.32, -0.05, 0.13, -1.08, 0.52, _CLOSED), 0.33),
    ("descend_high", (-0.37, 0.59, -0.44, -1.08, 0.16, _CLOSED), 0.33),
    ("descend_low", (-0.37, 1.23, -0.92, -1.07, -0.14, -0.07), 0.33),
    ("open_pregrasp", (-0.41, 1.75, -1.14, -1.07, -0.16, 0.23), 0.33),
    ("open_on_scissors", (-0.48, 1.92, -1.05, -1.00, -0.15, 0.25), 0.33),
    ("close_on_scissors", (-0.46, 1.93, -0.94, -0.95, -0.14, -0.15), 0.33),
    ("close_settle", (-0.43, 1.90, -0.92, -1.05, -0.16, _CLOSED), 0.33),
    ("seat_grip", (-0.27, 1.68, -1.08, -0.51, -0.10, _CLOSED), 0.33),
    ("lift_scissors", (-0.01, 1.36, -1.10, -0.42, -0.13, _CLOSED), 0.33),
    ("carry_mid", (0.50, 1.16, -1.18, -0.41, 0.12, -0.15), 0.33),
    ("carry_to_tray", (0.69, 1.33, -1.16, -0.72, 0.44, _CLOSED), 0.33),
    ("lower_over_tray", (0.58, 1.70, -1.24, -1.15, 0.62, _CLOSED), 0.33),
    ("release_in_tray", (0.52, 1.89, -1.14, -1.21, 0.65, 0.16), 0.33),
    ("release_settle", (0.50, 1.71, -1.14, -1.23, 0.66, 0.29), 0.33),
    ("withdraw_from_tray", (0.50, 1.33, -1.21, -1.24, 0.61, 0.29), 0.33),
    ("return_high", (0.47, 0.61, -0.52, -1.22, 0.32, 0.29), 0.33),
    ("return_mid", (0.38, -0.07, 0.00, -1.22, 0.30, 0.29), 0.33),
    ("return_near_home", (0.22, -0.15, 0.14, -0.69, 0.56, 0.29), 0.65),
    ("home", _HOME, 1.00),
)


class ScissorPickAndPlace(StateMachine):
    env_id = ENV_ID
    default_max_steps = get_arena_config(ENV_ID).max_timesteps or 600

    def __init__(self) -> None:
        self._pan: torch.Tensor | float = 0.0

    def initial_action(self, env: Any) -> torch.Tensor:
        return self._target(env, _HOME, "settle_home")

    def on_reset(self, env: Any) -> None:
        # Per-env shoulder-pan offset derived from each env's randomized scissors Y.
        scissors_y = env.unwrapped.scene["scissors"].data.root_pos_w[:, 1]
        self._pan = ((scissors_y - _REF_SCISSORS_Y) * _PAN_GAIN).clamp(0.0, _MAX_PAN)

    def build_stages(self, env: Any) -> Sequence[Stage]:
        dt = step_dt(env)
        return [self._keyframe_stage(name, keyframe, duration, dt) for name, keyframe, duration in _KEYFRAMES]

    def succeeded(self, env: Any) -> torch.Tensor:
        return success_scissors_placed(env.unwrapped)

    def log_fields(self, env: Any) -> dict[str, Any]:
        return {"in_tray": bool(success_scissors_in_tray(env.unwrapped).any().item())}

    def _target(self, env: Any, keyframe: _Target, name: str) -> torch.Tensor:
        action = torch.tensor(keyframe, dtype=torch.float32, device=env.device).reshape(1, -1).repeat(env.num_envs, 1)
        if name in _PAN_STAGES:
            action[:, 0] += self._pan
        return action

    def _keyframe_stage(self, name: str, keyframe: _Target, duration: float, dt: float) -> Stage:
        steps = max(1, round(duration / dt))

        def action(env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
            alpha = (i + 1) / steps
            return torch.lerp(start, self._target(env, keyframe, name), alpha * alpha * (3.0 - 2.0 * alpha))

        return Stage(name, action, steps)


def run_state_machine(*, args: Any, env: Any, app: Any, controller: Any) -> None:
    ScissorPickAndPlace().run(args=args, env=env, app=app, controller=controller)
