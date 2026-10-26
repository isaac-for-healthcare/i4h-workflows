# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted commands for the endoluminal catheter scenes' four velocity channels.

The action is ``[insertion (m/s), axial rotation (rad/s), tip bend (rad/s), C-arm
orbit (rad/s)]``. ``CatheterSweepPolicy`` exercises every channel once, as a
simulation smoke demonstration; ``FluoroscopyCArmSweepPolicy`` orbits the C-arm out
and back and reports whether the fluoroscopy image followed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


class _Timeline:
    """Per-env elapsed control steps, restarted by reset."""

    _ticks: torch.Tensor | None = None

    def _elapsed(self, env) -> torch.Tensor:
        if self._ticks is None:
            self._ticks = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Double precision, as the task graph's Python floats: in float32, 111 steps of 1/30 s
        # land on a segment boundary that float64 places just before it.
        elapsed = self._ticks.double() * env.step_dt
        self._ticks += 1
        return elapsed

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._ticks is not None:
            self._ticks[slice(None) if env_ids is None else env_ids] = 0


def _command_at(elapsed: torch.Tensor, schedule: list[tuple[float, tuple[float, float, float, float]]]) -> torch.Tensor:
    """The command of the schedule segment each env's elapsed time falls in; zero after the last."""
    command = torch.zeros(elapsed.shape[0], 4, device=elapsed.device, dtype=torch.float32)
    start = 0.0
    for duration, values in schedule:
        within = (elapsed >= start) & (elapsed < start + duration)
        command[within] = torch.tensor(values, device=elapsed.device)
        start += duration
    return command


@dataclass
class CatheterSweepPolicyCfg(PolicyCfg):
    insertion_speed_mps: float = 0.012
    rotation_rate_radps: float = 0.8
    bend_rate_radps: float = 0.5
    orbit_rate_radps: float = 0.45
    settle_s: float = 1.0
    advance_s: float = 0.6
    hold_s: float = 0.5
    rotate_s: float = 0.6
    bend_s: float = 0.5
    retract_s: float = 0.3
    orbit_s: float = 1.2


@register_policy
class CatheterSweepPolicy(_Timeline, PolicyBase[CatheterSweepPolicyCfg]):
    """Settle, advance, hold, rotate, bend and unbend the tip, retract, then orbit the C-arm out and back."""

    name = "i4h_catheter_sweep"

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        cfg = self.config
        speed, rotation, bend, orbit = (
            cfg.insertion_speed_mps,
            cfg.rotation_rate_radps,
            cfg.bend_rate_radps,
            cfg.orbit_rate_radps,
        )
        schedule = [
            (cfg.settle_s, (0.0, 0.0, 0.0, 0.0)),
            (cfg.advance_s, (speed, 0.0, 0.0, 0.0)),
            (cfg.hold_s, (0.0, 0.0, 0.0, 0.0)),
            (cfg.rotate_s, (0.0, rotation, 0.0, 0.0)),
            (cfg.bend_s, (0.0, 0.0, bend, 0.0)),
            (cfg.bend_s, (0.0, 0.0, -bend, 0.0)),
            (cfg.retract_s, (-speed, 0.0, 0.0, 0.0)),
            (cfg.orbit_s, (0.0, 0.0, 0.0, orbit)),
            (cfg.orbit_s, (0.0, 0.0, 0.0, -orbit)),
        ]
        env = env.unwrapped
        return _command_at(self._elapsed(env), schedule)


@dataclass
class FluoroscopyCArmSweepPolicyCfg(PolicyCfg):
    orbit_s: float = 1.2
    orbit_rate_radps: float = 0.45
    min_frame_delta: float = 2.0
    """Mean absolute pixel change, against the first frame, that counts as the image following the C-arm."""


@register_policy
class FluoroscopyCArmSweepPolicy(_Timeline, PolicyBase[FluoroscopyCArmSweepPolicyCfg]):
    """Orbit the C-arm out and back, and report whether the fluoroscopy image changed with it.

    The verdict is printed once the sweep is back, or when the episode ends sooner:
    ``PASS`` once the largest mean absolute change from the episode's first frame
    reaches ``min_frame_delta``. Arena's runner stops at its episode limit without
    resetting the policy, so the verdict cannot wait for the reset.
    """

    name = "i4h_fluoroscopy_carm_sweep"

    def __init__(self, config: FluoroscopyCArmSweepPolicyCfg):
        super().__init__(config)
        self._baseline: np.ndarray | None = None
        self._reported = False
        self.max_frame_delta = 0.0
        self.verdicts: list[bool] = []

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        image = env.scene["fluoroscopy"].data.output["rgb"][0].detach().cpu().numpy().astype(np.int16)
        if self._baseline is None:
            self._baseline = image.copy()
        else:
            self.max_frame_delta = max(self.max_frame_delta, float(np.abs(image - self._baseline).mean()))
        orbit = self.config.orbit_rate_radps
        schedule = [(self.config.orbit_s, (0.0, 0.0, 0.0, orbit)), (self.config.orbit_s, (0.0, 0.0, 0.0, -orbit))]
        elapsed = self._elapsed(env)
        if float(elapsed[0]) >= 2 * self.config.orbit_s:
            self._report()
        return _command_at(elapsed, schedule)

    def _report(self) -> None:
        if self._baseline is None or self._reported:
            return
        self._reported = True
        passed = self.max_frame_delta >= self.config.min_frame_delta
        self.verdicts.append(passed)
        print(
            f"[fluoroscopy] C-arm sweep {'PASS' if passed else 'FAIL'}: largest mean frame change "
            f"{self.max_frame_delta:.2f} (threshold {self.config.min_frame_delta})",
            flush=True,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._report()
        self._baseline = None
        self._reported = False
        self.max_frame_delta = 0.0
        super().reset(env_ids)
