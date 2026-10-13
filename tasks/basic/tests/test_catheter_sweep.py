# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_engine.status import Status
from i4h_tasks.basic.medical.catheter_sweep import CatheterSweep


class _Actuation:
    def __init__(self) -> None:
        self.value = np.zeros((1, 4), dtype=np.float32)

    @property
    def action_space(self) -> str:
        return "catheter_carm_velocity"

    @property
    def dof(self) -> int:
        return 4

    def set_raw_action(self, action: np.ndarray, robot: str = "robot") -> None:
        del robot
        self.value = np.asarray(action, dtype=np.float32)

    def hold(self, robot: str = "robot") -> None:
        del robot


def test_sweep_emits_catheter_commands_and_carm_orbit() -> None:
    task = CatheterSweep(settle_s=0, hold_s=0, bend_s=0, advance_s=0.1, rotate_s=0.1, retract_s=0.1, orbit_s=0.1)
    act = _Actuation()
    ctx = SimpleNamespace(act=act, dt=0.1, num_envs=1)
    task.on_enter(ctx, object())

    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(act.value, [[0.012, 0.0, 0.0, 0.0]])
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(act.value, [[0.0, 0.8, 0.0, 0.0]])
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(act.value, [[-0.012, 0.0, 0.0, 0.0]])
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(act.value, [[0.0, 0.0, 0.0, 0.45]])
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(act.value, [[0.0, 0.0, 0.0, -0.45]])
    assert task.tick(ctx) is Status.SUCCESS
    np.testing.assert_allclose(act.value, [[0.0, 0.0, 0.0, 0.0]])


def test_default_sweep_holds_still_stops_and_steers_before_retraction():
    task = CatheterSweep()
    act = _Actuation()
    ctx = SimpleNamespace(act=act, dt=1 / 30, num_envs=1)
    task.on_enter(ctx, object())
    actions = []
    for _ in range(300):
        status = task.tick(ctx)
        actions.append(act.value[0].copy())
        if status is Status.SUCCESS:
            break
    assert status is Status.SUCCESS
    actions = np.asarray(actions)
    np.testing.assert_array_equal(actions[:30], 0)
    forward = np.flatnonzero(actions[:, 0] > 0)
    backward = np.flatnonzero(actions[:, 0] < 0)
    bend = np.flatnonzero(actions[:, 2] > 0)
    unbend = np.flatnonzero(actions[:, 2] < 0)
    assert forward[-1] < bend[0] < unbend[0] < backward[0]
    np.testing.assert_array_equal(actions[forward[-1] + 1 : forward[-1] + 11], 0)
    assert abs(float(actions[:, 2].sum()) * ctx.dt) < 0.02
    np.testing.assert_array_equal(actions[-1], 0)
