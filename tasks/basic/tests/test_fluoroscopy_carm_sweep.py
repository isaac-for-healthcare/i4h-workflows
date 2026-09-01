# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_common.types import CameraFrame
from i4h_engine.status import Status
from i4h_tasks.basic.medical.fluoroscopy_carm_sweep import FluoroscopyCArmSweep


class _Actuation:
    action_space = "catheter_carm_velocity"
    dof = 3

    def __init__(self) -> None:
        self.value = np.zeros((1, 3), dtype=np.float32)

    def set_raw_action(self, action: np.ndarray, robot: str = "robot") -> None:
        del robot
        self.value = np.asarray(action, dtype=np.float32)

    def hold(self, robot: str = "robot") -> None:
        del robot


class _Scene:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def camera(self, name: str) -> CameraFrame:
        assert name == "fluoroscopy"
        self._last = next(self._values, self._last)
        image = np.full((4, 4, 3), self._last, dtype=np.uint8)
        return CameraFrame(name=name, width=4, height=4, data=image.tobytes())


def test_carm_sweep_requires_a_changed_fluoroscopy_frame() -> None:
    task = FluoroscopyCArmSweep(orbit_s=0.1, min_frame_delta=2.0)
    ctx = SimpleNamespace(scene=_Scene([10, 10, 20, 20]), act=_Actuation(), dt=0.1, num_envs=1)
    task.on_enter(ctx, object())

    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(ctx.act.value, [[0.0, 0.0, 0.45]])
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(ctx.act.value, [[0.0, 0.0, -0.45]])
    assert task.tick(ctx) is Status.SUCCESS
    assert task.on_exit(ctx).max_frame_delta == 10.0


def test_carm_sweep_fails_when_the_sensor_is_static() -> None:
    task = FluoroscopyCArmSweep(orbit_s=0.1, min_frame_delta=2.0)
    ctx = SimpleNamespace(scene=_Scene([10]), act=_Actuation(), dt=0.1, num_envs=1)
    task.on_enter(ctx, object())

    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.FAILURE
