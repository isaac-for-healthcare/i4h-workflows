# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import numpy as np
import pytest
from i4h_engine.status import Status
from i4h_tasks.basic.medical.patient_ultrasound_scan import PatientUltrasoundScan


@pytest.mark.parametrize("sensor_follows_probe", [True, False])
def test_validation_requires_contact_lift_and_recovery(sensor_follows_probe):
    task = PatientUltrasoundScan()
    target = np.array([0.55, -0.05, 0.28, 0.0, 1.0, 0.0, 0.0])

    class Scene:
        def sensor_signal(self, name, output):
            if output == "target_pose":
                return target
            return np.array([0.04 if 40 <= task.ticks < 120 else 0.0])

        def camera(self, name):
            lifted = sensor_follows_probe and 40 <= task.ticks < 120
            return SimpleNamespace(to_array=lambda: np.full((8, 8, 3), 0 if lifted else 30, dtype=np.uint8))

    commands = []
    ctx = SimpleNamespace(scene=Scene(), act=SimpleNamespace(set_ee_target=lambda pose: commands.append(pose)))
    task.on_enter(ctx, object())
    for _ in range(220):
        status = task.tick(ctx)
    assert status is (Status.SUCCESS if sensor_follows_probe else Status.FAILURE)
    assert commands[80].pos[0, 2] - commands[0].pos[0, 2] == pytest.approx(0.04)
    np.testing.assert_allclose(commands[-1].pos[0], target[:3])
