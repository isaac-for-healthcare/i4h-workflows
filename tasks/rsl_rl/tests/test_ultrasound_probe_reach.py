# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from i4h_common.types import Pose
from i4h_engine.task import TickContext
from i4h_tasks.basic.testing.fake_scene import FakeActuation, FakeScene
from i4h_tasks.rsl_rl.ultrasound_probe_reach import policy_observation


def test_policy_observation_matches_training_layout() -> None:
    target = Pose(
        pos=np.array([[0.60, -0.075, 0.196]], dtype=np.float32),
        quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    scene = FakeScene(
        dof=7,
        objects={"target": target},
        robot_pose=Pose.from_xyz(0.0, 0.0, 0.0),
    )
    scene.joint_pos[:] = np.arange(7, dtype=np.float32)
    scene.joint_vel[:] = np.arange(7, dtype=np.float32) * 0.1
    scene.tcp_pose = Pose(
        pos=np.array([[0.32, -0.01, 0.30]], dtype=np.float32),
        quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    ctx = TickContext(scene=scene, act=FakeActuation(dof=6, action_space="ee_pose"))
    previous = np.full((1, 6), 0.25, dtype=np.float32)

    observation = policy_observation(ctx, previous)

    assert observation.shape == (1, 34)
    np.testing.assert_allclose(observation[:, :7], scene.joint_pos)
    np.testing.assert_allclose(observation[:, 7:14], scene.joint_vel)
    np.testing.assert_allclose(observation[:, 14:20], previous)
    np.testing.assert_allclose(observation[:, 20:23], scene.tcp_pose.pos)
    np.testing.assert_allclose(observation[:, 23:27], [[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(observation[:, 27:30], target.pos)
    np.testing.assert_allclose(observation[:, 30:34], [[0.0, 0.0, 0.0, 1.0]])
