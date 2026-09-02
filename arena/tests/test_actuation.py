# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from i4h_arena.adapters.actuation import ArenaActuation, RobotSlice


def test_seed_does_not_alias_the_home_joint_reference() -> None:
    home = np.array([[0.0, -1.6, 1.4, 1.5, -1.8, 0.0]], dtype=np.float32)
    actuation = ArenaActuation(num_envs=1, action_dim=6)

    actuation.seed(home)
    actuation.set_joint_targets(np.ones((1, 6), dtype=np.float32))

    np.testing.assert_allclose(home, [[0.0, -1.6, 1.4, 1.5, -1.8, 0.0]])
    np.testing.assert_allclose(actuation.numpy(), np.ones((1, 6), dtype=np.float32))


def test_hold_repeats_the_previous_joint_position_target() -> None:
    target = np.arange(6, dtype=np.float32).reshape(1, 6)
    actuation = ArenaActuation(num_envs=1, action_dim=6)
    actuation.set_joint_targets(target)
    actuation.tensor()
    actuation.set_joint_targets(np.zeros((1, 6), dtype=np.float32))

    actuation.hold()

    np.testing.assert_allclose(actuation.numpy(), target)


def test_hold_zeros_a_relative_cartesian_delta() -> None:
    actuation = ArenaActuation(
        num_envs=1,
        action_dim=6,
        action_space="ee_pose",
        slices=(RobotSlice("robot", 0, 6, gripper_index=None),),
        relative_ee=True,
    )
    actuation.set_ee_delta(np.ones((1, 6), dtype=np.float32))
    actuation.tensor()

    actuation.hold()

    np.testing.assert_allclose(actuation.numpy(), np.zeros((1, 6), dtype=np.float32))
