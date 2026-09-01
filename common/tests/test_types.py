# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from i4h_common.types import JointState, ObjectState, Pose, as_batch, quat_mul, quat_rotate


def test_pose_always_batched():
    pose = Pose(pos=[1.0, 2.0, 3.0], quat=[1.0, 0.0, 0.0, 0.0])
    assert pose.pos.shape == (1, 3)
    assert pose.quat.shape == (1, 4)
    assert pose.num_envs == 1


def test_pose_rejects_wrong_width():
    with pytest.raises(ValueError, match="must be"):
        Pose(pos=[1.0, 2.0], quat=[1.0, 0.0, 0.0, 0.0])


def test_pose_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="batch mismatch"):
        Pose(pos=np.zeros((2, 3)), quat=np.zeros((3, 4)))


def test_pose_identity_and_translate():
    pose = Pose.identity(num_envs=4)
    assert pose.num_envs == 4
    assert np.allclose(pose.quat[:, 0], 1.0)
    moved = pose.translated([0.0, 0.0, 0.1])
    assert np.allclose(moved.pos[:, 2], 0.1)
    assert np.allclose(moved.quat, pose.quat)


def test_pose_distance():
    a = Pose.from_xyz(0.0, 0.0, 0.0, num_envs=2)
    b = Pose.from_xyz(3.0, 4.0, 0.0, num_envs=2)
    assert np.allclose(a.distance_to(b), 5.0)


def test_as_batch_broadcast_and_validate():
    assert as_batch(0.5, 3, 2).shape == (3, 2)
    assert np.allclose(as_batch([1.0, 2.0], 3, 2)[0], [1.0, 2.0])
    with pytest.raises(ValueError):
        as_batch([1.0, 2.0, 3.0], 3, 2)


def test_as_batch_broadcast_is_writable():
    # np.broadcast_to returns a read-only view; tasks mutate action buffers in place.
    batch = as_batch([1.0, 2.0], 3, 2)
    batch[0, 0] = 9.0
    assert batch[0, 0] == 9.0
    assert batch[1, 0] == 1.0


def test_joint_state_index_of():
    state = JointState(pos=np.zeros((1, 3)), vel=np.zeros((1, 3)), names=("a", "b", "c"))
    assert state.dof == 3
    assert state.index_of("b") == 1
    with pytest.raises(KeyError):
        state.index_of("z")


def test_object_settled():
    slow = ObjectState("x", Pose.identity(), lin_vel=[0.0, 0.0, 0.001])
    fast = ObjectState("x", Pose.identity(), lin_vel=[0.0, 0.0, 1.0])
    assert bool(slow.is_settled[0])
    assert not bool(fast.is_settled[0])


def test_object_state_defaults_angular_velocity_to_zero():
    state = ObjectState("x", Pose.identity(), lin_vel=[0.0, 0.0, 0.0])

    np.testing.assert_array_equal(state.ang_vel, np.zeros((1, 3), dtype=np.float32))


def test_quat_identity_roundtrip():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.allclose(quat_rotate(identity, vec), vec)
    assert np.allclose(quat_mul(identity, identity), identity)


def test_quat_rotate_90_deg_about_z():
    half = np.sqrt(0.5)
    quat = np.array([[half, 0.0, 0.0, half]], dtype=np.float32)
    rotated = quat_rotate(quat, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-6)


# -- satisfied -----------------------------------------------------------


def test_satisfied_passes_a_plain_bool_through():
    from i4h_common.types import satisfied

    assert satisfied(True) is True
    assert satisfied(False) is False


def test_satisfied_any_vs_all_differ_on_a_mixed_mask():
    """The quantifier is the whole reason this is a parameter.

    Three copies of this helper existed with `any`/`all` baked in and silently
    different, so a workflow's `until=` and WaitUntil disagreed about the same mask.
    """
    from i4h_common.types import satisfied

    mixed = np.array([False, True])
    assert satisfied(mixed, across="any") is True
    assert satisfied(mixed, across="all") is False


def test_satisfied_defaults_to_any():
    from i4h_common.types import satisfied

    assert satisfied(np.array([False, True])) is True


def test_satisfied_on_an_empty_mask_is_false():
    from i4h_common.types import satisfied

    assert satisfied(np.array([], dtype=bool)) is False
    assert satisfied(np.array([], dtype=bool), across="all") is False
