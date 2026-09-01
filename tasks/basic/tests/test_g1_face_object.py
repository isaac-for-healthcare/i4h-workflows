# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.types import Pose
from i4h_engine.status import Status
from i4h_engine.task import TickContext
from i4h_tasks.basic.motion.g1_face_object import G1FaceObject
from i4h_tasks.basic.motion.g1_locomotion import G1_NAVIGATION_SLICE, heading_error_to_position
from i4h_tasks.basic.testing.fake_scene import FakeActuation, FakeScene

YAW_90_WXYZ = np.array([[0.7071068, 0.0, 0.0, 0.7071068]], dtype=np.float32)


def make_context(quat: np.ndarray) -> TickContext:
    scene = FakeScene(
        dof=43,
        objects={"target": Pose.from_xyz(0.0, 2.0, 0.0)},
        robot_pose=Pose(pos=np.array([[0.0, 0.0, 0.8]], dtype=np.float32), quat=quat),
    )
    return TickContext(scene=scene, act=FakeActuation(dof=50), dt=0.1)


def test_heading_error_targets_object_in_world_xy() -> None:
    error = heading_error_to_position(
        np.array([[0.0, 0.0, 0.8]], dtype=np.float32),
        Pose.identity().quat,
        np.array([[0.0, 2.0, 0.0]], dtype=np.float32),
    )
    assert error == pytest.approx([np.pi / 2.0])


def test_face_object_commands_rotation_without_translation() -> None:
    ctx = make_context(Pose.identity().quat)
    task = G1FaceObject(object="target", stable_s=0.2)
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    navigation = ctx.act.raw_actions["robot"][0, G1_NAVIGATION_SLICE]
    np.testing.assert_allclose(navigation[:2], 0.0)
    assert navigation[2] > 0.0


def test_face_object_stops_after_stable_alignment() -> None:
    ctx = make_context(YAW_90_WXYZ)
    task = G1FaceObject(object="target", stable_s=0.2)
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.SUCCESS
    np.testing.assert_allclose(ctx.act.raw_actions["robot"][0, G1_NAVIGATION_SLICE], 0.0)
