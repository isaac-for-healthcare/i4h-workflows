# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.types import Pose
from i4h_engine.status import Status
from i4h_engine.task import TickContext
from i4h_tasks.basic.motion.g1_locomotion import G1_BASE_HEIGHT_INDEX, G1_NAVIGATION_SLICE
from i4h_tasks.basic.motion.g1_walk_to_object import (
    G1WalkToObject,
    collision_history_key,
    edge_distance_xy,
    upright_tilt_deg,
)
from i4h_tasks.basic.testing.fake_scene import FakeActuation, FakeScene

TABLE_HALF_EXTENTS = (0.64, 0.4)
YAW_90_WXYZ = np.array([[0.7071068, 0.0, 0.0, 0.7071068]], dtype=np.float32)


def make_context(
    *,
    robot_y: float = -2.4,
    velocity: float = 0.0,
    angular_velocity: float = 0.0,
) -> TickContext:
    scene = FakeScene(
        dof=43,
        objects={"table": Pose.from_xyz(0.0, 0.0, 0.2377)},
        footprint_half_extents={"table": TABLE_HALF_EXTENTS},
        robot_pose=Pose(pos=np.array([[0.0, robot_y, 0.7923]]), quat=YAW_90_WXYZ),
        robot_velocity=np.array([[0.0, velocity, 0.0]]),
        robot_angular_velocity=np.array([[0.0, 0.0, angular_velocity]]),
    )
    return TickContext(scene=scene, act=FakeActuation(dof=50), dt=0.1)


def make_task(**kwargs) -> G1WalkToObject:
    return G1WalkToObject(object="table", stable_s=0.2, **kwargs)


def test_edge_distance_uses_nearest_rectangle_edge() -> None:
    distance = edge_distance_xy(
        np.array([[0.0, -2.4, 0.8]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.2]], dtype=np.float32),
        TABLE_HALF_EXTENTS,
    )
    assert distance == pytest.approx([2.0])


def test_upright_check_ignores_yaw() -> None:
    assert upright_tilt_deg(YAW_90_WXYZ) == pytest.approx([0.0], abs=1e-3)


def test_task_commands_forward_wbc_velocity() -> None:
    ctx = make_context()
    task = make_task()
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    action = ctx.act.raw_actions["robot"]
    assert action.shape == (1, 50)
    assert action[0, G1_NAVIGATION_SLICE][0] > 0.0
    np.testing.assert_allclose(action[0, G1_NAVIGATION_SLICE][1:], 0.0)
    assert action[0, G1_BASE_HEIGHT_INDEX] == pytest.approx(0.75)


def test_task_keeps_approaching_until_inside_stopping_margin() -> None:
    ctx = make_context(robot_y=-1.0)
    task = make_task(success_distance_m=0.3, approach_gain=0.5)
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    action = ctx.act.raw_actions["robot"]
    assert action[0, G1_NAVIGATION_SLICE][0] == pytest.approx(0.5 * 0.6, abs=1e-6)


def test_task_latches_stopping_after_crossing_margin() -> None:
    ctx = make_context(robot_y=-0.88)
    task = make_task(success_distance_m=0.3)
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    assert ctx.act.raw_actions["robot"][0, G1_NAVIGATION_SLICE][0] > 0.0
    ctx.scene.robot_pose.pos[0, 1] = -0.63
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(ctx.act.raw_actions["robot"][0, G1_NAVIGATION_SLICE], 0.0)
    ctx.scene.robot_pose.pos[0, 1] = -0.88
    assert task.tick(ctx) is Status.RUNNING
    np.testing.assert_allclose(ctx.act.raw_actions["robot"][0, G1_NAVIGATION_SLICE], 0.0)


def test_task_stops_and_succeeds_inside_distance_band() -> None:
    ctx = make_context(robot_y=-0.64)
    task = make_task()
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.SUCCESS
    np.testing.assert_allclose(ctx.act.raw_actions["robot"][0, G1_NAVIGATION_SLICE], 0.0)


def test_task_latches_table_collision_as_failure() -> None:
    ctx = make_context()
    ctx.scene.contacts.add(("robot", "table"))
    task = make_task()
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.FAILURE
    assert ctx.blackboard[collision_history_key("robot", "table")].all()


def test_task_does_not_succeed_while_robot_is_rotating() -> None:
    ctx = make_context(robot_y=-0.64, angular_velocity=0.2)
    task = make_task()
    task.on_enter(ctx, None)

    assert task.tick(ctx) is Status.RUNNING
    assert not task.on_exit(ctx).stopped


def test_task_requires_scene_owned_footprint() -> None:
    ctx = make_context()
    ctx.scene._footprint_half_extents.clear()
    task = make_task()
    task.on_enter(ctx, None)

    with pytest.raises(KeyError, match="collision footprint"):
        task.tick(ctx)
