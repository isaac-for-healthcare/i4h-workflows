# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cartesian nodes, exercised without Isaac."""

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.paths import workflow_root
from i4h_common.types import Pose
from i4h_common.world import UnsupportedActuation
from i4h_engine.discover import discover_tasks
from i4h_engine.status import Status
from i4h_engine.task import TickContext
from i4h_tasks.basic.testing.fake_scene import FakeActuation, FakeScene
from i4h_tasks.ik.approach import Approach
from i4h_tasks.ik.hold_pose import HoldPose
from i4h_tasks.ik.lift import Lift
from i4h_tasks.ik.move_to_pose import MoveToPose, slerp

DT = 1 / 60
SPECS = {
    task_id: spec.resolve() for task_id, spec in discover_tasks(workflow_root())[0].items() if spec.project == "ik"
}


@pytest.fixture
def ctx():
    scene = FakeScene(dof=7, objects={"needle": Pose.from_xyz(0.1, 0.0, 0.2)})
    return TickContext(scene=scene, act=FakeActuation(dof=7, action_space="ee_pose"), dt=DT)


def drive(task, ctx, inputs, *, limit: int = 2000) -> Status:
    task.on_enter(ctx, inputs)
    status = Status.RUNNING
    for _ in range(limit):
        status = task.tick(ctx)
        if status.is_terminal:
            break
    return status


# -- slerp ---------------------------------------------------------------


def test_slerp_endpoints():
    a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(slerp(a, b, 0.0), a)
    assert np.allclose(slerp(a, b, 1.0), b)


def test_slerp_output_is_unit():
    a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.7071, 0.7071, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(np.linalg.norm(slerp(a, b, 0.5), axis=-1), 1.0, atol=1e-6)


def test_slerp_takes_the_short_way_round():
    # q and -q are the same rotation; interpolating naively goes the long way.
    a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    b = -a
    assert np.allclose(slerp(a, b, 0.5), a, atol=1e-6)


# -- MoveToPose ----------------------------------------------------------


def test_move_to_pose_reaches_target(ctx):
    target = Pose.from_xyz(0.1, 0.0, 0.3)
    task = MoveToPose(duration_s=0.1, name="m")
    # The fake has no controller, so teleport the measured TCP to the goal to
    # model a controller that tracks perfectly.
    ctx.scene.tcp_pose = target
    assert drive(task, ctx, MoveToPose.Inputs(target=target)) is Status.SUCCESS
    assert task.on_exit(ctx).reached is True
    assert np.allclose(ctx.act.ee_targets["robot"].pos, target.pos, atol=1e-5)


def test_move_to_pose_fails_when_arm_never_arrives(ctx):
    # Commanded pose != achieved pose. This is the check that stops a workflow
    # continuing with the tool nowhere near where it believes it is.
    task = MoveToPose(duration_s=0.05, settle_timeout_s=0.1, name="m")
    assert drive(task, ctx, MoveToPose.Inputs(target=Pose.from_xyz(5.0, 5.0, 5.0))) is Status.FAILURE
    assert task.on_exit(ctx).reached is False


def test_hold_pose_commands_the_current_tcp_each_tick(ctx):
    task = HoldPose(seconds=ctx.dt * 2)
    task.on_enter(ctx, object())

    expected = ctx.scene.tcp()
    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.SUCCESS

    actual = ctx.act.ee_targets["robot"]
    assert np.array_equal(actual.pos, expected.pos)
    assert np.array_equal(actual.quat, expected.quat)


def test_move_to_pose_applies_offset(ctx):
    target = Pose.from_xyz(0.1, 0.0, 0.2)
    offset = Pose.from_xyz(0.0, 0.0, 0.05)
    task = MoveToPose(duration_s=0.05, settle_timeout_s=0.0, name="m")
    task.on_enter(ctx, MoveToPose.Inputs(target=target, offset=offset))
    for _ in range(10):
        if task.tick(ctx).is_terminal:
            break
    assert ctx.act.ee_targets["robot"].pos[0, 2] == pytest.approx(0.25, abs=1e-4)


def test_move_to_pose_without_target_is_an_error(ctx):
    with pytest.raises(ValueError, match="no target pose"):
        MoveToPose(name="m").on_enter(ctx, MoveToPose.Inputs(target=None))


def test_move_to_pose_commands_gripper_when_asked(ctx):
    target = Pose.from_xyz(0.0, 0.0, 0.0)
    ctx.scene.tcp_pose = target
    drive(MoveToPose(duration_s=0.05, gripper=0.4, name="m"), ctx, MoveToPose.Inputs(target=target))
    assert ctx.act.gripper_cmd["robot"] == 0.4


def test_move_to_pose_rejected_on_a_joint_only_arm(ctx):
    # workflow-lint normally catches this via requires/provides; if one slips
    # through, the boundary must still refuse rather than silently no-op.
    ctx.act = FakeActuation(dof=6, action_space="joint_position")
    task = MoveToPose(duration_s=0.05, name="m")
    task.on_enter(ctx, MoveToPose.Inputs(target=Pose.from_xyz(0.0, 0.0, 0.0)))
    with pytest.raises(UnsupportedActuation, match="no Cartesian target"):
        task.tick(ctx)


# -- Approach ------------------------------------------------------------


def test_approach_targets_a_standoff_above(ctx):
    target = Pose.from_xyz(0.1, 0.0, 0.2)
    standoff = Pose.from_xyz(0.1, 0.0, 0.3)
    ctx.scene.tcp_pose = standoff
    task = Approach(standoff=(0.0, 0.0, 0.1), duration_s=0.05, name="a")
    assert drive(task, ctx, Approach.Inputs(target=target)) is Status.SUCCESS
    assert ctx.act.ee_targets["robot"].pos[0, 2] == pytest.approx(0.3, abs=1e-4)


def test_approach_can_use_object_local_offset_and_fixed_orientation(ctx):
    # 180 degrees around z maps local +x to world -x.
    target = Pose(
        pos=np.array([[0.2, 0.0, 0.1]], dtype=np.float32),
        quat=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )
    orientation = (0.0, 0.0, 1.0, 0.0)
    task = Approach(
        standoff=(0.1, 0.0, 0.0),
        local_standoff=True,
        orientation=orientation,
        duration_s=0.01,
        settle_timeout_s=0.0,
        name="a",
    )
    task.on_enter(ctx, Approach.Inputs(target=target))
    task.tick(ctx)
    assert ctx.act.ee_targets["robot"].pos[0, 0] == pytest.approx(0.1, abs=1e-4)
    assert np.allclose(ctx.act.ee_targets["robot"].quat[0], orientation)


def test_approach_can_compose_orientation_in_object_frame(ctx):
    # Object yaw 180 followed by a local roll 180 produces probe-down pitch 180.
    target = Pose(
        pos=np.array([[0.2, 0.0, 0.1]], dtype=np.float32),
        quat=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )
    task = Approach(
        orientation=(0.0, 1.0, 0.0, 0.0),
        local_orientation=True,
        duration_s=0.01,
        settle_timeout_s=0.0,
        name="a",
    )
    task.on_enter(ctx, Approach.Inputs(target=target))
    task.tick(ctx)
    assert np.allclose(ctx.act.ee_targets["robot"].quat[0], (0.0, 0.0, 1.0, 0.0))


# -- Lift ----------------------------------------------------------------


def test_lift_rises_from_the_current_tcp(ctx):
    ctx.scene.tcp_pose = Pose.from_xyz(0.1, 0.0, 0.2)
    task = Lift(height=0.15, duration_s=0.05, settle_timeout_s=0.0, name="l")
    task.on_enter(ctx, Lift.Inputs())
    for _ in range(10):
        if task.tick(ctx).is_terminal:
            break
    assert ctx.act.ee_targets["robot"].pos[0, 2] == pytest.approx(0.35, abs=1e-4)


def test_lift_uses_an_explicit_origin_when_wired(ctx):
    ctx.scene.tcp_pose = Pose.from_xyz(9.0, 9.0, 9.0)
    task = Lift(height=0.1, duration_s=0.05, settle_timeout_s=0.0, name="l")
    task.on_enter(ctx, Lift.Inputs(from_pose=Pose.from_xyz(0.1, 0.0, 0.2)))
    for _ in range(10):
        if task.tick(ctx).is_terminal:
            break
    assert ctx.act.ee_targets["robot"].pos[0, 2] == pytest.approx(0.3, abs=1e-4)


# -- discovery -----------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS.values(), ids=[s.name for s in SPECS.values()])
def test_every_ik_task_requires_ee_pose(spec):
    """Declared on the class; discovery reads it straight off there."""
    assert spec.requires.get("action_space") == "ee_pose", (
        f"{spec.id} must declare requires = {{'action_space': 'ee_pose'}} on its class "
        f"so workflow-lint rejects it against joint-only embodiments"
    )
