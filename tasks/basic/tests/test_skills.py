# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The skill library, exercised with no simulator."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from i4h_common.types import Pose
from i4h_engine.executor import Engine
from i4h_engine.graph import TaskGraph, node
from i4h_engine.status import Status, WorkflowStatus
from i4h_engine.task import TickContext
from i4h_tasks.basic.control.settle import Settle
from i4h_tasks.basic.control.wait import Wait
from i4h_tasks.basic.control.wait_until import WaitUntil
from i4h_tasks.basic.gripper.grasp import Grasp
from i4h_tasks.basic.gripper.release import Release
from i4h_tasks.basic.motion.home import Home
from i4h_tasks.basic.motion.keyframes import Frame, Keyframes, smoothstep
from i4h_tasks.basic.motion.move_joints import MoveJoints
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.basic.predicates import all_of, at_home, inside_box, near_object, object_above, object_within
from i4h_tasks.basic.replay.replay import Replay
from i4h_tasks.basic.testing.fake_scene import fake_world

DT = 1 / 60


@pytest.fixture
def world():
    scene, act = fake_world(
        dof=6,
        objects={"scissors": Pose.from_xyz(0.12, -0.02, 0.25), "tray": Pose.from_xyz(0.12, 0.25, 0.26)},
        home=np.array([0.0, -1.6, 1.4, 1.5, -1.8, 0.0]),
    )
    return scene, act


@pytest.fixture
def ctx(world):
    scene, act = world
    return TickContext(scene=scene, act=act, dt=DT)


def drive(task, ctx, *, limit: int = 2000) -> Status:
    """Run a task to completion, stepping the fake dynamics like the runner would."""
    task.on_enter(ctx, task.Inputs() if hasattr(task, "Inputs") else None)
    status = Status.RUNNING
    for _ in range(limit):
        status = task.tick(ctx)
        ctx.scene.step(ctx.act)
        ctx.node_step += 1
        if status.is_terminal:
            break
    return status


# -- replay --------------------------------------------------------------


def test_replay_succeeds_on_its_final_action(tmp_path, ctx):
    dataset = tmp_path / "recording.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle.create_group("data").create_group("demo_0").create_dataset(
            "actions",
            data=np.zeros((2, 6), dtype=np.float32),
        )

    replay = Replay(dataset)
    replay.on_enter(ctx, None)

    assert replay.tick(ctx) is Status.RUNNING
    assert replay.tick(ctx) is Status.SUCCESS
    assert replay.advance_on_success is True


# -- smoothstep ----------------------------------------------------------


def test_smoothstep_endpoints_and_midpoint():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)


def test_smoothstep_clamps_out_of_range():
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(2.0) == 1.0


# -- keyframes -----------------------------------------------------------


def test_move_joints_reaches_target_relative_to_home(ctx):
    task = MoveJoints([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=0.1, name="m")
    assert drive(task, ctx) is Status.SUCCESS
    expected = ctx.scene.home_joints() + np.array([0.1, 0, 0, 0, 0, 0], dtype=np.float32)
    assert np.allclose(ctx.act.joint_targets["robot"], expected, atol=1e-5)


def test_move_joints_absolute_mode_ignores_home(ctx):
    task = MoveJoints([0.1] * 6, duration_s=0.1, relative_to_home=False, name="m")
    drive(task, ctx)
    assert np.allclose(ctx.act.joint_targets["robot"], 0.1, atol=1e-5)


def test_keyframes_visit_each_frame_in_order(ctx):
    frames = [Frame("a", (0.1,) * 6, 0.05), Frame("b", (0.2,) * 6, 0.05), Frame("c", (0.3,) * 6, 0.05)]
    task = Keyframes(frames, relative_to_home=False, name="k")
    seen: list[float] = []
    task.on_enter(ctx, task.Inputs())
    for _ in range(200):
        status = task.tick(ctx)
        seen.append(float(ctx.act.joint_targets["robot"][0, 0]))
        ctx.scene.step(ctx.act)
        if status.is_terminal:
            break
    assert seen[-1] == pytest.approx(0.3, abs=1e-5)
    assert max(seen) == pytest.approx(0.3, abs=1e-5)  # never overshoots
    assert all(b >= a - 1e-6 for a, b in zip(seen, seen[1:], strict=False))  # monotone


def test_keyframes_duration_sets_tick_count(ctx):
    task = Keyframes([Frame("a", (0.1,) * 6, 0.5)], relative_to_home=False, name="k")
    task.on_enter(ctx, task.Inputs())
    ticks = 0
    while task.tick(ctx) is Status.RUNNING:
        ticks += 1
        assert ticks < 1000
    assert ticks + 1 == round(0.5 / DT)


def test_keyframes_offset_adapts_to_a_located_pose(ctx):
    # The one closed-loop adaptation: shift the pan joint by the object's Y.
    task = Keyframes(
        [Frame("a", (0.0,) * 6, 0.05)],
        relative_to_home=False,
        offset_joint=0,
        offset_axis=1,
        offset_gain=10.0,
        offset_reference=-0.023,
        offset_limits=(0.0, 0.12),
        name="k",
    )
    reference = Pose.from_xyz(0.0, 0.0, 0.0)  # y=0, so delta = 0.023 * 10 = 0.23 → clamped to 0.12
    task.on_enter(ctx, task.Inputs(reference=reference))
    for _ in range(20):
        if task.tick(ctx).is_terminal:
            break
        ctx.scene.step(ctx.act)
    assert ctx.act.joint_targets["robot"][0, 0] == pytest.approx(0.12, abs=1e-4)


def test_keyframes_offset_is_noop_without_reference(ctx):
    task = Keyframes(
        [Frame("a", (0.0,) * 6, 0.05)],
        relative_to_home=False,
        offset_joint=0,
        offset_gain=10.0,
        name="k",
    )
    drive(task, ctx)
    assert ctx.act.joint_targets["robot"][0, 0] == pytest.approx(0.0, abs=1e-5)


def test_keyframes_rejects_empty():
    with pytest.raises(ValueError, match="at least one keyframe"):
        Keyframes([], name="k")


# -- home ----------------------------------------------------------------


def test_home_succeeds_when_arm_arrives(ctx):
    ctx.scene.joint_pos = ctx.scene.home + 0.5
    assert drive(Home(duration_s=0.2, name="h"), ctx) is Status.SUCCESS


def test_home_keeps_running_when_arm_is_stuck(ctx):
    # follow_rate 1.0 = the arm never moves; "commanded home" must not read as
    # "is home", which is the bug this check exists to prevent.
    ctx.scene.follow_rate = 1.0
    ctx.scene.joint_pos = ctx.scene.home + 0.5
    task = Home(duration_s=0.05, name="h")
    task.on_enter(ctx, None)
    statuses = [task.tick(ctx) for _ in range(50)]
    assert all(s is Status.RUNNING for s in statuses)


# -- locate --------------------------------------------------------------


def test_locate_outputs_object_pose(ctx):
    task = Locate("scissors")
    assert drive(task, ctx) is Status.SUCCESS
    assert np.allclose(task.on_exit(ctx).pose.pos, [[0.12, -0.02, 0.25]])


def test_locate_applies_offset(ctx):
    task = Locate("scissors", offset=(0.0, 0.0, 0.1))
    drive(task, ctx)
    assert task.on_exit(ctx).pose.pos[0, 2] == pytest.approx(0.35)


def test_locate_unknown_object_raises(ctx):
    task = Locate("scisors")
    task.on_enter(ctx, None)
    with pytest.raises(KeyError, match="no object 'scisors'"):
        task.tick(ctx)


def test_locate_waits_for_settle_then_gives_up(ctx):
    ctx.scene.set_velocity("scissors", [0.0, 0.0, 5.0])
    task = Locate("scissors", wait_for_settle=True, settle_timeout_s=0.05)
    assert drive(task, ctx) is Status.SUCCESS
    assert task.on_exit(ctx).settled is False


def test_locate_does_not_leave_the_robot_undriven(ctx):
    drive(Locate("scissors"), ctx)
    assert ctx.act.holds  # explicit hold, not silence


# -- gripper -------------------------------------------------------------


def test_set_gripper_commands_width(ctx):
    drive(Release(width=0.35, duration_s=0.05, name="r"), ctx)
    assert ctx.act.gripper_cmd["robot"] == 0.35


def test_grasp_succeeds_on_contact(ctx):
    ctx.scene.contacts.add(("robot", "scissors"))
    task = Grasp(object="scissors", duration_s=0.05, name="g")
    assert drive(task, ctx) is Status.SUCCESS
    assert task.on_exit(ctx).grasped is True


def test_grasp_fails_without_contact(ctx):
    task = Grasp(object="scissors", duration_s=0.05, name="g")
    assert drive(task, ctx) is Status.FAILURE
    assert task.on_exit(ctx).grasped is False


def test_grasp_verify_false_never_fails(ctx):
    task = Grasp(object="scissors", duration_s=0.05, verify=False, name="g")
    assert drive(task, ctx) is Status.SUCCESS


def test_grasp_falls_back_to_jaw_width_when_no_contact_data(ctx):
    # No such object in the scene, so contact is unavailable; a jaw that stopped
    # short of its command implies something is in the way.
    ctx.scene.follow_rate = 1.0  # jaw stays at 0.0 while commanded to -0.16
    task = Grasp(object="", width=-0.16, duration_s=0.05, name="g")
    assert drive(task, ctx) is Status.SUCCESS


# -- control -------------------------------------------------------------


def test_wait_runs_for_the_requested_duration(ctx):
    task = Wait(0.1, name="w")
    task.on_enter(ctx, None)
    ticks = 0
    while task.tick(ctx) is Status.RUNNING:
        ticks += 1
    assert ticks + 1 == pytest.approx(round(0.1 / DT), abs=1)


def test_wait_until_succeeds_when_predicate_fires(ctx):
    calls = {"n": 0}

    def predicate(_c):
        calls["n"] += 1
        return calls["n"] >= 3

    assert drive(WaitUntil(predicate, timeout_s=1.0, name="wu"), ctx) is Status.SUCCESS


def test_wait_until_fails_on_timeout(ctx):
    assert drive(WaitUntil(lambda _c: False, timeout_s=0.05, name="wu"), ctx) is Status.FAILURE


def test_settle_succeeds_once_object_stops(ctx):
    ctx.scene.set_velocity("scissors", [0.0, 0.0, 0.0])
    task = Settle("scissors", timeout_s=1.0)
    assert drive(task, ctx) is Status.SUCCESS
    assert task.on_exit(ctx).settled is True


def test_settle_times_out_without_failing(ctx):
    # A scene that never comes to rest should not fail the workflow; it should say so.
    ctx.scene.set_velocity("scissors", [0.0, 0.0, 5.0])
    task = Settle("scissors", timeout_s=0.05)
    assert drive(task, ctx) is Status.SUCCESS
    assert task.on_exit(ctx).settled is False


# -- predicates ----------------------------------------------------------


def test_object_within(world):
    scene, _ = world
    assert not object_within(scene, "scissors", "tray", 0.1).all()
    assert object_within(scene, "scissors", "tray", 0.5).all()


def test_object_above(world):
    scene, _ = world
    assert object_above(scene, "scissors", 0.2).all()
    assert not object_above(scene, "scissors", 0.3).all()


def test_inside_box(world):
    scene, _ = world
    assert inside_box(scene, "tray", (0.0, 0.2, 0.2), (0.2, 0.3, 0.3)).all()
    assert not inside_box(scene, "scissors", (0.0, 0.2, 0.2), (0.2, 0.3, 0.3)).all()


def test_at_home_and_near_object(world):
    scene, _ = world
    assert at_home(scene).all()
    scene.joint_pos = scene.home + 0.5
    assert not at_home(scene).all()
    assert not near_object(scene, "scissors", 0.01).all()


def test_all_of_combines(world):
    scene, _ = world
    combined = all_of(lambda s: object_above(s, "scissors", 0.2), lambda s: at_home(s))
    assert combined(scene).all()


# -- integration through the engine --------------------------------------


def test_scripted_sequence_runs_as_a_workflow(ctx):
    locate = node(Locate("scissors"))
    approach = node(
        Keyframes([Frame("descend", (0.0, 1.9, -1.0, -1.0, -0.15, -0.16), 0.1)], name="approach"),
    )
    grasp = node(Grasp(object="scissors", duration_s=0.05, verify=False, name="grasp"))
    home = node(Home(duration_s=0.1, name="go_home"))

    workflow = TaskGraph().flow(locate >> approach >> grasp >> home).wire(locate.out.pose, approach.in_.reference)
    engine = Engine(workflow)
    engine.start(ctx)
    for _ in range(3000):
        if engine.status.is_terminal:
            break
        engine.tick(ctx)
        ctx.scene.step(ctx.act)
    assert engine.status is WorkflowStatus.SUCCEEDED, engine.detail
    assert [name for name, _t, _s, _e in engine.segments] == ["locate_scissors", "approach", "grasp", "go_home"]


def test_engine_rejects_two_motion_branches_on_one_arm(ctx):
    from i4h_common.world import ActuationConflict

    left = node(MoveJoints([0.1] * 6, duration_s=1.0, name="left"))
    right = node(MoveJoints([0.2] * 6, duration_s=1.0, name="right"))
    workflow = TaskGraph().flow(node(Wait(0.0, name="root")) >> (left, right))
    engine = Engine(workflow)
    engine.start(ctx)
    engine.tick(ctx)
    with pytest.raises(ActuationConflict):
        engine.tick(ctx)


def test_motion_and_gripper_branches_coexist(ctx):
    arm = node(MoveJoints([0.1] * 6, duration_s=0.2, name="arm"))
    jaw = node(Release(duration_s=0.2, name="jaw"))
    workflow = TaskGraph().flow(node(Wait(0.0, name="root")) >> (arm, jaw))
    engine = Engine(workflow)
    engine.start(ctx)
    for _ in range(200):
        if engine.status.is_terminal:
            break
        engine.tick(ctx)
        ctx.scene.step(ctx.act)
    assert engine.status is WorkflowStatus.SUCCEEDED
