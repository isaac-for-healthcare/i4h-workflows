# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The action contract negotiated on the ready handshake.

Before this existed, ``RemoteTask`` assumed every policy emitted joint targets.
Seven of twelve remote tasks target ``ee_pose`` scenes, so they would have
raised on the first tick, three minutes into a Kit launch.

The backend is authoritative because it is the only party that has loaded the
checkpoint — a manifest goes stale the moment someone passes ``--checkpoint``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeActuation, FakeScene

from i4h_common.bus.inproc import InProcBus
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import ActionChunk, TaskStatusMsg, encode
from i4h_common.manifest import BackendSpec, TaskSpec
from i4h_engine.remote import RemoteTask, RemoteTaskError, _axis_angle_to_quat, _euler_to_quat
from i4h_engine.status import Status
from i4h_engine.task import TickContext


def _spec(action_space: str, name: str = "t", robots: tuple[str, ...] = ()) -> TaskSpec:
    return TaskSpec(
        project="gr00t_n15",
        name=name,
        runtime="remote",
        backend=BackendSpec(project="tasks/gr00t_n15", entry="x:main"),
        outputs={"success": "bool"},
        requires={"action_space": action_space},
    )


class Backend:
    """Fake policy server that reports a contract and emits matching rows."""

    def __init__(self, bus, keys, uid, *, contract: dict, row: list[float]) -> None:
        self.bus, self.keys, self.uid = bus, keys, uid
        self.contract = contract
        self.row = row
        bus.subscribe(keys.task_spec(uid), self._on_spec)
        bus.subscribe(keys.task_obs(uid), self._on_obs)

    def _on_spec(self, _k, payload):
        self.bus.publish(
            self.keys.task_status(self.uid),
            encode(TaskStatusMsg(task_uid=self.uid, status="ready", **self.contract)),
        )

    def _on_obs(self, _k, payload):
        self.bus.publish(
            self.keys.task_action(self.uid),
            encode(ActionChunk(task_uid=self.uid, horizon=1, dof=len(self.row), actions=list(self.row))),
        )


def _wire(action_space: str, contract: dict, row: list[float], *, dof: int = 7, robots=("robot",)):
    bus, keys = InProcBus(), Keys("run")
    scene = FakeScene(dof=dof)

    class _Scene(type(scene)):
        pass

    ctx = TickContext(
        scene=scene, act=FakeActuation(dof=dof, action_space=action_space), dt=1 / 60, bus=bus, run_id="run"
    )
    task = RemoteTask(_spec(action_space), keys=keys)
    Backend(bus, keys, f"{task.name}-0", contract=contract, row=row)
    return task, ctx, bus


# -- joint scenes are unchanged ------------------------------------------


def test_joint_policy_writes_joint_targets():
    task, ctx, _ = _wire(
        "joint_position",
        {"action_space": "joint_position", "action_layout": "joints", "action_dof": 6},
        [0.1] * 6,
        dof=6,
    )
    task.on_enter(ctx, None)
    assert task.tick(ctx) is Status.RUNNING
    assert np.allclose(ctx.act.joint_targets["robot"], 0.1)


# -- action contract coverage --------------------------------------------


def test_ee_pose_policy_writes_a_cartesian_target():
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "pos_quat", "action_dof": 7},
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
    )
    task.on_enter(ctx, None)
    assert task.tick(ctx) is Status.RUNNING
    pose = ctx.act.ee_targets["robot"]
    assert np.allclose(pose.pos, [[0.1, 0.2, 0.3]])
    assert not ctx.act.joint_targets  # emphatically not joints


def test_ee_pose_with_a_trailing_jaw_column():
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "pos_quat", "action_dof": 8, "action_gripper": "last"},
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, -0.5],
        dof=8,
    )
    task.on_enter(ctx, None)
    task.tick(ctx)
    assert np.allclose(ctx.act.ee_targets["robot"].pos, [[0.1, 0.2, 0.3]])
    assert np.allclose(ctx.act.gripper_cmd["robot"], -0.5)


def test_euler_layout_is_converted():
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "pos_euler", "action_dof": 6},
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        dof=6,
    )
    task.on_enter(ctx, None)
    task.tick(ctx)
    assert np.allclose(ctx.act.ee_targets["robot"].quat, [[1.0, 0.0, 0.0, 0.0]], atol=1e-6)


def test_axis_angle_layout_is_converted():
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "pos_axis_angle", "action_dof": 6},
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        dof=6,
    )
    task.on_enter(ctx, None)
    task.tick(ctx)
    assert np.allclose(ctx.act.ee_targets["robot"].quat, [[1.0, 0.0, 0.0, 0.0]], atol=1e-6)


def test_relative_axis_angle_layout_is_passed_through():
    row = [0.01, -0.02, 0.03, 0.04, -0.05, 0.06]
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "delta_axis_angle", "action_dof": 6},
        row,
        dof=6,
    )
    task.on_enter(ctx, None)
    assert task.tick(ctx) is Status.RUNNING
    assert np.allclose(ctx.act.ee_deltas["robot"], [row])
    assert not ctx.act.ee_targets


# -- mismatch is fatal, not silent ---------------------------------------


def test_backend_scene_mismatch_refuses_to_run():
    # A joint-space checkpoint serving an ee_pose scene: continuing would drive
    # the arm with numbers that mean something else entirely.
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "joint_position", "action_layout": "joints", "action_dof": 7},
        [0.0] * 7,
    )
    task.on_enter(ctx, None)
    # The contract is negotiated on the ready handshake, which now lands in
    # tick() so the simulator keeps stepping while the backend starts.
    with pytest.raises(RemoteTaskError, match="does not match this scene"):
        task.tick(ctx)


def test_backend_action_width_mismatch_refuses_to_run():
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "delta_axis_angle", "action_dof": 7},
        [0.0] * 7,
        dof=6,
    )
    task.on_enter(ctx, None)
    with pytest.raises(RemoteTaskError, match="7-value actions.*accepts 6"):
        task.tick(ctx)


def test_unknown_layout_is_rejected_with_a_readable_error():
    # Guessing at an unknown layout would drive the arm with misread numbers,
    # so the proxy raises. The engine turns that into a node failure (below).
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "quaternion_first", "action_dof": 7},
        [0.0] * 7,
    )
    task.on_enter(ctx, None)
    with pytest.raises(RemoteTaskError, match="quaternion_first"):
        task.tick(ctx)


def test_short_row_for_declared_layout_is_rejected():
    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "pos_quat", "action_dof": 4},
        [0.1, 0.2, 0.3, 1.0],
        dof=4,
    )
    task.on_enter(ctx, None)
    with pytest.raises(RemoteTaskError, match="pos_quat needs"):
        task.tick(ctx)


# -- dual-arm splitting --------------------------------------------------


def test_dual_arm_chunk_splits_across_robots():
    bus, keys = InProcBus(), Keys("run")
    scene = FakeScene(dof=14)
    ctx = TickContext(scene=scene, act=FakeActuation(dof=14, action_space="ee_pose"), dt=1 / 60, bus=bus, run_id="run")
    task = RemoteTask(_spec("ee_pose"), keys=keys)
    row = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9, 0.8, 0.7, 1.0, 0.0, 0.0, 0.0]
    Backend(
        bus,
        keys,
        f"{task.name}-0",
        contract={
            "action_space": "ee_pose",
            "action_layout": "pos_quat",
            "action_dof": 14,
            "action_robots": ["psm1", "psm2"],
        },
        row=row,
    )
    task.on_enter(ctx, None)
    task.tick(ctx)
    assert np.allclose(ctx.act.ee_targets["psm1"].pos, [[0.1, 0.2, 0.3]])
    assert np.allclose(ctx.act.ee_targets["psm2"].pos, [[0.9, 0.8, 0.7]])


def test_indivisible_width_across_robots_is_an_error():
    bus, keys = InProcBus(), Keys("run")
    ctx = TickContext(
        scene=FakeScene(dof=9), act=FakeActuation(dof=9, action_space="ee_pose"), dt=1 / 60, bus=bus, run_id="run"
    )
    task = RemoteTask(_spec("ee_pose"), keys=keys)
    Backend(
        bus,
        keys,
        f"{task.name}-0",
        contract={
            "action_space": "ee_pose",
            "action_layout": "pos_quat",
            "action_dof": 9,
            "action_robots": ["psm1", "psm2"],
        },
        row=[0.0] * 9,
    )
    task.on_enter(ctx, None)
    with pytest.raises(RemoteTaskError, match="does not divide evenly"):
        task.tick(ctx)


def test_engine_turns_a_contract_error_into_a_node_failure():
    """Through the engine, a bad contract is a clean failure with the reason kept."""
    from i4h_engine.executor import Engine
    from i4h_engine.graph import TaskGraph, node
    from i4h_engine.status import WorkflowStatus

    task, ctx, _ = _wire(
        "ee_pose",
        {"action_space": "ee_pose", "action_layout": "quaternion_first", "action_dof": 7},
        [0.0] * 7,
    )
    engine = Engine(TaskGraph().flow(node(task)))
    engine.start(ctx)
    for _ in range(5):
        if engine.status.is_terminal:
            break
        engine.tick(ctx)
    assert engine.status is WorkflowStatus.FAILED
    assert "quaternion_first" in engine.states[task.name].detail


# -- backwards compatibility ---------------------------------------------


def test_backend_reporting_no_contract_falls_back_to_the_manifest():
    # An older backend that predates the contract fields still works, because
    # the manifest's requires.action_space is the fallback.
    task, ctx, _ = _wire("joint_position", {}, [0.2] * 6, dof=6)
    task.on_enter(ctx, None)
    task.tick(ctx)
    assert np.allclose(ctx.act.joint_targets["robot"], 0.2)


# -- rotation helpers ----------------------------------------------------


def test_axis_angle_identity_and_quarter_turn():
    assert np.allclose(_axis_angle_to_quat(np.zeros((1, 3), np.float32)), [[1, 0, 0, 0]])
    half_pi_z = np.array([[0.0, 0.0, np.pi / 2]], np.float32)
    expected = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]], np.float32)
    assert np.allclose(_axis_angle_to_quat(half_pi_z), expected, atol=1e-6)


def test_euler_identity_and_quarter_turn():
    assert np.allclose(_euler_to_quat(np.zeros((1, 3), np.float32)), [[1, 0, 0, 0]])
    yaw = np.array([[0.0, 0.0, np.pi / 2]], np.float32)
    expected = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]], np.float32)
    assert np.allclose(_euler_to_quat(yaw), expected, atol=1e-6)


def test_rotation_helpers_return_unit_quaternions():
    rot = np.array([[0.3, -0.4, 0.5]], np.float32)
    for fn in (_axis_angle_to_quat, _euler_to_quat):
        assert np.allclose(np.linalg.norm(fn(rot), axis=-1), 1.0, atol=1e-6)
