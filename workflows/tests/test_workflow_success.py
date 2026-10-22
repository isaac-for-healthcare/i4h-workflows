# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for workflow predicates that span simulator resets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_common.types import ObjectState, Pose
from i4h_engine.loader import load_workflow_module
from i4h_engine.task import TickContext
from i4h_tasks.basic.predicates import env_success

success = load_workflow_module("scissor_pick_and_place").module.success
endoluminal = load_workflow_module("endoluminal_navigation").workflow


class _ScissorScene:
    def __init__(self) -> None:
        self.termination_value = np.array([True])

    def tcp(self) -> Pose:
        return Pose.from_xyz(0.0, 0.0, 0.0)

    def termination(self, _name: str) -> np.ndarray:
        return self.termination_value

    def object(self, name: str) -> ObjectState:
        return ObjectState(name=name, pose=Pose.from_xyz(9.0, 9.0, 9.0), lin_vel=np.zeros((1, 3)))


def test_scissor_success_ignores_previous_episode_pulse_for_initial_step() -> None:
    scene = _ScissorScene()
    ctx = TickContext(scene=scene, act=SimpleNamespace())
    assert not success(ctx).any()
    assert not success(ctx).any()

    ctx.step = 1
    assert success(ctx).all()


class _CatheterScene:
    """Reports the simulator term, which is absent without a patient twin."""

    def __init__(self, arrived: np.ndarray) -> None:
        self.arrived = arrived

    def termination(self, _name: str) -> np.ndarray:
        return self.arrived


def test_env_success_reads_the_simulator_arrival_term() -> None:
    ctx = TickContext(scene=_CatheterScene(np.array([False])), act=SimpleNamespace())
    assert not env_success(ctx).any()

    ctx.scene.arrived = np.array([True])
    assert env_success(ctx).all()


def test_env_success_is_unmet_without_a_centerline() -> None:
    """No twin means no arrival term, so the scene view returns absent-term zeros."""
    ctx = TickContext(scene=_CatheterScene(np.zeros(1, dtype=bool)), act=SimpleNamespace())

    assert not env_success(ctx).any()


def test_endoluminal_exposes_replay() -> None:
    assert "replay" in endoluminal.modes


def test_scissor_home_ignores_stale_success_and_consumes_the_current_pulse(monkeypatch) -> None:
    from i4h_engine.executor import Engine
    from i4h_engine.graph import TaskGraph
    from i4h_engine.status import WorkflowStatus
    from i4h_tasks.basic.testing.fake_scene import fake_world

    graph = load_workflow_module("scissor_pick_and_place").workflow.modes["rule-based"]()
    home = next(node for node in graph.nodes if node.id == "home")
    scene, act = fake_world(dof=6)
    scene.terminations["success"] = np.array([True])
    ctx = TickContext(scene=scene, act=act)
    engine = Engine(TaskGraph().flow(home))
    engine.start(ctx)

    # A previous attempt's pulse at step zero must not bypass homing.
    assert engine.tick(ctx) is WorkflowStatus.RUNNING
    scene.terminations["success"][:] = False
    assert engine.tick(ctx) is WorkflowStatus.RUNNING

    def post_reset_access(*_args):
        raise AssertionError("homing must not read or command the post-reset scene")

    scene.terminations["success"][:] = True
    for obj, names in [(scene, ("joints", "home_joints")), (act, ("set_joint_targets", "set_gripper"))]:
        for name in names:
            monkeypatch.setattr(obj, name, post_reset_access)
    assert engine.tick(ctx) is WorkflowStatus.SUCCEEDED
    assert not engine.terminal_advance_requested
    assert home.ref.on_exit(ctx).at_home
