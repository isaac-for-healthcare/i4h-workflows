# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for workflow predicates that span simulator resets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_common.types import ObjectState, Pose
from i4h_engine.loader import load_workflow_module
from i4h_engine.task import TickContext

success = load_workflow_module("scissor_pick_and_place").workflow.success
block_lift_success = load_workflow_module("surgical_lift_block").workflow.success
needle_lift_success = load_workflow_module("surgical_lift_needle").workflow.success


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


class _SurgicalScene:
    def __init__(self, z: float) -> None:
        self.z = z

    def object(self, name: str) -> ObjectState:
        return ObjectState(name=name, pose=Pose.from_xyz(0.0, 0.0, self.z), lin_vel=np.zeros((1, 3)))


def test_surgical_lift_success_uses_psm_root_relative_height() -> None:
    block_ctx = TickContext(scene=_SurgicalScene(-0.08), act=SimpleNamespace())
    needle_ctx = TickContext(scene=_SurgicalScene(-0.09), act=SimpleNamespace())

    assert not block_lift_success(block_ctx).any()
    assert not needle_lift_success(needle_ctx).any()

    block_ctx.scene.z = -0.07
    needle_ctx.scene.z = -0.08
    assert block_lift_success(block_ctx).all()
    assert needle_lift_success(needle_ctx).all()
