# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The scissor rule-based controller's release and step budget (NVBug 6903612)."""

from __future__ import annotations

import numpy as np

from i4h_engine.loader import load_workflow_module
from i4h_engine.registry import default_registry

scissors = load_workflow_module("scissor_pick_and_place").module
JAW = 5


def test_the_release_lowers_with_the_jaw_closed_then_opens_with_the_arm_still() -> None:
    """Opening during the descent let the gripper flip the falling scissors onto their end."""
    lower, release = scissors._RELEASE
    assert lower.target[JAW] == scissors._CLOSED
    assert release.target[:JAW] == lower.target[:JAW]
    assert release.target[JAW] == scissors._OPEN


def test_the_retreat_keeps_the_jaw_fully_open() -> None:
    assert all(frame.target[JAW] == scissors._OPEN for frame in scissors._RETREAT)


def test_the_pan_reach_correction_moves_along_the_reach_direction_only() -> None:
    """It slides the grip along the scissors without moving the blade sideways in the jaw."""
    pan_reach = np.asarray(scissors._PAN_REACH.gains)
    reach = np.asarray(scissors._REACH_GAINS)
    scale = pan_reach @ reach / (reach @ reach)
    assert scale > 0
    np.testing.assert_allclose(pan_reach, scale * reach, atol=1e-9)


def test_the_worst_case_success_path_fits_the_scene_step_cap() -> None:
    """The cap is not raised; settle times out at most, verify passes at once, and home has room to converge."""
    scene = default_registry().scene("soarm_scissors")
    dt = 1.0 / scene.control_hz
    keyframes = (
        scissors._LEAVE_HOME
        + scissors._ALIGN
        + scissors._DESCEND
        + scissors._CLOSE
        + scissors._LIFT
        + scissors._CARRY
        + scissors._RELEASE
        + scissors._RETREAT
    )
    locate = 1
    keyframe_steps = sum(max(1, round(frame.duration_s / dt)) for frame in keyframes)
    settle = round(0.4 / dt)
    verify = 1
    home = round(1.0 / dt)
    worst = locate + keyframe_steps + settle + verify + home
    assert scene.max_steps - worst >= 15, f"{worst} steps of {scene.max_steps}"
