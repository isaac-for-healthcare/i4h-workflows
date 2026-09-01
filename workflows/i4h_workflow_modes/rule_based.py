# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable rule-based shapes for the Cartesian surgical scenes.

Most workflows write their own rule-based builder. The six dVRK/STAR scenes differ
only in which arm is mounted and which object sits on the table — both scene
properties, not workflow structure — so they share these two.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from i4h_engine.graph import TaskGraph, node
from i4h_tasks.basic.control.wait import Wait
from i4h_tasks.basic.control.wait_until import WaitUntil
from i4h_tasks.basic.gripper.grasp import Grasp
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.ik.approach import Approach
from i4h_tasks.ik.hold_pose import HoldPose
from i4h_tasks.ik.lift import Lift
from i4h_tasks.ik.move_to_pose import MoveToPose


def cartesian_reach(
    target: str,
    *,
    pre_hold_s: float = 0.0,
    duration_s: float = 1.5,
    position_tolerance: float = 0.01,
    settle_timeout_s: float = 2.0,
    timeout_success: Callable[[Any], Any] | None = None,
) -> TaskGraph:
    """Locate a target, servo the tool tip onto it, hold."""
    locate = node(Locate(target, name="locate"))
    pre_hold = node(HoldPose(pre_hold_s, name="rest")) if pre_hold_s > 0.0 else None
    reach = node(
        MoveToPose(
            duration_s=duration_s,
            position_tolerance=position_tolerance,
            settle_timeout_s=settle_timeout_s,
            name="reach",
        )
    )
    hold = node(Wait(0.5, name="hold"))
    graph = TaskGraph(
        description="Rule-based Cartesian reach.",
        timeout_success=timeout_success,
    )
    graph.flow(locate >> pre_hold >> reach >> hold if pre_hold is not None else locate >> reach >> hold)
    return graph.wire(locate.out.pose, reach.in_.target)


def cartesian_lift(
    target: str,
    *,
    success: Callable[[Any], Any],
    height: float = 0.10,
    position_tolerance: float = 0.04,
) -> TaskGraph:
    """Approach, descend, grasp, lift, and verify the object moved.

    Approach and descend remain separate so each phase has an explicit,
    inspectable trajectory segment.
    """
    # Let gravity/contact settle the randomized object before Locate snapshots
    # it. The source controller spent the same initial 0.5 s in REST while
    # continuing to read the live object pose.
    rest = node(HoldPose(0.5, name="rest"))
    locate = node(Locate(target, wait_for_settle=True, name="locate"))
    # The dVRK differential IK controller has a roughly 2 cm residual near the
    # insertion limit even when the jaws are correctly over the object, so use
    # a tolerance that reflects the controller's measured accuracy.
    approach = node(
        Approach(
            standoff=(0.0, 0.0, 0.05),
            duration_s=1.0,
            position_tolerance=position_tolerance,
            name="approach",
        )
    )
    descend = node(MoveToPose(duration_s=0.6, position_tolerance=position_tolerance, name="descend"))
    # Surgical scenes currently have no contact sensor, and their binary jaw
    # action is not numerically comparable to the measured joint width. Verify
    # the grasp by the terminal object-height check instead.
    grasp = node(Grasp(object=target, duration_s=0.4, verify=False, name="grasp"))
    lift = node(
        Lift(
            height=height,
            duration_s=1.0,
            position_tolerance=position_tolerance,
            name="lift",
        )
    )
    verify = node(WaitUntil(success, timeout_s=0.5, name="verify_lift"))
    hold = node(Wait(0.5, name="hold"))

    graph = (
        TaskGraph(description="Rule-based approach, grasp and lift.")
        .flow(rest >> locate >> approach >> descend >> grasp >> lift >> verify >> hold)
        .wire(locate.out.pose, approach.in_.target)
        .wire(locate.out.pose, descend.in_.target)
    )
    return graph
