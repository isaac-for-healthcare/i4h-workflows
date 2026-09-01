# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two dVRK PSM arms reach their targets simultaneously.

The rule-based mode is the reason the engine models parallel branches: both
arms are genuinely active in the same tick. They claim different robots, so the
actuation-conflict check permits it — two nodes driving the *same* arm would
raise.
"""

from __future__ import annotations

from i4h_engine.graph import TaskGraph, node
from i4h_engine.interface import Workflow
from i4h_tasks.basic.control.wait import Wait
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.basic.predicates import all_of, near_object
from i4h_tasks.ik.hold_pose import HoldPose
from i4h_tasks.ik.move_to_pose import MoveToPose
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.replay import replay


def rule_based() -> TaskGraph:
    """Both arms drive in the same tick, claiming different robots."""
    locate_1 = node(Locate("reach_target_1", robot="psm1", name="locate_1"))
    locate_2 = node(Locate("reach_target_2", robot="psm2", name="locate_2"))
    rest_1 = node(HoldPose(0.5, robot="psm1", name="rest_1"))
    rest_2 = node(HoldPose(0.5, robot="psm2", name="rest_2"))
    reach_1 = node(MoveToPose(duration_s=1.5, position_tolerance=0.01, robot="psm1", name="reach_1"))
    reach_2 = node(MoveToPose(duration_s=1.5, position_tolerance=0.01, robot="psm2", name="reach_2"))
    settle = node(Wait(0.5, name="settle"))
    return (
        TaskGraph(description="Both arms reach in parallel.")
        .flow(locate_1 >> rest_1 >> reach_1 >> settle, locate_2 >> rest_2 >> reach_2 >> settle)
        .wire(locate_1.out.pose, reach_1.in_.target)
        .wire(locate_2.out.pose, reach_2.in_.target)
    )


def success(ctx) -> object:
    return all_of(
        lambda s: near_object(s, "reach_target_1", radius=0.01, robot="psm1"),
        lambda s: near_object(s, "reach_target_2", radius=0.01, robot="psm2"),
    )(ctx.scene)


WORKFLOW = Workflow(
    scene="dual_psm_reach",
    success=success,
    modes={
        "rule-based": rule_based,
        "replay": replay,
        "idle": idle,
    },
)
