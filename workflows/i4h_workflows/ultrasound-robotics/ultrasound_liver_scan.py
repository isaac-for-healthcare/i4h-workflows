# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka Panda sweeps an ultrasound probe across an abdominal phantom.

The only openpi PI0 env. Nothing about the workflow says so — the backend is a
manifest lookup, so swapping to a GR00T checkpoint would be a one-string edit.
"""

from __future__ import annotations

from i4h_engine.graph import TaskGraph, node
from i4h_engine.interface import Workflow
from i4h_tasks.basic.control.hold import Hold
from i4h_tasks.basic.control.wait_until import WaitUntil
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.ik.approach import Approach
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

# Organ-local waypoints and probe-down orientation.
APPROACH = (0.0030, 0.0500, 0.2000)
CONTACT = (0.0030, 0.0500, 0.0863)
SWEEP = (
    (-0.0310, -0.01075, 0.0800),
    (-0.0650, -0.0715, 0.0737),
)
PROBE_DOWN_LOCAL_WXYZ = (0.0, 1.0, 0.0, 0.0)


def rule_based() -> TaskGraph:
    """Locate the phantom, descend onto it, then sweep through waypoints.

    Each waypoint is its own node so a recording segments by sweep position —
    which is what lets the annotator grade individual passes.
    """
    locate = node(Locate("organs", name="locate"))
    # The relative Cartesian controller converges to about 4 cm under the
    # loaded probe, so rule-based stages advance on time.
    approach = node(
        Approach(
            standoff=APPROACH,
            local_standoff=True,
            orientation=PROBE_DOWN_LOCAL_WXYZ,
            local_orientation=True,
            duration_s=0.6,
            position_tolerance=0.05,
            name="approach",
        )
    )
    contact = node(
        Approach(
            standoff=CONTACT,
            local_standoff=True,
            orientation=PROBE_DOWN_LOCAL_WXYZ,
            local_orientation=True,
            duration_s=0.6,
            position_tolerance=0.05,
            name="make_contact",
        )
    )

    graph = (
        TaskGraph(description="Rule-based probe sweep across the phantom.")
        .flow(locate >> approach >> contact)
        .wire(locate.out.pose, approach.in_.target)
        .wire(locate.out.pose, contact.in_.target)
    )

    previous = contact
    for index, offset in enumerate(SWEEP):
        waypoint = node(
            Approach(
                standoff=offset,
                local_standoff=True,
                orientation=PROBE_DOWN_LOCAL_WXYZ,
                local_orientation=True,
                duration_s=1.0,
                position_tolerance=0.05,
                name=f"sweep_{index}",
            )
        )
        graph.flow(previous >> waypoint)
        graph.wire(locate.out.pose, waypoint.in_.target)
        previous = waypoint

    hold = node(Hold(0.5, name="hold"))
    verify = node(WaitUntil(success, timeout_s=1.2, name="verify_scan"))
    graph.flow(previous >> hold >> verify)
    return graph


def success(ctx) -> object:
    return ctx.scene.termination("success")


WORKFLOW = Workflow(
    scene="panda_phantom",
    success=success,
    modes={
        "policy": lambda: policy("openpi_pi0/ultrasound_liver_scan", until=success),
        "rule-based": rule_based,
        "teleop": lambda device="keyboard", **kwargs: teleop(device, until=success, **kwargs),
        "replay": replay,
        "idle": idle,
    },
)
