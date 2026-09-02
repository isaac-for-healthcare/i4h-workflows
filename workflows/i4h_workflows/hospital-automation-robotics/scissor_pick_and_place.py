# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM 101 picks up surgical scissors and places them in the tray.

The SO-ARM exposes joint-position control, so the rule-based path is
joint-space keyframes rather than IK. The single closed-loop adaptation is a
shoulder-pan correction proportional to the randomized scissors Y, read live
from the scene by ``locate`` and wired into the four pan-sensitive stages.
"""

from __future__ import annotations

import numpy as np

from i4h_engine.graph import TaskGraph, node
from i4h_engine.interface import Workflow
from i4h_tasks.basic.control.settle import Settle
from i4h_tasks.basic.control.wait_until import WaitUntil
from i4h_tasks.basic.motion.home import Home
from i4h_tasks.basic.motion.keyframes import Frame, Keyframes
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.basic.predicates import inside_box
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

_OPEN = 0.35
_CLOSED = -0.16

# Shoulder-pan correction for the randomized scissors Y.
_PAN = {
    "offset_joint": 0,
    "offset_axis": 1,
    "offset_gain": 10.0,
    "offset_reference": -0.023,
    "offset_limits": (0.0, 0.12),
}

# Joint deltas from the home pose; index 5 is the jaw (+open / -close).
_LEAVE_HOME = [
    Frame("settle_home", (0.0, 0.0, 0.0, 0.0, 0.0, _OPEN), 0.25),
    Frame("leave_home", (0.03, 0.34, -0.19, -0.33, 0.25, -0.08), 0.35),
    Frame("clear_front", (0.09, 0.37, -0.20, -0.48, 0.42, -0.12), 0.33),
    Frame("orient_front", (0.16, 0.25, -0.14, -0.71, 0.51, -0.12), 0.33),
    Frame("orient_scissors", (0.25, 0.06, 0.03, -0.97, 0.50, -0.15), 0.33),
]
_ALIGN = [
    Frame("align_left", (-0.14, -0.03, 0.11, -1.08, 0.51, -0.15), 0.33),
    Frame("align_scissors", (-0.32, -0.05, 0.13, -1.08, 0.52, _CLOSED), 0.33),
]
_DESCEND = [
    Frame("descend_high", (-0.37, 0.59, -0.44, -1.08, 0.16, _CLOSED), 0.33),
    Frame("descend_low", (-0.37, 1.23, -0.92, -1.07, -0.14, -0.07), 0.33),
    Frame("open_pregrasp", (-0.41, 1.75, -1.14, -1.07, -0.16, 0.23), 0.33),
    Frame("open_on_scissors", (-0.48, 1.92, -1.05, -1.00, -0.15, 0.25), 0.33),
]
_CLOSE = [
    Frame("close_on_scissors", (-0.46, 1.93, -0.94, -0.95, -0.14, -0.15), 0.33),
    Frame("close_settle", (-0.43, 1.90, -0.92, -1.05, -0.16, _CLOSED), 0.33),
    Frame("seat_grip", (-0.27, 1.68, -1.08, -0.51, -0.10, _CLOSED), 0.33),
]
_LIFT = [Frame("lift_scissors", (-0.01, 1.36, -1.10, -0.42, -0.13, _CLOSED), 0.33)]
_CARRY = [
    Frame("carry_mid", (0.50, 1.16, -1.18, -0.41, 0.12, -0.15), 0.33),
    Frame("carry_to_tray", (0.69, 1.33, -1.16, -0.72, 0.44, _CLOSED), 0.33),
    Frame("lower_over_tray", (0.58, 1.70, -1.24, -1.15, 0.62, _CLOSED), 0.33),
]
_RELEASE = [
    Frame("release_in_tray", (0.52, 1.89, -1.14, -1.21, 0.65, 0.16), 0.33),
    Frame("release_settle", (0.50, 1.71, -1.14, -1.23, 0.66, 0.29), 0.33),
]
_RETREAT = [
    Frame("withdraw_from_tray", (0.50, 1.33, -1.21, -1.24, 0.61, 0.29), 0.33),
    Frame("return_high", (0.47, 0.61, -0.52, -1.22, 0.32, 0.29), 0.33),
    Frame("return_mid", (0.38, -0.07, 0.00, -1.22, 0.30, 0.29), 0.33),
    Frame("return_near_home", (0.22, -0.15, 0.14, -0.69, 0.56, 0.29), 0.65),
]

#: Success: the scissors ended up inside the tray volume and the arm went home.
_TRAY_BOX = ((0.02, 0.15, 0.24), (0.22, 0.35, 0.34))
#: How close the tool tip must return to where it started, in metres.
_HOME_TOLERANCE_M = 0.03


def rule_based() -> TaskGraph:
    """The pick-and-place goal decomposed into joint-space stages."""
    locate = node(Locate("scissors", name="locate"))
    leave_home = node(Keyframes(_LEAVE_HOME, name="leave_home"))
    align = node(Keyframes(_ALIGN, name="align", **_PAN))
    descend = node(Keyframes(_DESCEND, name="descend", **_PAN))
    close = node(Keyframes(_CLOSE, name="close_grip", **_PAN))
    lift = node(Keyframes(_LIFT, name="lift", **_PAN))
    carry = node(Keyframes(_CARRY, name="carry_to_tray"))
    release = node(Keyframes(_RELEASE, name="release"))
    settle = node(Settle("scissors", timeout_s=1.0, name="settle"))
    retreat = node(Keyframes(_RETREAT, name="retreat"))
    home = node(Home(duration_s=1.0, gripper=_OPEN, name="home"))
    verify = node(
        WaitUntil(
            lambda ctx: inside_box(ctx.scene, "scissors", *_TRAY_BOX),
            timeout_s=0.5,
            name="verify_placement",
        )
    )

    graph = TaskGraph(
        description="Rule-based joint rollout with a pan correction for randomized scissors.",
    ).flow(
        locate
        >> leave_home
        >> align
        >> descend
        >> close
        >> lift
        >> carry
        >> release
        >> retreat
        >> settle
        >> verify
        >> home
    )
    # The one closed-loop adaptation: every pan-sensitive stage reads the live
    # scissors pose rather than assuming the reference position.
    for stage in (align, descend, close, lift):
        graph.wire(locate.out.pose, stage.in_.reference)
    return graph


def success(ctx) -> object:
    """Scissors in the tray, and the tool tip back where it started.

    "Back home" is the tool tip's Cartesian distance from its pose at episode
    start, not joint angles: the final keyframe leaves the jaw open, so a
    joint-space comparison against the home pose can never match. The
    termination pulse is included because IsaacLab resets a successful
    environment immediately; by the next workflow tick the live object poses
    belong to the new episode.
    """
    scene = ctx.scene
    home_tcp = ctx.blackboard.get("home_tcp")
    if home_tcp is None:
        ctx.blackboard["home_tcp"] = np.array(scene.tcp().pos, copy=True)
    home_tcp = ctx.blackboard["home_tcp"]
    # A remote task may wait for inference while the engine ticks several
    # times without advancing the simulator. Keep the gate closed for that
    # entire initial step, while the termination buffer can still contain
    # the previous episode's pulse.
    if ctx.step == 0:
        return np.zeros(home_tcp.shape[:-1], dtype=bool)
    terminated = np.asarray(scene.termination("success"), dtype=bool)
    returned = np.linalg.norm(scene.tcp().pos - home_tcp, axis=-1) < _HOME_TOLERANCE_M
    live_success = np.asarray(inside_box(scene, "scissors", *_TRAY_BOX)) & returned
    return terminated | live_success


def timeout_success(ctx) -> object:
    """Accept success when the scissors finish inside the tray."""
    return inside_box(ctx.scene, "scissors", *_TRAY_BOX)


WORKFLOW = Workflow(
    scene="soarm_scissors",
    success=success,
    modes={
        "policy": lambda: policy(
            "gr00t_n15/scissor_pick_and_place",
            until=success,
            timeout_success=timeout_success,
        ),
        "rule-based": rule_based,
        "policy_n17": lambda: policy(
            "gr00t_n17/scissor_pick_and_place",
            until=success,
            timeout_success=timeout_success,
        ),
        "teleop": lambda device="keyboard", **kwargs: teleop(device, until=success, **kwargs),
        "replay": replay,
        "idle": idle,
    },
)
