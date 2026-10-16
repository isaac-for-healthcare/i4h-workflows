# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM 101 picks up surgical scissors and places them in the tray.

The SO-ARM exposes joint-position control, so the rule-based path is
joint-space keyframes rather than IK. Two closed-loop adaptations read the
randomized scissors pose live from the scene through ``locate``: a
shoulder-pan correction proportional to its Y, wired into the four
pan-sensitive stages, and a reach correction proportional to its X, wired into
the descent and the grasp, so the jaw closes on the same part of the blade
wherever the scissors start.
"""

from __future__ import annotations

import numpy as np

from i4h_common.types import satisfied
from i4h_engine.graph import TaskGraph, node
from i4h_engine.interface import Workflow
from i4h_engine.status import Status
from i4h_engine.task import TickContext
from i4h_tasks.basic.control.settle import Settle
from i4h_tasks.basic.control.wait_until import WaitUntil
from i4h_tasks.basic.motion.home import Home
from i4h_tasks.basic.motion.keyframes import Correction, Frame, Keyframes
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.basic.predicates import inside_box
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

_OPEN = 0.35
_CLOSED = -0.16

# Both corrections were measured, not assumed (NVBug 6903612). The wrist
# camera rides on the gripper, so the blade's column in its image as the jaw
# closes is the lateral grasp error; across 76 recorded episodes it fits pose
# and correction linearly to 2.4 px. The earlier pan rule (10 rad/m on Y,
# clipped to [0, 0.12]) turned the jaw twice as far as Y moves the blade: at
# the high-Y end of the randomization the blade sat against the fixed jaw and
# the grasp slipped.
#
# Shoulder-pan correction for the randomized scissors Y (-0.025..-0.010 m):
# 4.51 rad/m holds the blade mid-jaw. It gives the old rule's pan at the
# nominal pose, so the keyframes' own reference is unchanged.
_PAN = Correction(axis=1, reference=-0.0267, gains=(4.51, 0.0, 0.0, 0.0, 0.0, 0.0), limits=(-0.005, 0.025))

# Reach correction for the randomized scissors X (0.11..0.13 m). The scissors
# lie along the arm's reach, so X decides where along them the jaw closes:
# uncorrected, a pose 7-10 mm out closed near the finger rings, which slipped
# during the lift, and one 10 mm in left the scissors caught in the jaw at
# release. The shoulder, elbow and wrist gains move the gripper along world X,
# solved from its PhysX Jacobian at open_on_scissors and close_on_scissors; the
# pan term cancels the sideways part of that motion at the blade.
_REACH = Correction(
    axis=0,
    reference=0.12,
    gains=(-1.35, -6.05, 9.70, -3.64, -1.86, 0.0),
    limits=(-0.015, 0.015),
)

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
# Open the jaw fully and hold the arm still while the scissors drop. Opening
# part-way (0.16, then 0.29) while already lifting left them caught in the jaw
# on some grasps, dragged up and dropped tip-first, so they stood in the tray
# and toppled for seconds afterwards.
_RELEASE = [
    Frame("release_in_tray", (0.52, 1.89, -1.14, -1.21, 0.65, _OPEN), 0.33),
    Frame("release_settle", (0.52, 1.89, -1.14, -1.21, 0.65, _OPEN), 0.33),
]
_RETREAT = [
    Frame("withdraw_from_tray", (0.50, 1.33, -1.21, -1.24, 0.61, _OPEN), 0.33),
    Frame("return_high", (0.47, 0.61, -0.52, -1.22, 0.32, _OPEN), 0.33),
    Frame("return_mid", (0.38, -0.07, 0.00, -1.22, 0.30, _OPEN), 0.33),
    Frame("return_near_home", (0.22, -0.15, 0.14, -0.69, 0.56, _OPEN), 0.65),
]

#: Success: the scissors ended up inside the tray volume and the arm went home.
_TRAY_BOX = ((0.02, 0.15, 0.24), (0.22, 0.35, 0.34))
#: How close the tool tip must return to where it started, in metres.
_HOME_TOLERANCE_M = 0.03


class _ScissorHome(Home):
    """Finish homing when the scene has already confirmed placement and return."""

    def tick(self, ctx: TickContext) -> Status:
        # The scene's success term checks settled scissors and the tool's home
        # tolerance. It can reset the scene before Home's interpolation ends.
        # Consume that retained result before reading or commanding the new
        # scene; the runner will finish without another physics step.
        if ctx.step > 0 and satisfied(ctx.scene.termination("success")):
            self._arrived = True
            return Status.SUCCESS
        return super().tick(ctx)


def rule_based() -> TaskGraph:
    """The pick-and-place goal decomposed into joint-space stages."""
    locate = node(Locate("scissors", name="locate"))
    leave_home = node(Keyframes(_LEAVE_HOME, name="leave_home"))
    align = node(Keyframes(_ALIGN, name="align", corrections=(_PAN,)))
    descend = node(Keyframes(_DESCEND, name="descend", corrections=(_PAN, _REACH)))
    close = node(Keyframes(_CLOSE, name="close_grip", corrections=(_PAN, _REACH)))
    lift = node(Keyframes(_LIFT, name="lift", corrections=(_PAN,)))
    carry = node(Keyframes(_CARRY, name="carry_to_tray"))
    release = node(Keyframes(_RELEASE, name="release"))
    # The retreat already gives the scissors 1.65 s to come to rest after the
    # release, so 0.4 s more is enough, and keeps the worst case inside the
    # scene's 600-step cap: 495 keyframe steps + settle 24 + verify 1 + home
    # (60 steps, up to ~70 while the arm converges on the home tolerance).
    settle = node(Settle("scissors", timeout_s=0.4, name="settle"))
    retreat = node(Keyframes(_RETREAT, name="retreat"))
    home = node(_ScissorHome(duration_s=1.0, gripper=_OPEN, name="home"))
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
    # The closed-loop adaptations: every pan- or reach-sensitive stage reads the
    # live scissors pose rather than assuming the reference position.
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
