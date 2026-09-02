# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 grasps a tray from the shelf, turns, and places it on the cart."""

from __future__ import annotations

from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import object_within
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop


def success(ctx) -> object:
    """Success: the tray ended up on the cart."""
    return object_within(ctx.scene, "tray", "cart", radius=0.9)


WORKFLOW = Workflow(
    scene="g1_tray",
    success=success,
    modes={
        "policy": lambda: policy("gr00t_n16/locomanip_tray_pick_and_place", until=success),
        "teleop": lambda device="keyboard_23d", **kwargs: teleop(device, until=success, **kwargs),
        "replay": replay,
        "idle": idle,
    },
)
