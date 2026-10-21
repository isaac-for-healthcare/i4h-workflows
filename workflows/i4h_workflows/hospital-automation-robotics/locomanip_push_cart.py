# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 walks to a cart, grips it, and moves it forward."""

from __future__ import annotations

from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import env_success
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

WORKFLOW = Workflow(
    scene="g1_cart",
    modes={
        "policy": lambda: policy("gr00t_n16/locomanip_push_cart", until=env_success),
        "teleop": lambda device="keyboard_23d", **kwargs: teleop(device, until=env_success, **kwargs),
        "replay": replay,
        "idle": idle,
    },
)
