# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State machine for ``surgical_reach_star`` (single STAR tool reach)."""

from __future__ import annotations

from typing import Any

from arena.statemachine.core.reach import ReachStateMachine


class SurgicalReachStar(ReachStateMachine):
    env_id = "surgical_reach_star"
    # The STAR reaches a much larger workspace (x 0.45-0.55) than the PSM, so its
    # absolute-IK residual is looser; 0.02 m is too tight for edge-of-range goals.
    success_tolerance_m = 0.03


def run_state_machine(*, args: Any, env: Any, app: Any, controller: Any) -> None:
    SurgicalReachStar().run(args=args, env=env, app=app, controller=controller)
