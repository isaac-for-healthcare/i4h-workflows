# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State machine for ``surgical_reach_psm`` (single dVRK PSM reach)."""

from __future__ import annotations

from typing import Any

from arena.statemachine.core.reach import ReachStateMachine


class SurgicalReachPsm(ReachStateMachine):
    env_id = "surgical_reach_psm"


def run_state_machine(*, args: Any, env: Any, app: Any, controller: Any) -> None:
    SurgicalReachPsm().run(args=args, env=env, app=app, controller=controller)
