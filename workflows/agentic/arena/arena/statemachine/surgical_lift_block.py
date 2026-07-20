# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State machine for ``surgical_lift_block`` (lift the peg-transfer block)."""

from __future__ import annotations

from typing import Any

from arena.statemachine.core.lift import LiftStateMachine


class SurgicalLiftBlock(LiftStateMachine):
    env_id = "surgical_lift_block"


def run_state_machine(*, args: Any, env: Any, app: Any, controller: Any) -> None:
    SurgicalLiftBlock().run(args=args, env=env, app=app, controller=controller)
