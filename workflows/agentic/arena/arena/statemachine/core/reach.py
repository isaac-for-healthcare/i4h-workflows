# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable reach behaviors: servo an arm's tool tip to its commanded pose.

Shared by the surgical reach envs. Success is an explicit predicate -- the tool
tip reaching the commanded pose within ``success_tolerance_m`` -- rather than the
command task's timeout-only termination.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from arena.statemachine.core.common import (
    dual_action_layout,
    frame_pose_b,
    set_gripper,
    step_dt,
    zero_dual_pose_action,
    zero_pose_action,
)
from arena.statemachine.core.machine import Stage, StateMachine


class ReachStateMachine(StateMachine):
    """Single-arm reach: hold, then servo the tool tip to the commanded pose."""

    robot_name: str = "robot"
    frame_name: str = "ee_frame"
    command_name: str = "ee_pose"
    default_max_steps = 150
    hold_last = True
    success_tolerance_m: float = 0.02

    def initial_action(self, env: Any) -> torch.Tensor:
        return zero_pose_action(env)

    def build_stages(self, env: Any) -> Sequence[Stage]:
        dt = step_dt(env)
        return [
            Stage("rest", self._hold_current, max(1, round(0.5 / dt))),
            Stage("reach", self._reach, max(1, round(1.0 / dt))),
        ]

    def succeeded(self, env: Any) -> torch.Tensor:
        return self._command_error(env) <= self.success_tolerance_m

    def log_fields(self, env: Any) -> dict[str, Any]:
        return {"error_m": f"{float(self._command_error(env).min()):.4f}"}

    def _hold_current(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        pos_b, quat_w = frame_pose_b(env, self.robot_name, self.frame_name)
        action = zero_pose_action(env)
        action[:, :3] = pos_b
        action[:, 3:7] = quat_w
        set_gripper(action, 1.0)
        return action

    def _reach(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        desired = env.unwrapped.command_manager.get_command(self.command_name)
        action = zero_pose_action(env)
        action[:, :7] = desired[:, :7]
        set_gripper(action, 1.0)
        return action

    def _command_error(self, env: Any) -> torch.Tensor:
        pos_b, _ = frame_pose_b(env, self.robot_name, self.frame_name)
        desired = env.unwrapped.command_manager.get_command(self.command_name)
        return torch.linalg.norm(pos_b - desired[:, :3], dim=-1)


class DualReachStateMachine(StateMachine):
    """Two-arm reach: servo both PSM tool tips to their commanded poses."""

    default_max_steps = 150
    hold_last = True
    success_tolerance_m: float = 0.02

    def initial_action(self, env: Any) -> torch.Tensor:
        return zero_dual_pose_action(env)

    def build_stages(self, env: Any) -> Sequence[Stage]:
        dt = step_dt(env)
        return [
            Stage("rest", self._hold_current, max(1, round(0.5 / dt))),
            Stage("reach", self._reach, max(1, round(1.0 / dt))),
        ]

    def succeeded(self, env: Any) -> torch.Tensor:
        return self._command_error(env) <= self.success_tolerance_m

    def log_fields(self, env: Any) -> dict[str, Any]:
        return {"error_m": f"{float(self._command_error(env).min()):.4f}"}

    def _hold_current(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        arm_1, _, arm_2, _ = dual_action_layout(env.action_space.shape[-1])
        ee_1_pos, ee_1_quat = frame_pose_b(env, "robot_1", "ee_frame")
        ee_2_pos, ee_2_quat = frame_pose_b(env, "robot_2", "ee_2_frame")
        action = zero_dual_pose_action(env)
        action[:, arm_1.start : arm_1.start + 3] = ee_1_pos
        action[:, arm_1.start + 3 : arm_1.start + 7] = ee_1_quat
        action[:, arm_2.start : arm_2.start + 3] = ee_2_pos
        action[:, arm_2.start + 3 : arm_2.start + 7] = ee_2_quat
        return action

    def _reach(self, env: Any, i: int, start: torch.Tensor) -> torch.Tensor:
        arm_1, _, arm_2, _ = dual_action_layout(env.action_space.shape[-1])
        desired_1 = env.unwrapped.command_manager.get_command("ee_1_pose")
        desired_2 = env.unwrapped.command_manager.get_command("ee_2_pose")
        action = zero_dual_pose_action(env)
        action[:, arm_1] = desired_1[:, :7]
        action[:, arm_2] = desired_2[:, :7]
        return action

    def _command_error(self, env: Any) -> torch.Tensor:
        ee_1_pos, _ = frame_pose_b(env, "robot_1", "ee_frame")
        ee_2_pos, _ = frame_pose_b(env, "robot_2", "ee_2_frame")
        desired_1 = env.unwrapped.command_manager.get_command("ee_1_pose")
        desired_2 = env.unwrapped.command_manager.get_command("ee_2_pose")
        error_1 = torch.linalg.norm(ee_1_pos - desired_1[:, :3], dim=-1)
        error_2 = torch.linalg.norm(ee_2_pos - desired_2[:, :3], dim=-1)
        return torch.maximum(error_1, error_2)
