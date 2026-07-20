# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared tensor/scene helpers for scripted state-machine rollouts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


def frame_pose_b(env: Any, robot_name: str, frame_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame target pose expressed in the robot base frame (position) and world (quat)."""
    from isaaclab.utils.math import subtract_frame_transforms

    scene = env.unwrapped.scene
    robot = scene[robot_name]
    frame = scene[frame_name]
    target_pos_w = frame.data.target_pos_w[..., 0, :].clone() - scene.env_origins
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        target_pos_w,
    )
    quat_w = frame.data.target_quat_w[..., 0, :].clone()
    return pos_b, quat_w


def object_pose_for_lift(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``object`` position in the robot base frame and its world orientation."""
    from isaaclab.utils.math import subtract_frame_transforms

    scene = env.unwrapped.scene
    robot = scene["robot"]
    object_data = scene["object"].data
    object_pos_w = object_data.root_pos_w - scene.env_origins
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        object_pos_w,
    )
    return object_pos_b, object_data.root_quat_w


def zero_pose_action(env: Any) -> torch.Tensor:
    """A neutral absolute-pose action (identity quat, gripper open)."""
    action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    action[:, 3] = 1.0
    if action.shape[-1] > 7:
        action[:, 7] = 1.0
    return action


def zero_dual_pose_action(env: Any) -> torch.Tensor:
    action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    arm_1_slice, gripper_1_index, arm_2_slice, gripper_2_index = dual_action_layout(action.shape[-1])
    action[:, arm_1_slice.start + 3] = 1.0
    action[:, arm_2_slice.start + 3] = 1.0
    if gripper_1_index is not None:
        action[:, gripper_1_index] = 1.0
    if gripper_2_index is not None:
        action[:, gripper_2_index] = 1.0
    return action


def dual_action_layout(action_dim: int) -> tuple[slice, int | None, slice, int | None]:
    if action_dim >= 16:
        return slice(0, 7), 7, slice(8, 15), 15
    return slice(0, 7), None, slice(7, 14), None


def set_gripper(action: torch.Tensor, value: float) -> None:
    if action.shape[-1] > 7:
        action[:, 7] = value


def step_dt(env: Any) -> float:
    return float(getattr(env.unwrapped, "step_dt", 1.0 / 30.0))


def controller_should_abort(controller: Any) -> bool:
    return bool(controller is not None and controller.should_abort())


@contextmanager
def disabled_termination(env: Any, term_name: str) -> Iterator[None]:
    """Temporarily replace a termination term with an always-False stub."""
    from isaaclab.managers import TerminationTermCfg

    base_env = env.unwrapped
    termination_manager = base_env.termination_manager
    if term_name not in termination_manager.active_terms:
        yield
        return

    original_cfg = termination_manager.get_term_cfg(term_name)
    original_env_cfg = getattr(base_env.cfg.terminations, term_name, None)
    false_cfg = TerminationTermCfg(func=_false_termination)
    termination_manager.set_term_cfg(term_name, false_cfg)
    setattr(base_env.cfg.terminations, term_name, false_cfg)
    try:
        yield
    finally:
        termination_manager.set_term_cfg(term_name, original_cfg)
        setattr(base_env.cfg.terminations, term_name, original_env_cfg)


def _false_termination(env: Any) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def should_stop(env: Any, app: Any, controller: Any, terminated: torch.Tensor, truncated: torch.Tensor) -> bool:
    done = torch.as_tensor(terminated, device=env.device) | torch.as_tensor(truncated, device=env.device)
    return bool(done.any().item()) or not app.is_running() or controller_should_abort(controller)
