# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM teleop adapter: device factory + joint-action post-processing.

Wraps the SO-ARM specifics (joint clamping, so101_leader → sim joint remap)
behind the ``ActionPostprocess`` contract used by ``arena.teleop``.
"""

from __future__ import annotations

import torch
from common.config import get_robot_config

_SO101_CONFIG = get_robot_config("so101")
_ROBOT_ISAACLAB_JOINT_POS_LIMIT_RANGE = _SO101_CONFIG.isaaclab_joint_pos_limit_range


def make_teleop_interface(env, args_cli):
    if args_cli.teleop_device == "keyboard":
        from arena.teleop.devices import so101_keyboard

        return so101_keyboard.make_interface(env, args_cli)
    if args_cli.teleop_device == "so101_leader":
        from arena.teleop.devices import so101_leader

        return so101_leader.make_interface(env, args_cli)
    raise ValueError(f"unsupported teleop device for SO-ARM: {args_cli.teleop_device}")


def action_device_for_teleop(teleop_device: str | None) -> str:
    """Map a physical teleop input device to the sim action backend it drives."""
    if teleop_device == "keyboard":
        return "keyboard"
    if teleop_device == "so101_leader":
        return "joint_position"
    raise ValueError(f"unsupported teleop device for SO-ARM: {teleop_device}")


def soarm_action_postprocess(
    actions: torch.Tensor, leader_reference: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture the leader-arm reference (first call only) and clamp to joint limits."""
    if leader_reference is None:
        leader_reference = actions.detach().clone()
    return _clamp_joint_actions(actions), leader_reference


def soarm_leader_action_postprocess(
    actions: torch.Tensor,
    leader_reference: torch.Tensor | None,
    home_joint_pos_rad: list[float] | tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map physical SO-ARM leader joint readings into sim joint-position commands."""

    if leader_reference is None:
        leader_reference = actions.detach().clone()
    absolute_targets = soarm_leader_absolute_targets(actions, leader_reference, home_joint_pos_rad)
    return absolute_targets - _home_joint_positions(absolute_targets, home_joint_pos_rad), leader_reference


def soarm_leader_absolute_targets(
    actions: torch.Tensor,
    leader_reference: torch.Tensor,
    home_joint_pos_rad: list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Return absolute sim joint targets for syncing the robot to the leader."""
    from arena.teleop.devices.so101_leader import leader_joints_to_sim_joints

    return _clamp_joint_actions(leader_joints_to_sim_joints(actions, leader_reference, home_joint_pos_rad))


def _clamp_joint_actions(actions: torch.Tensor) -> torch.Tensor:
    limits = torch.deg2rad(
        torch.tensor(_ROBOT_ISAACLAB_JOINT_POS_LIMIT_RANGE, device=actions.device, dtype=actions.dtype)
    )
    return torch.clamp(actions, min=limits[:, 0], max=limits[:, 1])


def _home_joint_positions(
    actions: torch.Tensor,
    home_joint_pos_rad: list[float] | tuple[float, ...],
) -> torch.Tensor:
    return torch.tensor(home_joint_pos_rad, device=actions.device, dtype=actions.dtype).reshape(1, -1)
