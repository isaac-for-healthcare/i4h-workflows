# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Cartesian state machines share: arms, action columns and eased motion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


@dataclass(frozen=True)
class Arm:
    """One arm: its robot asset, tool-tip frame sensor, and action columns."""

    robot: str
    frame: str
    #: Columns of the absolute IK pose ``[x, y, z, qx, qy, qz, qw]`` in the action vector.
    pose: slice
    #: Column of the arm's binary jaw action, if it has one.
    jaw: int | None = None


def arms_of(env) -> tuple[Arm, ...]:
    """The env's absolute-IK arms in action order, each paired with its tool-tip frame.

    Single-arm scenes name their frame ``ee_frame``; the dual-PSM scene names the
    second ``ee_2_frame``.
    """
    manager = env.action_manager
    arms: list[Arm] = []
    jaws: dict[str, int] = {}
    start = 0
    for name in manager.active_terms:
        term = manager.get_term(name)
        width = term.action_dim
        asset = term.cfg.asset_name
        if isinstance(term, DifferentialInverseKinematicsAction):
            if term.cfg.controller.use_relative_mode:
                raise ValueError(f"action term {name!r} is relative; these controllers command absolute poses")
            frame = "ee_frame" if not arms else f"ee_{len(arms) + 1}_frame"
            arms.append(Arm(robot=asset, frame=frame, pose=slice(start, start + width)))
        elif width == 1:
            jaws[asset] = start
        start += width
    return tuple(Arm(arm.robot, arm.frame, arm.pose, jaws.get(arm.robot)) for arm in arms)


def smoothstep(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.clamp(0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def nlerp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Shortest-arc normalized interpolation between batched quaternions (any component order)."""
    b = torch.where((a * b).sum(-1, keepdim=True) < 0.0, -b, b)
    out = a * (1.0 - alpha) + b * alpha
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def ease_pose(
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothstep from start to goal; at the end, the goal quaternion exactly (its sign matters to DLS IK)."""
    eased = smoothstep(alpha).unsqueeze(-1)
    pos = start_pos + (goal_pos - start_pos) * eased
    quat = torch.where(eased >= 1.0, goal_quat, nlerp(start_quat, goal_quat, eased))
    return pos, quat
