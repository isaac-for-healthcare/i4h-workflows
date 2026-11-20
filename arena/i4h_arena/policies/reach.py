# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule-based Cartesian reach for the dVRK PSM, dual-PSM and STAR scenes.

Every arm holds its measured tool-tip pose for ``pre_hold_s``, then eases to its sampled
target pose over ``duration_s`` and keeps commanding it. The scene's success term ends
the episode when every tool tip is within its radius of the target; the time-out ends
it otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_arena.envcfg._surgical import tcp_pose_in_root
from i4h_arena.policies._cartesian import arms_of, ease_pose


@dataclass
class CartesianReachPolicyCfg(PolicyCfg):
    pre_hold_s: float = 0.5
    """Time each arm holds its measured tool-tip pose before moving [s]."""

    duration_s: float = 1.5
    """Time each arm takes to ease onto its target [s]. STAR uses 0.01: its controller converges on its own."""


@register_policy
class CartesianReachPolicy(PolicyBase[CartesianReachPolicyCfg]):
    """Hold, then ease every arm's tool tip onto its sampled ``ee_pose`` (or ``ee_<n>_pose``) command."""

    name = "i4h_cartesian_reach"

    def __init__(self, config: CartesianReachPolicyCfg):
        super().__init__(config)
        self._step: torch.Tensor | None = None
        self._start: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._step is None:
            return
        if env_ids is None:
            self._step.zero_()
        else:
            self._step[env_ids] = 0

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        if self._step is None:
            self._step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        arms = arms_of(env)
        commands = ["ee_pose"] if len(arms) == 1 else [f"ee_{index + 1}_pose" for index in range(len(arms))]
        # The first step only snapshots the target, as the controller's Locate step did.
        pre_steps = 1 + round(self.config.pre_hold_s / env.step_dt)
        reach_steps = max(1, round(self.config.duration_s / env.step_dt))
        holding = (self._step < pre_steps).unsqueeze(-1)
        entering = self._step == pre_steps
        alpha = ((self._step - pre_steps + 1).float() / reach_steps).clamp(0.0, 1.0)

        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        for index, (arm, command_name) in enumerate(zip(arms, commands)):
            tcp_pos, tcp_quat = tcp_pose_in_root(env, arm.frame, arm.robot)
            if index not in self._start:
                self._start[index] = (tcp_pos.clone(), tcp_quat.clone())
            start_pos, start_quat = self._start[index]
            start_pos[entering] = tcp_pos[entering]
            start_quat[entering] = tcp_quat[entering]
            command = env.command_manager.get_command(command_name)
            pos, quat = ease_pose(start_pos, start_quat, command[:, :3], command[:, 3:7], alpha)
            action[:, arm.pose] = torch.where(holding, torch.cat([tcp_pos, tcp_quat], -1), torch.cat([pos, quat], -1))
        self._step += 1
        return action
