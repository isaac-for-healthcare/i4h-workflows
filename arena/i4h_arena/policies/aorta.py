# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qualified 115-command static-aorta sweep as an Arena policy."""

from dataclasses import dataclass

import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class AortaFluoroscopySweepPolicyCfg(PolicyCfg):
    pass


@register_policy
class AortaFluoroscopySweepPolicy(PolicyBase[AortaFluoroscopySweepPolicyCfg]):
    name = "i4h_aorta_fluoroscopy_sweep"
    phases = (
        (15, (0, 0, 0)),
        (30, (1, 0, 0)),
        (15, (0, 0.5, 0)),
        (15, (0, 0, 0.5)),
        (15, (0, 0, -0.5)),
        (10, (-1, 0, 0)),
        (15, (0, 0, 0)),
    )

    def __init__(self, config):
        super().__init__(config)
        self.ticks = None

    def get_action(self, env, observation):
        host = env.unwrapped
        if self.ticks is None:
            self.ticks = torch.zeros(host.num_envs, dtype=torch.long, device=host.device)
        action = torch.zeros((host.num_envs, 3), device=host.device)
        start = 0
        for count, value in self.phases:
            within = (self.ticks >= start) & (self.ticks < start + count)
            action[within] = torch.tensor(value, device=host.device, dtype=action.dtype)
            start += count
        self.ticks += 1
        return action

    def reset(self, env_ids=None):
        if self.ticks is not None:
            self.ticks[slice(None) if env_ids is None else env_ids] = 0
