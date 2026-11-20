# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference and exported liver policies for Arena's normal policy runner."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class LiverReferencePolicyCfg(PolicyCfg):
    pass


@register_policy
class LiverReferencePolicy(PolicyBase[LiverReferencePolicyCfg]):
    name = "i4h_liver_reference"

    def get_action(self, env, observation):
        host = env.unwrapped
        if host.liver.objective.objective != "full_cycle":
            raise ValueError("normal workflow validation requires the full-cycle objective")
        return torch.as_tensor(host.liver.teacher_action(), device=host.device)


@dataclass
class LiverExportedPolicyCfg(PolicyCfg):
    checkpoint: str = ""


@register_policy
class LiverExportedPolicy(PolicyBase[LiverExportedPolicyCfg]):
    name = "i4h_liver_exported"

    def __init__(self, config):
        super().__init__(config)
        self.actor = None

    def get_action(self, env, observation):
        host = env.unwrapped
        if host.liver.objective.objective != "full_cycle":
            raise ValueError("normal workflow validation requires the full-cycle objective")
        if self.actor is None:
            path = Path(self.config.checkpoint).expanduser()
            path = path / "policy.pt" if path.is_dir() else path
            contract = json.loads(path.with_name("contract.json").read_text())
            if contract["objective"] != "full_cycle" or contract["encoder"]["revision"] != "liver_57_v1":
                raise ValueError("full-cycle workflow requires a compatible liver export")
            self.actor = torch.jit.load(str(path), map_location=host.device).eval()
        with torch.inference_mode():
            action = self.actor(host.liver.observation())
        if action.shape != (host.num_envs, 4) or not torch.isfinite(action).all():
            raise ValueError(f"invalid liver actor output: {action.shape}")
        return action
