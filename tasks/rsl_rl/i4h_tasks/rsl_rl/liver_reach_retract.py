# SPDX-License-Identifier: Apache-2.0
"""Exported liver actor; reads the shared Arena encoder through SceneView."""

from pathlib import Path

import numpy as np

from i4h_common.world import apply_action
from i4h_engine.status import Status
from i4h_engine.task import Task


class LiverReachRetractPolicy(Task):
    requires = {"action_space": "liver_cartesian_delta", "dof": 4}

    def __init__(self, *, checkpoint=None, device="cuda:0", until=None, name=None):
        super().__init__(name=name)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device, self.until, self.actor = device, until, None

    def on_enter(self, ctx, inputs):
        import json

        import torch

        if self.checkpoint is None:
            raise ValueError("liver policy requires --checkpoint exported/policy.pt")
        path = self.checkpoint / "policy.pt" if self.checkpoint.is_dir() else self.checkpoint
        contract = json.loads(path.with_name("contract.json").read_text())
        if contract["objective"] != "full_cycle" or contract["encoder"]["revision"] != "liver_57_v1":
            raise ValueError("full-cycle Workflow cannot execute an incompatible curriculum export")
        self.actor = torch.jit.load(str(path), map_location=self.device).eval()

    def tick(self, ctx):
        import torch

        if ctx.scene.liver_success().all():
            return Status.SUCCESS
        with torch.inference_mode():
            action = self.actor(torch.as_tensor(ctx.scene.liver_observation(), device=self.device)).cpu().numpy()
        if action.shape != (ctx.num_envs, 4) or not np.isfinite(action).all():
            raise ValueError(f"invalid liver actor output: {action.shape}")
        apply_action(ctx.act, action)
        return Status.RUNNING
