# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Virtual grasper over the numerical liver fixture, shared by Workflow and RL."""

from __future__ import annotations

import os
from pathlib import Path

from i4h_arena.scenes.base import Scene, arena_builder_cfg
from i4h_common.paths import workflow_root


class LiverReachRetractScene(Scene):
    name = "liver_reach_retract"

    def configure_args(self, args):
        if getattr(args, "presets", None) not in (None, "newton"):
            raise ValueError("liver_reach_retract requires Newton")
        args.presets = "newton"
        args.solve_relations = False
        numerical = Path(workflow_root()) / "third_party/i4h-numerical-sim-7edecea/numerical_sim"
        os.environ.setdefault("I4H_DATA", str(numerical / "data"))
        super().configure_args(args)

    def build(self):
        self.configure_args(self.args)
        from i4h_arena.envcfg.liver_reach_retract.environment import make_environment

        directory = os.environ.get("I4H_LIVER_SNAPSHOTS", str(Path(workflow_root()) / "outputs/liver_snapshots"))
        return make_environment(directory, os.environ.get("I4H_LIVER_OBJECTIVE", "full_cycle"))

    def gym_spec(self, config_transform=None):
        from i4h_arena.envcfg.liver_reach_retract.environment import LiverBuilder

        builder = LiverBuilder(self.arena_environment(), arena_builder_cfg(self.args))
        cfg, runtime_kwargs = builder.build_env_cfg()
        self.configure_env_cfg(cfg)
        if config_transform is not None:
            cfg = config_transform(cfg)
        gym_id, cfg, _ = builder.build_registered(env_cfg=cfg, env_kwargs=runtime_kwargs)
        return gym_id, cfg

    def configure_env_cfg(self, cfg):
        from isaaclab.envs.common import ViewerCfg

        cfg.viewer = ViewerCfg(eye=(0.35, 0.3, -0.15), lookat=(0.04, 0.07, -0.51))
