# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-channel normalized catheter control with a fixed synthetic projection."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from i4h_arena.medical.catheter import CatheterState
from i4h_arena.scenes.base import Scene, arena_builder_cfg
from i4h_common.paths import workflow_root
from i4h_common.types import JointState

JOINT_NAMES = ("insertion_m", "rotation_rad", "tip_bend_rad")


class AortaStateProvider:
    """Read synchronized live component views; retain no particle buffer across rebuilds."""

    def __init__(self, env):
        self.env = env.unwrapped

    def points(self):
        self.env.i4h.sync_views()
        return self.env.i4h.positions("catheter").detach().cpu().numpy().reshape(self.env.num_envs, -1, 3).copy()

    def targets(self):
        self.env.i4h.sync_views()
        products = self.env.i4h.state()
        targets = [value for key, value in products.items() if key.endswith(".targets")]
        if len(targets) != 1:
            raise RuntimeError(f"Expected one catheter target product, received {list(products)}")
        return targets[0].detach().cpu().numpy().copy()

    def snapshot(self, num_envs):
        points = self.points()
        if num_envs != len(points):
            raise ValueError(f"Expected {len(points)} environments, received {num_envs}")
        return CatheterState(points, np.full(num_envs, points.shape[1], dtype=np.int32), radius_m=0.02)

    def joint_state(self):
        targets = self.targets()
        return JointState(pos=targets, vel=np.zeros_like(targets), names=JOINT_NAMES)

    def diagnostics(self):
        points = self.points()[0]
        return {"root_world_m": points[0], "tip_world_m": points[-1], "command_targets": self.targets()[0]}


class EndoluminalAortaFluoroscopyScene(Scene):
    name = "endoluminal_aorta_fluoroscopy"

    def configure_args(self, args):
        if args.num_envs != 1:
            raise ValueError("endoluminal_aorta_fluoroscopy requires exactly one environment")
        if args.presets not in (None, "newton"):
            raise ValueError("endoluminal_aorta_fluoroscopy requires the Newton backend")
        args.presets = "newton"
        args.solve_relations = False
        numerical = Path(workflow_root()) / "third_party/i4h-numerical-sim-7edecea/numerical_sim"
        os.environ["I4H_DATA"] = str(numerical / "data")
        super().configure_args(args)

    def build(self):
        self.configure_args(self.args)
        from i4h_arena.envcfg.endoluminal_aorta_fluoroscopy import make_environment

        return make_environment()

    def gym_spec(self, config_transform=None):
        from i4h_arena.envcfg.endoluminal_aorta_fluoroscopy import AortaArenaBuilder

        builder = AortaArenaBuilder(self.arena_environment(), arena_builder_cfg(self.args))
        cfg, runtime_kwargs = builder.build_env_cfg()
        self.configure_env_cfg(cfg)
        if config_transform is not None:
            cfg = config_transform(cfg)
        gym_id, cfg, _runtime_kwargs = builder.build_registered(env_cfg=cfg, env_kwargs=runtime_kwargs)
        return gym_id, cfg

    def default_sensor_views(self):
        return ("fluoroscopy",)

    def sensor_view_titles(self):
        return {"fluoroscopy": "Static aorta: fixed XY fluoroscopy"}

    def sensor_view_outputs(self):
        return {"fluoroscopy": (("Guidance", "guidance"), ("Raw fluoroscopy", "rgb"))}
