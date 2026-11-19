# SPDX-License-Identifier: Apache-2.0
"""Virtual grasper over the numerical liver fixture, shared by Workflow and RL."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from i4h_arena.adapters.scene_view import ArenaSceneView
from i4h_arena.scenes.base import Scene, arena_builder_cfg
from i4h_common.paths import workflow_root
from i4h_common.types import JointState


class LiverSceneView(ArenaSceneView):
    def liver_observation(self):
        return self._env.unwrapped.liver.observation().detach().cpu().numpy()

    def liver_teacher_action(self):
        return self._env.unwrapped.liver.teacher_action()

    def liver_success(self):
        liver = self._env.unwrapped.liver
        if liver.objective.objective != "full_cycle":
            raise ValueError("normal Workflow success requires the full-cycle objective")
        return self.termination("success")


class LiverStateProvider:
    def __init__(self, env):
        self.env = env.unwrapped

    def joint_state(self):
        return JointState(
            pos=self.env.liver.held.detach().cpu().numpy(),
            vel=np.zeros((self.env.num_envs, 10)),
            names=("x", "y", "z", "qx", "qy", "qz", "qw", "grip", "energy", "fire"),
        )


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
        from i4h_arena.envcfg.liver_reach_retract import make_environment

        directory = os.environ.get("I4H_LIVER_SNAPSHOTS", str(Path(workflow_root()) / "outputs/liver_snapshots"))
        return make_environment(directory, os.environ.get("I4H_LIVER_OBJECTIVE", "full_cycle"))

    def gym_spec(self, config_transform=None):
        from i4h_arena.envcfg.liver_reach_retract import LiverBuilder

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

    def make_view(self, env):
        return LiverSceneView(
            env,
            objects=(),
            robots=("robot",),
            cameras=(),
            gripper=False,
            joint_state_providers={"robot": LiverStateProvider(env)},
        )

    def make_actuation(self, env, view=None):
        actuation = super().make_actuation(env, view)
        initial = np.zeros((env.unwrapped.num_envs, 4), dtype=np.float32)
        initial[:, 3] = -1.0  # normalized open grip during Workflow entry/hold ticks
        actuation.seed(initial)
        return actuation
