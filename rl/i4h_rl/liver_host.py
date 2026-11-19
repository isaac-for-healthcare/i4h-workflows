# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the standard Isaac Lab app for offline liver qualification tools."""

from contextlib import contextmanager
from types import SimpleNamespace


def build_parser():
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--workflow", default="liver_reach_retract")
    parser.add_argument("--mode", default="idle")
    parser.add_argument("--envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7341)
    parser.set_defaults(presets="newton", solve_relations=False, no_cameras=True, episode_steps=0, rl_observations=True)
    return parser


@contextmanager
def launch_app(args):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)

    def make_env(scene):
        import gymnasium as gym

        def configure(cfg):
            cfg.recorders = None
            return cfg

        gym_id, cfg = scene.gym_spec(config_transform=configure)
        return gym.make(gym_id, cfg=cfg).unwrapped

    try:
        yield SimpleNamespace(app=launcher.app, make_env=make_env)
    finally:
        launcher.app.close()
