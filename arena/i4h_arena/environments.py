# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every i4h scene as an IsaacLab-Arena external environment.

Arena's runners load one external environment per run, named by its class path and
selected by its scene name::

    python -m isaaclab_arena.evaluation.policy_runner --policy_type zero_action --num_steps 100 \\
        --external_environment_class_path i4h_arena.environments:PsmReach psm_reach

Isaac Lab's scripts reach the same classes through Arena's registration callback::

    python third_party/IsaacLab-e3f7d9f/scripts/tools/replay_demos.py \\
        --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \\
        --external_environment_class_path i4h_arena.environments:PsmReach --task psm_reach \\
        --dataset_file demos.hdf5

The class for scene ``psm_reach`` is ``PsmReach``: the scene name in CamelCase.
"""

from __future__ import annotations

import argparse
from typing import Any

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from i4h_arena.scenes.base import load_scene, scene_class, scene_specs


class SceneEnvironment(ExampleEnvironmentBase):
    """An i4h scene, built by its :class:`~i4h_arena.scenes.base.Scene` from the parsed CLI."""

    #: The scene name, which is also the Arena environment name and the gym id.
    name: str = ""

    def get_env(self, args_cli: argparse.Namespace) -> Any:
        scene = load_scene(self.name, args_cli)
        arena_env = scene.arena_environment()
        tune = arena_env.env_cfg_callback

        # The workflow scene's own settings come last, as they did after the runner's build.
        # RL registers scenes through ``arena_environment`` directly and keeps the cfg without them.
        def env_cfg_callback(env_cfg: Any) -> Any:
            env_cfg = tune(env_cfg)
            scene.configure_env_cfg(env_cfg)
            # Time out after the scene manifest's step budget, the one its workflows were
            # validated against, counted in this env's own control steps. Isaac Lab rounds the
            # episode length up to whole steps, so half a step short lands exactly on the budget.
            steps = getattr(args_cli, "episode_steps", None) or scene.spec.max_steps
            env_cfg.episode_length_s = (steps - 0.5) * env_cfg.sim.dt * env_cfg.decimation
            if getattr(args_cli, "enable_cameras", False):
                # Render the reset scene before the first observation, as Isaac Lab's camera
                # tasks do; otherwise the first frames after a reset show the previous episode.
                env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 3)
            return env_cfg

        arena_env.env_cfg_callback = env_cfg_callback
        return arena_env

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        spec = scene_specs()[cls.name]
        parser.add_argument(
            "--episode_steps",
            type=int,
            default=None,
            help=f"Episode length in control steps (default {spec.max_steps}).",
        )
        # Arena's builder reads these two by name; its runners do not expose them.
        parser.add_argument(
            "--recorder_dataset_export_dir_path",
            default=None,
            help="Directory the recorded HDF5 dataset is written to (Arena default /tmp/isaaclab/logs).",
        )
        parser.add_argument(
            "--recorder_dataset_filename",
            default=None,
            help="Recorded dataset file name without .hdf5 (Arena default dataset_<time>_rank<r>).",
        )
        scene_class(cls.name).add_cli_args(parser)


def _class_name(scene: str) -> str:
    return "".join(part.capitalize() for part in scene.split("_"))


for _scene in scene_specs():
    globals()[_class_name(_scene)] = type(_class_name(_scene), (SceneEnvironment,), {"name": _scene})
