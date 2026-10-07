# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register an i4h Scene with Isaac Lab's stock RSL-RL scripts."""

from __future__ import annotations

import argparse

_APP_LAUNCHER = None


def _simulation_is_running() -> bool:
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except (ImportError, RuntimeError):
        return False


def environment_registration_callback() -> list[str]:
    """Build the requested workflow-owned Scene and register its Gym ID."""
    global _APP_LAUNCHER

    from isaaclab.app import AppLauncher
    from isaaclab_arena.cli.isaaclab_arena_cli import add_isaac_lab_cli_args, add_isaaclab_arena_cli_args

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rl_training_mode", action="store_true")
    # The Arena CLI has no notion of a patient twin, but the medical scenes
    # resolve their route, target and lumen widths from one, and return None for
    # every reward and termination config without it. Left defaulting to None so
    # such a scene fails on the missing twin rather than quietly training
    # against a vessel that is not there.
    parser.add_argument("--patient-twin", dest="patient_twin", default=None)
    # Both default to None so the scene's own resolver picks the backend from
    # whether a twin is present, exactly as it does under the workflow CLI.
    parser.add_argument("--fluoro-backend", dest="fluoro_backend", default=None)
    parser.add_argument("--fluoro-device", dest="fluoro_device", default=None)
    args, remaining_args = parser.parse_known_args()

    if not _simulation_is_running():
        _APP_LAUNCHER = AppLauncher(args)

    # Imports that depend on Kit must remain below AppLauncher.  RSL-RL loads
    # ``isaaclab.utils.configclass`` as a submodule first, which can leave the
    # package attribute pointing at that module instead of the decorator.
    import isaaclab.utils as isaaclab_utils
    from isaaclab.utils.configclass import configclass

    isaaclab_utils.configclass = configclass

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from i4h_arena.scenes.base import load_scene

    args.episode_steps = 0
    args.no_cameras = True
    args.rl_observations = True
    scene = load_scene(args.task, args)
    scene.configure_args(args)
    ArenaEnvBuilder(scene.build(), args).build_registered()
    return remaining_args
