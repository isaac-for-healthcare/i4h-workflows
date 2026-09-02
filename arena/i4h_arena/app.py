# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim lifecycle.

Isolated here for one reason: importing ``isaacsim``/``isaaclab`` has global
side effects and must not happen until the workflow has already been resolved and
linted. Everything above this module stays importable without Isaac.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("i4h_arena.app")


@dataclass
class AppContext:
    app: Any
    args: argparse.Namespace

    def make_env(self, scene: Any) -> Any:
        """Build the gym env for ``scene``. Isaac is already running by now."""
        import gymnasium as gym  # noqa: PLC0415

        print("[arena] registering assets", flush=True)
        scene.register_assets()
        print("[arena] building arena env", flush=True)
        gym_id, env_cfg = scene.gym_spec()
        print(f"[arena] built {gym_id}", flush=True)
        # The workflow decides when an episode ends, so the cfg's own time-out has to
        # go: it fires on the first step and the runner never gets to tick.
        if getattr(env_cfg, "terminations", None) is not None and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        logger.info("creating env %s for scene %s", gym_id, scene.name)
        env = gym.make(gym_id, cfg=env_cfg).unwrapped
        # Apply the task viewer after the visualizer is fully initialized so
        # the scene opens with its configured default camera.
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg  # noqa: PLC0415

        reapply_viewer_cfg(env)
        print("[arena] gym env ready", flush=True)
        return env


@contextlib.contextmanager
def launch_app(args: argparse.Namespace) -> Iterator[AppContext]:
    """Start ``AppLauncher`` and shut it down on the way out."""
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    # Built through AppLauncher's own parser rather than by hand. A partial
    # Namespace makes it fall back to its defaults and to the HEADLESS /
    # ENABLE_CAMERAS env vars, which silently selected the
    # `headless.rendering` kit config — Isaac then started with --no-window
    # while we logged headless=False.
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args = parser.parse_args([])
    launcher_args.headless = bool(args.headless)
    launcher_args.device = str(args.device)
    launcher_args.num_envs = int(args.num_envs)
    launcher_args.enable_cameras = bool(getattr(args, "enable_cameras", not args.no_cameras))
    # Interactive runs require the Kit visualizer.
    launcher_args.visualizer = None if args.headless else ["kit"]
    # ExplicitAction pairs each option with a `<name>_explicit` flag; without it
    # AppLauncher treats the value as a default and falls back to headless.
    launcher_args.visualizer_explicit = not args.headless
    if args.python_server:
        launcher_args.kit_args = " ".join(
            part
            for part in (
                getattr(launcher_args, "kit_args", ""),
                "--enable isaacsim.code_editor.python_server",
                "--enable isaacsim.test.utils",
            )
            if part
        )
    logger.info(
        "launching Isaac Sim (headless=%s device=%s envs=%s cameras=%s python_server=%s)",
        launcher_args.headless,
        launcher_args.device,
        launcher_args.num_envs,
        launcher_args.enable_cameras,
        args.python_server,
    )
    launcher = AppLauncher(launcher_args)
    app = launcher.app
    try:
        yield AppContext(app=app, args=args)
    except BaseException:
        # The active ovphysx backend registers an atexit os._exit(0) workaround
        # for its native-library teardown. Exit explicitly here so that handler
        # cannot turn a simulator error into a successful shell status.
        logger.exception("arena failed before Isaac Sim was closed")
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        logger.info("closing Isaac Sim")
        with contextlib.suppress(Exception):
            app.close()
