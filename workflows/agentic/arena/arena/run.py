# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic Arena CLI dispatcher."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import traceback
from types import SimpleNamespace

from arena.arena_config import get_arena_config
from arena.environments import ENVIRONMENTS, get_environment
from common.utils import nonnegative_int

logger = logging.getLogger("arena")


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", default="scissor_pick_and_place")
    parser.add_argument("--list-envs", action="store_true")
    return parser


def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--num-steps",
        type=nonnegative_int,
        default=100,
        help="Number of zero-action simulation steps to run.",
    )
    parser.add_argument("--episodes", type=nonnegative_int, default=0)
    parser.add_argument("--max-timesteps", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_spacing", type=float, default=4.0)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument(
        "--bridge",
        dest="bridge",
        action="store_true",
        help=("Open the scene in edit mode and start the local scene-edit HTTP bridge."),
    )
    parser.add_argument("--bridge-host", default="127.0.0.1", help="Host interface for the scene-edit HTTP bridge.")
    parser.add_argument("--bridge-port", type=int, default=8765, help="Port for the scene-edit HTTP bridge.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--disable-cameras", action="store_true")
    parser.add_argument("--mimic", action="store_true", default=False)
    parser.add_argument(
        "--dump-scene",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "During zero-action runs, save camera frames and scene poses for offline inspection. "
            "If DIR is omitted, writes under runs/<env>/scene_dumps."
        ),
    )
    parser.add_argument(
        "--dump-scene-cameras",
        default=None,
        help="Comma-separated camera names to include in --dump-scene. Defaults to all discovered cameras.",
    )
    parser.add_argument(
        "--dump-scene-entities",
        default=None,
        help="Comma-separated scene entity names to include in scene pose output. Defaults to discovered scene entities.",
    )
    parser.add_argument(
        "--dump-scene-steps",
        default="0,1,10,20,30",
        help="Comma-separated zero-action step indices to include in frame and pose output.",
    )


def _full_parser(env_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run agentic Arena env={env_cls.name}.")
    parser.add_argument("--env", default=env_cls.name)
    parser.add_argument("--list-envs", action="store_true")
    _shared_args(parser)
    env_cls.add_cli_args(parser)
    _add_app_launcher_args(parser)
    return parser


def _add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
        return
    AppLauncher.add_app_launcher_args(parser)


class _Controller:
    def __init__(self) -> None:
        self.completed = 0

    def should_abort(self) -> bool:
        return False

    def is_paused(self) -> bool:
        return False

    def episode_completed(self) -> None:
        self.completed += 1


def _print_environment_specs() -> None:
    width = max(len(name) for name in ENVIRONMENTS)
    print("Available environments:")
    for name, env_cls in ENVIRONMENTS.items():
        print(f"  {name:<{width}}  {env_cls.description}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")

    if "--list-envs" in sys.argv[1:]:
        _print_environment_specs()
        return
    if "--dry-run" in sys.argv[1:]:
        base = _base_parser().parse_known_args()[0]
        env_cls = get_environment(base.env)
        print(f"[agentic-arena] dry run ok: env={env_cls.name}")
        return

    base, remaining = _base_parser().parse_known_args()
    env_cls = get_environment(base.env)
    args = _full_parser(env_cls).parse_args(remaining)
    args.env = env_cls.name
    no_run_mode = (
        args.episodes == 0 and not getattr(args, "teleop", False) and not getattr(args, "replay_dataset_path", None)
    )
    edit_mode = no_run_mode and args.bridge
    if edit_mode:
        args.num_steps = 0
        args.keep_open = True
        args.bridge = True
        logger.info(
            "bridge edit mode: opening env=%s with scene-edit HTTP bridge at http://%s:%s",
            args.env,
            args.bridge_host,
            args.bridge_port,
        )
    if not args.max_timesteps:
        env_default = get_arena_config(env_cls.name).max_timesteps
        if env_default:
            args.max_timesteps = env_default

    env_instance = env_cls()
    env_instance.configure_args(args)

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    import gymnasium as gym  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    env = None
    bridge_server = None
    try:
        env_instance.register_assets()
        env_name, env_cfg = env_instance.build(args)
        if args.seed is not None:
            env_cfg.seed = args.seed
        if not (getattr(args, "record", False) or getattr(args, "record_to", None)):
            env_cfg.recorders = None
        if (args.episodes > 0 or getattr(args, "teleop", False) or getattr(args, "bridge", False)) and hasattr(
            env_cfg.terminations, "time_out"
        ):
            env_cfg.terminations.time_out = None
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        if args.seed is not None:
            env.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        controller = _Controller()
        if args.bridge:
            from arena.bridge import BridgeServer  # noqa: PLC0415

            bridge_ctx = SimpleNamespace(
                app=app,
                args=args,
                controller=controller,
                env=env,
                env_id=args.env,
                env_instance=env_instance,
            )
            bridge_server = BridgeServer(bridge_ctx, host=args.bridge_host, port=args.bridge_port)
            bridge_server.start()
            print("[agentic-arena] scene-edit bridge ready", flush=True)
            print(f"[agentic-arena] scene-edit bridge endpoint: {bridge_server.url}", flush=True)
        env_instance.run(args, env, app, controller)
        if args.episodes > 0:
            print(f"[agentic-arena] run complete: {controller.completed}/{args.episodes} episodes succeeded")

        if args.keep_open:
            _target_dt = 1.0 / 60.0
            _idle_action = None
            if args.bridge and env is not None:
                try:
                    _obs, _ = env.reset()
                    _idle_action = env_instance.build_idle_action(args, env, _obs)
                except Exception as _err:
                    logger.warning("could not build idle action for edit mode: %r", _err)
                    _idle_action = None
            while app.is_running():
                _frame_start = time.monotonic()
                if bridge_server is not None:
                    bridge_server.pump()
                if _idle_action is not None:
                    with torch.inference_mode():
                        env.step(_idle_action)
                else:
                    app.update()
                _sleep_for = _target_dt - (time.monotonic() - _frame_start)
                if _sleep_for > 0:
                    time.sleep(_sleep_for)
    except BaseException as exc:
        print(f"[agentic-arena] failed with {type(exc).__name__}: {exc!r}", file=sys.stderr, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        raise
    finally:
        if bridge_server is not None:
            bridge_server.shutdown()
        if env is not None:
            env.close()
            print("[agentic-arena] env closed", flush=True)
        app.close()
        print("[agentic-arena] simulation app closed", flush=True)


if __name__ == "__main__":
    main()
