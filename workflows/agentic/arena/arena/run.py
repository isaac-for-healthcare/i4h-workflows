# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic Arena CLI dispatcher."""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import time
import traceback
from types import SimpleNamespace

from arena.arena_config import get_arena_config
from arena.environments import ENVIRONMENTS, get_environment
from common.utils import nonnegative_int

logger = logging.getLogger("arena")


def _announce_bridge_ready(bridge_server) -> None:
    print("[agentic-arena] scene-edit bridge ready", flush=True)
    print(f"[agentic-arena] scene-edit bridge endpoint: {bridge_server.url}", flush=True)


def _render_keep_open(env, app) -> None:
    """Keep Kit responsive during non-edit keep-open without advancing physics."""
    app.update()


def _render_scene_only(env) -> bool:
    """Refresh scene render buffers without advancing the app/timeline."""
    sim = getattr(getattr(env, "unwrapped", env), "sim", None)
    render = getattr(sim, "render", None)
    if not callable(render):
        return False
    render()
    return True


def _edit_mode_can_pump_window(bridge_server) -> bool:
    """Return true when a throttled Kit update should not delay bridge work."""
    if bridge_server is None:
        return True
    status = bridge_server.status()
    return not status.get("busy") and int(status.get("queue_depth") or 0) == 0


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
    parser.add_argument("--bridge-port", type=int, default=None, help="Port for the scene-edit HTTP bridge.")
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
        self._abort_requested = False
        self._interrupt_count = 0

    def should_abort(self) -> bool:
        return self._abort_requested

    def request_abort(self) -> int:
        self._abort_requested = True
        self._interrupt_count += 1
        return self._interrupt_count

    def is_paused(self) -> bool:
        return False

    def episode_completed(self) -> None:
        self.completed += 1


def _install_signal_handlers(controller: _Controller):
    previous_handlers = {}

    def _handle_signal(signum, _frame) -> None:
        count = controller.request_abort()
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = str(signum)
        print(
            f"\n[agentic-arena] received {signame}; stopping after the current simulation step",
            file=sys.stderr,
            flush=True,
        )
        if count > 1 and signum == signal.SIGINT:
            raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_signal)

    def _restore() -> None:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return _restore


def _noop_restore_signal_handlers() -> None:
    pass


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
    arena_defaults = get_arena_config(env_cls.name)
    if args.bridge_port is None:
        args.bridge_port = arena_defaults.bridge_port or 8765
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
        env_default = arena_defaults.max_timesteps
        if env_default:
            args.max_timesteps = env_default

    controller = _Controller()
    env_instance = env_cls()
    env_instance.configure_args(args)

    app = None
    env = None
    bridge_server = None
    restore_signal_handlers = _noop_restore_signal_handlers
    try:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        app = AppLauncher(args).app
        restore_signal_handlers = _install_signal_handlers(controller)

        import gymnasium as gym  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

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
        if edit_mode:
            logger.info("bridge edit mode: entering keep-open loop without normal env run")
        else:
            env_instance.run(args, env, app, controller)
        if args.episodes > 0:
            print(f"[agentic-arena] run complete: {controller.completed}/{args.episodes} episodes succeeded")

        if args.keep_open:
            _target_dt = 1.0 / 60.0
            _edit_window_update_dt = 0.25
            _last_edit_window_update_at = 0.0
            _idle_action = None
            _idle_obs = None
            _use_idle_action = False
            logger.info("keep-open setup: app.is_running=%s", app.is_running())
            if args.bridge and env is not None:
                try:
                    _idle_obs, _ = env.reset()
                    if not edit_mode:
                        _idle_action = env_instance.build_idle_action(args, env, _idle_obs)
                        _use_idle_action = _idle_action is not None
                    logger.info(
                        "keep-open setup after reset: app.is_running=%s idle_action=%s edit_mode=%s",
                        app.is_running(),
                        "yes" if _use_idle_action else "no",
                        edit_mode,
                    )
                except Exception as _err:
                    logger.warning("could not build idle action for edit mode: %r", _err)
                    _idle_action = None
                    _idle_obs = None
                    _use_idle_action = False
            if args.bridge and bridge_server is not None:
                _announce_bridge_ready(bridge_server)
            if edit_mode:
                logger.info("bridge edit mode: keeping scene alive with throttled render-only updates")
            while app.is_running() and not controller.should_abort():
                _frame_start = time.monotonic()
                if bridge_server is not None:
                    bridge_server.pump()
                if _use_idle_action and _idle_obs is not None:
                    with torch.inference_mode():
                        _idle_action = env_instance.build_idle_action(args, env, _idle_obs)
                        if _idle_action is None:
                            _use_idle_action = False
                        else:
                            _step_result = env.step(_idle_action)
                            if isinstance(_step_result, tuple) and _step_result:
                                _idle_obs = _step_result[0]
                elif edit_mode:
                    # Refresh render buffers sparingly without calling app.update().
                    # A free-standing G1 can fall if Kit advances app/physics
                    # without a WBC hold action, while bridge camera/capture jobs
                    # still render on demand in bridge.py.
                    now = time.monotonic()
                    if (
                        now - _last_edit_window_update_at >= _edit_window_update_dt
                        and _edit_mode_can_pump_window(bridge_server)
                        and _render_scene_only(env)
                    ):
                        _last_edit_window_update_at = time.monotonic()
                else:
                    _render_keep_open(env, app)
                _sleep_for = _target_dt - (time.monotonic() - _frame_start)
                if _sleep_for > 0:
                    time.sleep(_sleep_for)
            logger.info(
                "keep-open loop ended: app.is_running=%s abort=%s",
                app.is_running(),
                controller.should_abort(),
            )
    except KeyboardInterrupt:
        controller.request_abort()
        print("[agentic-arena] interrupted; closing simulation app", file=sys.stderr, flush=True)
    except BaseException as exc:
        print(f"[agentic-arena] failed with {type(exc).__name__}: {exc!r}", file=sys.stderr, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        raise
    finally:
        if bridge_server is not None:
            logger.info("shutting down scene-edit bridge")
            bridge_server.shutdown()
        if env is not None:
            logger.info("closing env")
            env.close()
            print("[agentic-arena] env closed", flush=True)
        if app is not None:
            logger.info("closing simulation app")
            app.close()
            print("[agentic-arena] simulation app closed", flush=True)
        restore_signal_handlers()


if __name__ == "__main__":
    main()
