# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena entry point.

Normally reached through ``run.sh``, which lints the workflow and starts any needed
backends first. Invoking this directly is supported and skips both.

The mode is resolved while the workflow is built, before Isaac starts.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

# Light by construction: it defers torch to the call that needs it, so the
# parser stays importable without Isaac or a GPU.
from i4h_arena.profiling import DEFAULT_WARMUP_STEPS, CaptureRange

logger = logging.getLogger("arena")

# isaacsim.code_editor.python_server's fixed port; local-agent/bridge.sh and the scene-edit skill use it too.
PYTHON_SERVER_PORT = 8226


def _step_cap(value: str) -> int:
    """``--episode-steps``: 0 keeps the scene default; a negative cap is refused."""
    try:
        steps = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if steps < 0:
        raise argparse.ArgumentTypeError(f"must be 0 (the scene default) or a positive cap, got {steps}")
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arena", description="Run a workflow in Isaac Sim.")

    parser.add_argument("--workflow", required=True, help="workflow name (see ./run.sh list)")
    parser.add_argument("--mode", default=None, help="workflow run mode; defaults to idle")

    episodes = parser.add_argument_group("episodes")
    episodes.add_argument("--episodes", type=int, default=1)
    episodes.add_argument(
        "--episode-steps",
        "--timesteps",
        dest="episode_steps",
        type=_step_cap,
        default=0,
        help="cap per episode; 0 uses the workflow/scene default",
    )
    episodes.add_argument("--seed", type=int, default=None)

    recording = parser.add_argument_group("recording")
    recording.add_argument(
        "--record", "--record-to", dest="record", metavar="PATH", default=None, help="write episodes to this HDF5 file"
    )
    recording.add_argument(
        "--record-failures",
        "--save-all-episodes",
        dest="record_failures",
        action="store_true",
        help="keep failed episodes too",
    )
    recording.add_argument(
        "--attempts",
        "--max-attempts",
        dest="attempts",
        type=int,
        default=1,
        help="attempts per episode before moving on",
    )

    policy = parser.add_argument_group("policy")
    policy.add_argument("--checkpoint", default=None, help="override the manifest's model repo/path")
    policy.add_argument("--prompt", default=None, help="override the task description")

    teleop = parser.add_argument_group("teleop")
    teleop.add_argument("--teleop-device", default=None)
    teleop.add_argument("--teleop-port", default="/dev/ttyACM1")
    teleop.add_argument("--teleop-sensitivity", type=float, default=1.0)
    teleop.add_argument("--teleop-base-height", type=float, default=0.75)
    teleop.add_argument("--teleop-recalibrate", action="store_true")
    teleop.add_argument("--cloudxr-env", default=None, help="CloudXR environment file for an embedded XR runtime")
    teleop.add_argument("--auto-launch-cloudxr", action="store_true", help="launch CloudXR from --cloudxr-env")

    replay = parser.add_argument_group("replay")
    replay.add_argument("--dataset", default=None, help="recording to replay")
    replay.add_argument(
        "--episode",
        "--episode-index",
        dest="episode",
        default="0",
        help="comma-separated episode indices",
    )

    simulation = parser.add_argument_group("simulation")
    simulation.add_argument("--envs", type=int, default=1, dest="num_envs")
    # Read by isaaclab_arena.ArenaEnvBuilder off the same namespace. Declared
    # here because it reads them as plain attributes and raises AttributeError
    # rather than defaulting when they are absent.
    simulation.add_argument(
        "--no-solve-relations",
        dest="solve_relations",
        action="store_false",
        default=True,
        help="disable Arena spatial relation solving",
    )
    simulation.add_argument("--disable-fabric", dest="disable_fabric", action="store_true")
    simulation.add_argument("--mimic", action="store_true", help="build the Arena mimic env cfg")
    # The scene selects its default backend unless explicitly overridden.
    simulation.add_argument(
        "--presets", choices=("physx", "newton"), default=None, help="override scene physics backend"
    )
    simulation.add_argument("--spacing", type=float, default=4.0, dest="env_spacing")
    simulation.add_argument("--no-cameras", action="store_true", help="disable scene cameras and record without images")
    simulation.add_argument("--headless", action="store_true")
    simulation.add_argument("--device", default="cuda:0", help="CUDA device for the simulation")
    simulation.add_argument(
        "--fluoro-backend",
        choices=("synthetic", "slang"),
        default=None,
        help="override fluoroscopy backend (default: Slang with --patient-twin, otherwise synthetic)",
    )
    simulation.add_argument(
        "--fluoro-device",
        choices=("auto", "cuda", "vulkan"),
        default="auto",
        help="Slang device used by the patient-backed fluoroscopy renderer "
        "(default: auto, CUDA when Slang can create a CUDA device, otherwise Vulkan)",
    )
    simulation.add_argument(
        "--hu-to-mu",
        choices=("interventional", "linear"),
        default=None,
        help="HU-to-attenuation preset for patient twins (default: interventional)",
    )
    simulation.add_argument(
        "--patient-twin",
        default=None,
        help="patient-twin YAML manifest used by patient-specific medical sensors",
    )
    simulation.add_argument(
        "--view-sensor",
        action="append",
        default=None,
        metavar="NAME",
        help="open a live Kit image window for a declared camera-compatible sensor; repeat for multiple sensors",
    )
    simulation.add_argument(
        "--no-sensor-view",
        action="store_true",
        help="disable scene-selected live sensor image windows",
    )
    simulation.add_argument(
        "--idle-seconds",
        type=float,
        default=60.0,
        help="idle-mode duration; --live supplies a long duration and relies on explicit stop",
    )
    simulation.add_argument(
        "--python-server",
        action="store_true",
        help="enable the Isaac Sim Python bridge on 127.0.0.1:8226 for live authoring",
    )

    diagnostics = parser.add_argument_group("diagnostics")
    diagnostics.add_argument(
        "--nsys-range",
        action="store_true",
        help="bracket steady-state stepping with cudaProfilerStart/Stop and mark tick/step/record as NVTX "
        "ranges, for nsys profile --capture-range=cudaProfilerApi",
    )
    diagnostics.add_argument(
        "--nsys-warmup",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        metavar="N",
        help=f"steps to skip before --nsys-range opens the capture (default {DEFAULT_WARMUP_STEPS})",
    )
    diagnostics.add_argument("--dry-run", action="store_true", help="resolve and lint, then report without launching")
    diagnostics.add_argument("--verbose", action="store_true")
    diagnostics.add_argument("--run-id", default=None)
    diagnostics.add_argument("--namespace", default=None, help="zenoh key prefix; defaults to the workflow name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    # Imported before Isaac so a bad workflow fails in milliseconds, not after Kit boots.
    from i4h_engine.lint import lint_workflow
    from i4h_engine.loader import apply_overrides, resolve_workflow
    from i4h_engine.registry import default_registry

    workflow = _build(args, resolve_workflow)
    scene_spec = default_registry().scene(workflow.scene).for_mode(workflow.mode)
    declared_cap = scene_spec.max_steps
    if args.episode_steps > declared_cap:
        parser.error(
            f"--episode-steps cannot exceed {declared_cap} for {args.workflow}; "
            "the validated per-environment cap may only be lowered"
        )
    if args.nsys_warmup < 0:
        parser.error(f"--nsys-warmup cannot be negative, got {args.nsys_warmup}")
    _check_view_sensors(parser, args, scene_spec.cameras)
    overridden = apply_overrides(
        workflow.graph,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        model_device=args.device,
    )
    if overridden:
        model_device_applied = any(
            node.spec is not None
            and node.spec.runtime == "inprocess"
            and bool(node.spec.model)
            and node.params.get("device") == args.device
            for node in workflow.graph.nodes
        )
        what = ", ".join(
            x
            for x in (
                ("checkpoint" if args.checkpoint else ""),
                ("prompt" if args.prompt else ""),
                ("model device" if model_device_applied else ""),
            )
            if x
        )
        print(f"override {what} on: {', '.join(overridden)}")
    report = lint_workflow(workflow, default_registry())
    print(report.render())
    if not report.ok:
        return 1
    if args.dry_run:
        print(f"dry-run: would launch scene {workflow.scene!r} for {args.episodes} episode(s)")
        return 0

    return _launch(args, workflow)


def _check_view_sensors(parser: argparse.ArgumentParser, args: argparse.Namespace, cameras) -> None:
    """Refuse an undeclared ``--view-sensor`` whatever the display flags, before Kit boots."""
    if not args.view_sensor:
        return
    unknown = sorted(set(args.view_sensor) - set(cameras))
    if unknown:
        parser.error(
            f"cannot view undeclared scene sensor(s) {unknown}; available camera-compatible sensors: {list(cameras)}"
        )
    disabled_by = [
        flag
        for flag, value in (
            ("--headless", args.headless),
            ("--no-sensor-view", args.no_sensor_view),
            ("--no-cameras", args.no_cameras),
        )
        if value
    ]
    if disabled_by:
        print(f"[arena] --view-sensor is ignored: {', '.join(disabled_by)} opens no sensor windows", flush=True)


def _check_replay_dataset(args: argparse.Namespace, workflow) -> None:
    """Name a missing recording before Kit boots, as the replay task would after it."""
    if workflow.mode != "replay":
        return
    from pathlib import Path

    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(f"no recording at {dataset}")


def _check_python_server_port(args: argparse.Namespace) -> None:
    """Refuse a second live bridge: two servers on 127.0.0.1:8226 split live edits between sessions."""
    if not args.python_server:
        return
    import socket

    try:
        with socket.create_connection(("127.0.0.1", PYTHON_SERVER_PORT), timeout=0.5):
            pass
    except OSError:
        return
    raise SystemExit(
        f"--python-server: 127.0.0.1:{PYTHON_SERVER_PORT} is already in use, most likely by another live session; "
        "stop it first (./local-agent/bridge.sh stop <workflow> or ./stop.sh all)"
    )


def _python_server_lock_path() -> Path:
    """The session lock, inside a private per-user directory.

    The port is machine-wide, so the lock is too. It is per user because the bridge listens with
    SO_REUSEPORT, which the kernel already refuses across users. The directory is created 0700 and
    checked on every use, so another local user cannot plant the lock path in a shared temp dir.
    """
    base = Path(tempfile.gettempdir()) / f"i4h-arena-{os.getuid()}"
    with contextlib.suppress(FileExistsError):
        base.mkdir(mode=0o700)
    info = base.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit(
            f"--python-server: {base} must be a directory owned by you with mode 0700 (it holds the live-session "
            "lock); remove it or fix it, then retry"
        )
    return base / f"python-server-{PYTHON_SERVER_PORT}.lock"


def _open_python_server_lock(path: Path) -> int:
    """Open the lock file without following a link or blocking on a FIFO, and insist it is ours."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        os.close(fd)
        raise SystemExit(f"--python-server: {path} is not a regular lock file owned by you; remove it and retry")
    return fd


@contextlib.contextmanager
def _python_server_guard(args: argparse.Namespace) -> Iterator[None]:
    """Own the live bridge port from before Kit boots until this session ends.

    The bridge only listens once Kit has finished booting, and it binds with SO_REUSEPORT, so a
    second launch inside that window passes the port check and then shares port 8226 without any
    error, splitting live edits between the two sessions. An exclusive lock held for the whole
    session closes the window. The kernel drops it when the process exits, however it exits, so
    no stale lock is left behind. The port check stays for sessions started outside this launcher.
    """
    if not args.python_server:
        yield
        return
    try:
        import fcntl
    except ImportError:  # no flock (Windows): the port check is the only guard
        _check_python_server_port(args)
        yield
        return

    fd = _open_python_server_lock(_python_server_lock_path())
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"--python-server: another live session is starting or running and owns "
                f"127.0.0.1:{PYTHON_SERVER_PORT}; stop it first "
                "(./local-agent/bridge.sh stop <workflow> or ./stop.sh all)"
            ) from None
        _check_python_server_port(args)
        yield
    finally:
        os.close(fd)


def _build(args: argparse.Namespace, resolve_workflow) -> object:
    kwargs: dict[str, object] = {}
    mode = args.mode or ""
    if mode == "replay":
        if not args.dataset:
            raise SystemExit("--replay needs a dataset path")
        kwargs = {"dataset": args.dataset, "episode": int(str(args.episode).split(",")[0])}
    elif mode == "teleop":
        kwargs = {
            "port": args.teleop_port,
            "sensitivity": args.teleop_sensitivity,
            "base_height": args.teleop_base_height,
            "recalibrate": args.teleop_recalibrate,
            "cloudxr_env": args.cloudxr_env,
            "auto_launch_cloudxr": args.auto_launch_cloudxr,
        }
        if args.teleop_device:
            kwargs["device"] = args.teleop_device
    elif mode == "idle":
        kwargs = {"seconds": args.idle_seconds}
    return resolve_workflow(args.workflow, args.mode or None, **kwargs)


def _resolve_fluoroscopy_device(args: argparse.Namespace) -> None:
    """Replace ``auto`` with a concrete Slang device, deciding before Kit boots.

    Vulkan reads every frame back through the host, which costs a patient frame about twice
    the CUDA time, so ``auto`` takes CUDA whenever Slang can create it and falls back to
    Vulkan on hosts without a usable CUDA Toolkit. An explicit ``cuda`` still fails in
    seconds rather than after Kit boots.
    """
    from i4h_arena.scenes.endoluminal_navigation import resolve_fluoroscopy_backend

    if resolve_fluoroscopy_backend(args.fluoro_backend, args.patient_twin) != "slang":
        if args.fluoro_device == "auto":
            args.fluoro_device = "vulkan"
        return
    if args.fluoro_device == "vulkan":
        return
    from i4h_arena.medical.slang_cuda import check_slang_cuda_device

    failure = check_slang_cuda_device()
    if args.fluoro_device == "cuda":
        if failure is not None:
            raise SystemExit(f"--fluoro-device cuda: {failure}")
        return
    if failure is None:
        args.fluoro_device = "cuda"
    else:
        args.fluoro_device = "vulkan"
        print(f"[arena] fluoroscopy renders on Vulkan, which is slower: {failure}", flush=True)
    print(f"[arena] fluoroscopy device: {args.fluoro_device}", flush=True)


def _replay_initial_state(args: argparse.Namespace, workflow) -> dict | None:
    """The scene state the replayed episode was recorded from, if stored for this workflow's scene."""
    if workflow.mode != "replay":
        return None
    from i4h_arena.recording.scene_state import load_initial_state

    return load_initial_state(args.dataset, int(str(args.episode).split(",")[0]), workflow.scene)


def _launch(args: argparse.Namespace, workflow) -> int:
    # Checks that only need the filesystem or a socket answer in seconds, not after Kit boots.
    _check_replay_dataset(args, workflow)
    with _python_server_guard(args):
        return _run_session(args, workflow)


def _run_session(args: argparse.Namespace, workflow) -> int:
    # This function-local import must remain below workflow resolution.
    from i4h_arena.app import launch_app
    from i4h_arena.scenes.base import load_scene

    # Scenes may require XR or early Pinocchio loading. Configure the launcher
    # before Kit starts, keeping asset registration inside AppContext.make_env.
    args.mode = workflow.mode
    scene = load_scene(workflow.scene, args, register_assets=False)
    scene.configure_args(args)
    _resolve_fluoroscopy_device(args)
    with launch_app(args) as app_ctx:
        from i4h_arena.io.publishers import ScenePublisher
        from i4h_arena.recording.hdf5 import EpisodeRecorder
        from i4h_arena.runner import SimulationRunner
        from i4h_common.bus.keys import Keys
        from i4h_common.bus.zenoh_bus import open_zenoh_bus

        # A SystemExit here is not ours: IsaacLab-Arena and the asset SDK call
        # sys.exit() on failures they consider fatal, which looks like a clean
        # shutdown from the outside — Isaac closes, the runner never ticks, and
        # the process reports 0. Name it so the next reader is not hunting.
        try:
            env = app_ctx.make_env(scene)
        except SystemExit as exit_request:
            raise RuntimeError(
                f"building scene {workflow.scene!r} exited with code {exit_request.code}; "
                "the asset or env builder aborted"
            ) from exit_request

        bus = None
        needs_bus = any((node.spec is not None and node.spec.runtime == "remote") for node in workflow.graph.nodes)
        if needs_bus:
            bus = open_zenoh_bus()

        cameras = () if args.no_cameras else scene.spec.cameras
        sensor_views = ()
        if not args.headless and not args.no_sensor_view and not args.no_cameras:
            sensor_views = tuple(args.view_sensor or scene.default_sensor_views())
            unknown_views = sorted(set(sensor_views) - set(cameras))
            if unknown_views:
                raise ValueError(
                    f"cannot view undeclared scene sensor(s) {unknown_views}; "
                    f"available camera-compatible sensors: {list(cameras)}"
                )
        recorder = EpisodeRecorder(args.record, workflow=workflow, cameras=cameras) if args.record else None
        initial_state = _replay_initial_state(args, workflow)
        keys = Keys(args.namespace or workflow.name)
        publisher = ScenePublisher(bus, keys, cameras=cameras) if bus is not None else None
        try:
            runner = SimulationRunner(
                scene=scene,
                workflow=workflow,
                env=env,
                app=app_ctx.app,
                bus=bus,
                keys=keys,
                recorder=recorder,
                publisher=publisher,
                episodes=args.episodes,
                attempts=args.attempts,
                max_steps=args.episode_steps or None,
                record_failures=args.record_failures,
                seed=args.seed,
                sensor_views=sensor_views,
                sensor_view_titles=scene.sensor_view_titles(),
                sensor_view_outputs=scene.sensor_view_outputs(),
                sensor_view_keyboard_toggles=scene.sensor_view_keyboard_toggles(),
                sensor_view_projection_presets=scene.sensor_view_projection_presets(),
                sensor_view_projection_defaults=scene.sensor_view_projection_defaults(),
                sensor_view_appearances=scene.sensor_view_appearances(),
                sensor_view_display_controls=scene.sensor_view_display_controls(),
                sensor_view_sliders=scene.sensor_view_sliders(),
                sensor_view_readouts=scene.sensor_view_readouts(env),
                initial_state=initial_state,
                capture=CaptureRange(enabled=args.nsys_range, warmup_steps=args.nsys_warmup),
            )
            summary = runner.run()
            print(summary.render())
            exit_code = 0 if summary.complete else 1
        finally:
            if publisher is not None:
                publisher.close()
            if recorder is not None:
                recorder.close()
            if bus is not None:
                bus.close()
        if exit_code:
            # ovphysx installs an atexit os._exit(0) safety handler. Bypass it
            # after our resources are flushed so an incomplete run remains a
            # non-zero result to run.sh and CI.
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
