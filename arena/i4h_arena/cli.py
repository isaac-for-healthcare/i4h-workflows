# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena entry point.

Normally reached through ``run.sh``, which lints the workflow and starts any needed
backends first. Invoking this directly is supported and skips both.

The mode is resolved while the workflow is built, before Isaac starts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("arena")


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
        type=int,
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
    # Arena only populates env_cfg.sim.physics when this is set, and the
    # embodiments then tune it — leave it out and physics is None.
    simulation.add_argument("--presets", choices=("physx", "newton"), default="physx", help="Arena physics backend")
    simulation.add_argument("--spacing", type=float, default=4.0, dest="env_spacing")
    simulation.add_argument("--no-cameras", action="store_true")
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
        choices=("cuda", "vulkan"),
        default="vulkan",
        help="Slang device used by the patient-backed fluoroscopy renderer",
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
    declared_cap = default_registry().scene(workflow.scene).for_mode(workflow.mode).max_steps
    if args.episode_steps > declared_cap:
        parser.error(
            f"--episode-steps cannot exceed {declared_cap} for {args.workflow}; "
            "the validated per-environment cap may only be lowered"
        )
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
        }
        if args.teleop_device:
            kwargs["device"] = args.teleop_device
    elif mode == "idle":
        kwargs = {"seconds": args.idle_seconds}
    return resolve_workflow(args.workflow, args.mode or None, **kwargs)


def _launch(args: argparse.Namespace, workflow) -> int:
    # This function-local import must remain below workflow resolution.
    from i4h_arena.app import launch_app

    with launch_app(args) as app_ctx:
        from i4h_arena.io.publishers import ScenePublisher
        from i4h_arena.recording.hdf5 import EpisodeRecorder
        from i4h_arena.runner import SimulationRunner
        from i4h_arena.scenes.base import load_scene
        from i4h_common.bus.keys import Keys
        from i4h_common.bus.zenoh_bus import open_zenoh_bus

        # A SystemExit here is not ours: IsaacLab-Arena and the asset SDK call
        # sys.exit() on failures they consider fatal, which looks like a clean
        # shutdown from the outside — Isaac closes, the runner never ticks, and
        # the process reports 0. Name it so the next reader is not hunting.
        try:
            scene = load_scene(workflow.scene, args)
            scene.configure_args(args)
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
