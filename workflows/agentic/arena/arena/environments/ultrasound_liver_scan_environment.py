# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import logging
import uuid
from types import SimpleNamespace
from typing import Any

import torch
from arena.dump import SceneDumper, parse_scene_pose_names, should_dump_scene_step
from arena.environments.core.base import AgenticEnvironmentBase, policy_io_factory
from arena.statemachine.core.dispatch import (
    add_state_machine_cli_args,
    configure_state_machine_args,
    run_state_machine_module,
    state_machine_requested,
)
from common.utils import nonnegative_int, resolve_path
from tqdm import trange


def _episode_indices(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("episode indices must be comma-separated non-negative integers")
    return tuple(nonnegative_int(part) for part in parts)


class UltrasoundLiverScanEnvironment(AgenticEnvironmentBase):
    name: str = "ultrasound_liver_scan"
    state_machine_module: str = "arena.statemachine.ultrasound_liver_scan"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_state_machine_cli_args(parser)
        parser.add_argument(
            "--episode-length-s",
            type=float,
            default=5.0,
            help="Per-episode time-out in seconds (default: 5.0).",
        )
        parser.add_argument("--record", action="store_true")
        parser.add_argument("--record-to", default=None)
        parser.add_argument(
            "--max-attempts",
            type=nonnegative_int,
            default=0,
            help=(
                "Maximum policy attempts per requested episode. Default is 3 when recording successes only, "
                "otherwise 1."
            ),
        )
        parser.add_argument(
            "--save-all-episodes",
            action="store_true",
            help="Save every attempted policy episode, including failures. Default saves only successful episodes.",
        )
        parser.add_argument("--teleop", action="store_true")
        parser.add_argument(
            "--teleop-device",
            dest="teleop_device",
            type=str,
            default="keyboard",
            choices=("keyboard", "gamepad"),
        )
        parser.add_argument("--teleop-sensitivity", type=float, default=1.0)
        parser.add_argument(
            "--success-target-tolerance-m",
            type=float,
            default=0.20,
            help="Probe-to-liver scan target distance, in meters, required for policy success.",
        )
        parser.add_argument(
            "--success-alignment-threshold",
            type=float,
            default=0.80,
            help="Minimum probe orientation alignment with the scan goal frame required for policy success.",
        )
        parser.add_argument(
            "--success-min-scan-steps",
            type=int,
            default=180,
            help="Do not stop on action saturation before this many policy steps.",
        )
        parser.add_argument(
            "--success-action-saturation-threshold",
            type=float,
            default=0.02,
            help="Log low-action saturation after min scan steps when actions stay below this max-abs threshold.",
        )
        parser.add_argument(
            "--success-action-saturation-steps",
            type=int,
            default=20,
            help="Consecutive low-action steps required before logging policy saturation.",
        )
        parser.add_argument("--replay", dest="replay_dataset_path", default=None, metavar="DATASET_PATH")
        parser.add_argument("--episode-index", dest="replay_episode_index", type=_episode_indices, default=(0,))

    def configure_args(self, args: argparse.Namespace) -> None:
        configure_state_machine_args(args, default_episodes=1)
        super().configure_args(args)

    def register_assets(self) -> None:
        # IsaacLab-Arena lazily scans installed packages for @register_asset
        # entries. The package scan tries to import optional Lightwheel SDK
        # assets we don't ship — short-circuit it before any IsaacLab-Arena
        # embodiment / asset module loads (notably the stock franka.franka).
        from isaaclab_arena.assets import asset_registry

        asset_registry._assets_registered = True

    def get_env(self, args: argparse.Namespace) -> Any:
        from arena.assets.ultrasound_liver_scan import make_ultrasound_scene_assets
        from arena.embodiments.franka import FrankaUltrasoundEmbodiment
        from arena.tasks.ultrasound_liver_scan import UltrasoundLiverScanTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        embodiment = FrankaUltrasoundEmbodiment(enable_cameras=args.enable_cameras)
        scene = Scene(assets=make_ultrasound_scene_assets())
        task = UltrasoundLiverScanTask(episode_length_s=getattr(args, "episode_length_s", 5.0))
        return IsaacLabArenaEnvironment(name=self.name, embodiment=embodiment, scene=scene, task=task)

    def run(self, args, env, app, controller) -> None:
        if state_machine_requested(args):
            run_state_machine_module(self.state_machine_module, args=args, env=env, app=app, controller=controller)
        elif getattr(args, "replay_dataset_path", None):
            self._run_replay(args, env, app, controller)
        elif getattr(args, "teleop", False):
            self._run_teleop(args, env, app, controller)
        elif args.episodes and args.episodes > 0:
            self._run_policy(args, env, app, controller)
        else:
            self._run_zero_action(args, env, app)

    def _run_replay(self, args, env, app, controller) -> None:
        from arena.replay import run_recorded_episode

        dataset_path = str(resolve_path(args.replay_dataset_path, self.name))
        ctx = SimpleNamespace(
            env=env,
            io=None,
            controller=controller,
            simulation_app=app,
            device=args.device,
            teleop_device=None,
            env_id=self.name,
        )
        for index in args.replay_episode_index:
            run_recorded_episode(ctx, dataset_path=dataset_path, episode_index=index)

    def _run_teleop(self, args, env, app, controller) -> None:
        from arena.teleop.helpers.franka import make_teleop_interface
        from arena.teleop.teleop import run_teleop_job

        ctx = SimpleNamespace(
            env=env,
            io=None,
            controller=controller,
            simulation_app=app,
            device=args.device,
            teleop_device=args.teleop_device,
            env_id=self.name,
        )
        run_teleop_job(ctx, args, make_teleop_interface=make_teleop_interface)

    def _run_zero_action(self, args, env, app) -> None:
        obs, _ = env.reset()
        action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
        steps = args.num_steps or 1000
        frame_dumper = SceneDumper.from_args(args, env_id=self.name)
        ctx = SimpleNamespace(
            env=env,
            io=None,
            controller=None,
            simulation_app=app,
            device=args.device,
            teleop_device=getattr(args, "teleop_device", None),
            env_id=self.name,
        )
        try:
            debug_names = parse_scene_pose_names(getattr(args, "dump_scene_entities", None))
            if should_dump_scene_step(args, 0) and frame_dumper is not None:
                frame_dumper.dump_frames(step=0, observation=obs, env=env)
                frame_dumper.dump_pose(ctx, "after env.reset", step=0, names=debug_names)
                frame_dumper.dump_pose(ctx, "before zero-action step 0", step=0, names=debug_names, actions=action)
            for step in trange(steps, desc=self.name):
                with torch.inference_mode():
                    obs, _, _, _, _ = env.step(action)
                after_step = step + 1
                if should_dump_scene_step(args, after_step) and frame_dumper is not None:
                    frame_dumper.dump_frames(step=after_step, observation=obs, env=env)
                    frame_dumper.dump_pose(
                        ctx, f"after zero-action step {after_step}", step=after_step, names=debug_names
                    )
                if not app.is_running():
                    break
        finally:
            if frame_dumper is not None:
                frame_dumper.close()

    def _run_policy(self, args, env, app, controller) -> None:
        from arena.arena_config import get_arena_config
        from arena.recording import (
            close_recording,
            discard_episode,
            save_episode,
            save_successful_episode,
            setup_recording,
        )

        logger = logging.getLogger("arena")
        runtime = self.import_runtime_module()
        ultrasound_arena_io = policy_io_factory(runtime)
        reset_to_home = getattr(runtime, "reset_to_home")
        run_policy_based_episode = getattr(runtime, "run_policy_based_episode")
        max_timesteps = args.max_timesteps or get_arena_config(self.name).max_timesteps or 1000
        record_to = resolve_path(args.record_to, self.name)
        save_all = bool(getattr(args, "save_all_episodes", False))
        if args.max_attempts and args.max_attempts > 0:
            max_attempts_per_episode = args.max_attempts
        elif save_all or not record_to:
            max_attempts_per_episode = 1
        else:
            max_attempts_per_episode = 3
        logger.info(
            "policy job: env=%s episodes=%s record=%s max_timesteps=%s save_all=%s max_attempts_per_episode=%s",
            self.name,
            args.episodes,
            record_to,
            max_timesteps,
            save_all,
            max_attempts_per_episode,
        )

        progress_label = "saved" if record_to else "completed"
        policy_job_id = uuid.uuid4().hex[:8]
        with ultrasound_arena_io(env_id=self.name) as io:
            ctx = SimpleNamespace(
                env=env, io=io, controller=controller, simulation_app=app, device=args.device, env_id=self.name
            )
            if record_to:
                setup_recording(
                    ctx,
                    record_to,
                    streaming=False,
                    filter_success=not save_all,
                )
            try:
                episode_attempts = 0
                total_attempts = 0
                saved = 0
                failed = 0
                current_episode = 1
                while current_episode <= args.episodes and app.is_running():
                    if episode_attempts >= max_attempts_per_episode:
                        logger.info(
                            "policy episode %s/%s exhausted %s attempts; marking failed and moving to next requested episode",
                            current_episode,
                            args.episodes,
                            max_attempts_per_episode,
                        )
                        failed += 1
                        current_episode += 1
                        episode_attempts = 0
                        continue
                    episode_attempts += 1
                    total_attempts += 1
                    reset_to_home(env)
                    run_id = f"{self.name}-{policy_job_id}-episode-{current_episode:03d}-attempt-{episode_attempts:02d}"
                    io.set_run_context(run_id=run_id, episode_index=current_episode, attempt_index=episode_attempts)
                    logger.info(
                        "policy episode %s/%s attempt %s/%s started; %s=%s/%s",
                        current_episode,
                        args.episodes,
                        episode_attempts,
                        max_attempts_per_episode,
                        progress_label,
                        saved,
                        args.episodes,
                    )
                    status = run_policy_based_episode(
                        ctx,
                        max_timesteps=max_timesteps,
                        success_target_tolerance_m=args.success_target_tolerance_m,
                        success_alignment_threshold=args.success_alignment_threshold,
                        success_min_scan_steps=args.success_min_scan_steps,
                        success_action_saturation_threshold=args.success_action_saturation_threshold,
                        success_action_saturation_steps=args.success_action_saturation_steps,
                    )
                    metadata = {
                        "env_id": self.name,
                        "run_id": run_id,
                        "episode_index": current_episode,
                        "attempt_index": episode_attempts,
                        "status": status,
                    }
                    is_success = status == "completed"
                    if record_to:
                        if is_success:
                            save_successful_episode(ctx, metadata=metadata)
                            saved += 1
                            current_episode += 1
                            episode_attempts = 0
                            controller.episode_completed()
                        elif save_all:
                            save_episode(ctx, metadata=metadata)
                            saved += 1
                            failed += 1
                            current_episode += 1
                            episode_attempts = 0
                        else:
                            discard_episode(ctx)
                    elif is_success:
                        saved += 1
                        current_episode += 1
                        episode_attempts = 0
                        controller.episode_completed()
                logger.info(
                    "policy job complete: %s=%s/%s failed=%s total_attempts=%s",
                    progress_label,
                    saved,
                    args.episodes,
                    failed,
                    total_attempts,
                )
            finally:
                if record_to:
                    close_recording(ctx)
