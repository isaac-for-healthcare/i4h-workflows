# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rollout helpers for native surgical Arena environments."""

from __future__ import annotations

import argparse
import logging
import uuid
from types import SimpleNamespace

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


def add_surgical_policy_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-device", choices=("joint_position", "keyboard"), default="joint_position")
    parser.add_argument("--episode-length-s", type=float, default=5.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-to", default=None)
    parser.add_argument(
        "--max-attempts",
        type=nonnegative_int,
        default=1,
        help="Cap attempts per requested policy episode.",
    )
    parser.add_argument(
        "--save-all-episodes",
        action="store_true",
        help="Save every attempted policy episode, including timeout/failure smoke runs.",
    )
    parser.add_argument("--replay", dest="replay_dataset_path", default=None, metavar="DATASET_PATH")
    parser.add_argument("--episode-index", dest="replay_episode_index", type=_episode_indices, default=(0,))


class SurgicalPolicyEnvironmentBase(AgenticEnvironmentBase):
    """Common policy, replay, and zero-action loops for surgical envs."""

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_surgical_policy_cli_args(parser)

    def run(self, args, env, app, controller) -> None:
        if getattr(args, "replay_dataset_path", None):
            self._run_replay(args, env, app, controller)
        elif args.episodes > 0:
            self._run_policy_episodes(args, env, app, controller)
        else:
            self._run_zero(args, env, app)

    def _run_policy_episodes(self, args, env, app, controller) -> None:
        from arena.recording import (
            close_recording,
            discard_episode,
            save_episode,
            save_successful_episode,
            setup_recording,
        )

        logger = logging.getLogger("arena")
        runtime = self.import_runtime_module()
        arena_policy_io = policy_io_factory(runtime)
        run_policy_based_episode = getattr(runtime, "run_policy_based_episode")
        max_timesteps = int(args.max_timesteps or 300)
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
        with arena_policy_io(env_id=self.name) as io:
            ctx = SimpleNamespace(
                env=env,
                io=io,
                controller=controller,
                simulation_app=app,
                device=args.device,
                env_id=self.name,
            )
            if record_to:
                setup_recording(ctx, record_to, streaming=save_all, filter_success=False)
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
                    env.reset()
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
                    status = run_policy_based_episode(ctx, max_timesteps=max_timesteps)
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
                    else:
                        failed += 1
                        current_episode += 1
                        episode_attempts = 0
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

    def _run_zero(self, args, env, app) -> None:
        logger = logging.getLogger("arena")
        runtime = self.import_runtime_module()
        policy_io_cls = policy_io_factory(runtime)
        publish_obs = getattr(runtime, "publish_obs")
        obs, _ = env.reset()
        action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
        publish_cameras = bool(getattr(args, "enable_cameras", False))
        io_context = policy_io_cls(env_id=self.name) if publish_cameras else None
        frame_dumper = None
        try:
            frame_dumper = SceneDumper.from_args(args, env_id=self.name)
            if io_context is not None:
                io_context.__enter__()
                logger.info("zero-action camera publishing enabled for env=%s", self.name)
            ctx = SimpleNamespace(
                env=env,
                io=io_context,
                controller=None,
                simulation_app=app,
                device=args.device,
                env_id=self.name,
            )
            debug_names = parse_scene_pose_names(getattr(args, "dump_scene_entities", None))
            if should_dump_scene_step(args, 0) and frame_dumper is not None:
                frame_dumper.dump_frames(step=0, observation=obs, env=env)
                frame_dumper.dump_pose(ctx, "after env.reset", step=0, names=debug_names)
                frame_dumper.dump_pose(ctx, "before zero-action step 0", step=0, names=debug_names, actions=action)
            for step in trange(args.num_steps, desc=self.name):
                with torch.inference_mode():
                    obs, _, _, _, _ = env.step(action)
                after_step = step + 1
                if should_dump_scene_step(args, after_step) and frame_dumper is not None:
                    frame_dumper.dump_frames(step=after_step, observation=obs, env=env)
                    frame_dumper.dump_pose(
                        ctx, f"after zero-action step {after_step}", step=after_step, names=debug_names
                    )
                if publish_cameras:
                    publish_obs(ctx)
                if not app.is_running():
                    break
        finally:
            if frame_dumper is not None:
                frame_dumper.close()
            if io_context is not None:
                io_context.__exit__(None, None, None)

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


class SurgicalStateMachineEnvironmentBase(SurgicalPolicyEnvironmentBase):
    """Surgical env that also exposes a scripted absolute-IK state-machine demo.

    Subclasses set ``name`` + ``state_machine_module`` and implement ``get_env``;
    the CLI, arg configuration, and the ``--state-machine`` vs policy dispatch are
    shared here.
    """

    state_machine_module: str = ""

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_surgical_policy_cli_args(parser)
        add_state_machine_cli_args(
            parser,
            help_text="Run the original robotic_surgery state-machine demo with absolute IK actions.",
        )

    def configure_args(self, args: argparse.Namespace) -> None:
        configure_state_machine_args(args, action_device="ik_abs")
        super().configure_args(args)

    def run(self, args, env, app, controller) -> None:
        if state_machine_requested(args):
            run_state_machine_module(self.state_machine_module, args=args, env=env, app=app, controller=controller)
        else:
            super().run(args, env, app, controller)
