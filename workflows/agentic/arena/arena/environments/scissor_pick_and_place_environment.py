# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import uuid
from types import SimpleNamespace
from typing import Any

import torch
from arena.arena_config import get_arena_config
from arena.dump import SceneDumper, parse_scene_pose_names, should_dump_scene_step
from arena.environments.core.base import AgenticEnvironmentBase, policy_io_factory
from arena.statemachine.core.dispatch import (
    add_state_machine_cli_args,
    configure_state_machine_args,
    run_state_machine_module,
    state_machine_requested,
)
from common.config import get_robot_config
from common.utils import nonnegative_int, resolve_path
from tqdm import trange

_SO101_CONFIG = get_robot_config("so101")


def _episode_indices(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("episode indices must be comma-separated non-negative integers")
    return tuple(nonnegative_int(part) for part in parts)


class ScissorPickAndPlaceEnvironment(AgenticEnvironmentBase):
    name: str = "scissor_pick_and_place"
    state_machine_module: str = "arena.statemachine.scissor_pick_and_place"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_state_machine_cli_args(parser)
        parser.add_argument("--action-device", choices=("keyboard", "joint_position"), default="joint_position")
        parser.add_argument("--episode-length-s", type=float, default=8.0)
        parser.add_argument("--record", action="store_true")
        parser.add_argument("--record-to", default=None)
        parser.add_argument(
            "--max-attempts",
            type=nonnegative_int,
            default=0,
            help=(
                "Cap attempts per requested episode when recording. 0 (default): "
                "1 without filtering, 3 when saving only successful episodes."
            ),
        )
        parser.add_argument(
            "--save-all-episodes",
            action="store_true",
            help=(
                "Save every attempted episode (success or failure). Default is to "
                "save only successful ones and retry on failure up to --max-attempts."
            ),
        )
        parser.add_argument("--teleop", action="store_true")
        parser.add_argument(
            "--teleop-device",
            choices=_SO101_CONFIG.teleop_devices,
            default=_SO101_CONFIG.teleop_devices[0],
        )
        parser.add_argument("--teleop-sensitivity", type=float, default=1.0)
        parser.add_argument("--teleop-port", type=str, default="/dev/ttyACM1")
        parser.add_argument("--teleop-recalibrate", action="store_true")
        parser.add_argument("--replay", dest="replay_dataset_path", default=None, metavar="DATASET_PATH")
        parser.add_argument("--episode-index", dest="replay_episode_index", type=_episode_indices, default=(0,))

    def configure_args(self, args: argparse.Namespace) -> None:
        if args.teleop:
            from arena.teleop.helpers.soarm import action_device_for_teleop

            args.action_device = action_device_for_teleop(args.teleop_device)
        configure_state_machine_args(args, action_device="joint_position", default_episodes=1)
        super().configure_args(args)

    def get_env(self, args: argparse.Namespace) -> Any:
        print("[arena] importing Arena scene/environment classes", flush=True)
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        print("[arena] importing scissor-pick-and-place Arena components", flush=True)
        from arena.assets.scissor_pick_and_place import make_scissor_pick_and_place_scene_assets
        from arena.embodiments.so_arm import SoArm101Embodiment
        from arena.tasks.scissor_pick_and_place import ScissorPickAndPlaceTask

        arena_config = get_arena_config(self.name)
        if arena_config.home_joint_pos_rad is None:
            raise ValueError(f"arena config for {self.name!r} is missing home_joint_pos_rad")
        embodiment = SoArm101Embodiment(
            enable_cameras=args.enable_cameras,
            action_device=getattr(args, "action_device", "joint_position"),
            home_joint_pos_rad=arena_config.home_joint_pos_rad,
        )
        print(f"[arena] embodiment: {type(embodiment).__name__} (name={embodiment.name})", flush=True)
        print("[arena] constructing scene assets", flush=True)
        scene = Scene(assets=make_scissor_pick_and_place_scene_assets())
        print("[arena] constructing task", flush=True)
        episode_length_s = getattr(args, "episode_length_s", 8.0)
        if (
            not getattr(args, "episodes", 0)
            and not getattr(args, "teleop", False)
            and not getattr(args, "replay_dataset_path", None)
        ):
            episode_length_s = max(episode_length_s, (getattr(args, "num_steps", 1000) + 1) / 60.0)
        task = ScissorPickAndPlaceTask(
            episode_length_s=episode_length_s,
            env_spacing=getattr(args, "env_spacing", 4.0),
            home_joint_pos_rad=arena_config.home_joint_pos_rad,
        )

        return IsaacLabArenaEnvironment(name=self.name, embodiment=embodiment, scene=scene, task=task)

    def run(self, args, env, app, controller) -> None:
        if state_machine_requested(args):
            run_state_machine_module(self.state_machine_module, args=args, env=env, app=app, controller=controller)
        elif getattr(args, "replay_dataset_path", None):
            self._run_replay(args, env, app, controller)
        elif args.teleop:
            self._run_teleop(args, env, app, controller)
        elif args.episodes > 0:
            self._run_policy_episodes(args, env, app, controller)
        else:
            self._run_zero(args, env, app)

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
        with arena_policy_io(env_id=self.name) as io:
            ctx = SimpleNamespace(
                env=env, io=io, controller=controller, simulation_app=app, device=args.device, env_id=self.name
            )
            if record_to:
                setup_recording(ctx, record_to, streaming=save_all, filter_success=not save_all)
            external_success = bool(record_to and not save_all)
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
                    status = run_policy_based_episode(
                        ctx,
                        max_timesteps=max_timesteps,
                        external_success=external_success,
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

    def _run_teleop(self, args, env, app, controller) -> None:
        from arena.teleop.helpers.soarm import (
            make_teleop_interface,
            soarm_action_postprocess,
            soarm_leader_absolute_targets,
            soarm_leader_action_postprocess,
        )
        from arena.teleop.teleop import run_teleop_job

        sync_robot_joints = getattr(self.import_runtime_module(), "sync_robot_joints")
        ctx = SimpleNamespace(
            env=env,
            io=None,
            controller=controller,
            simulation_app=app,
            device=args.device,
            teleop_device=args.teleop_device,
            env_id=self.name,
        )
        leader_reference = {"absolute_targets": None}
        home_joint_pos_rad = get_arena_config(self.name).home_joint_pos_rad
        if home_joint_pos_rad is None:
            raise ValueError(f"arena config for {self.name!r} is missing home_joint_pos_rad")

        def _leader_postprocess(actions, current_reference):
            processed, new_reference = soarm_leader_action_postprocess(actions, current_reference, home_joint_pos_rad)
            leader_reference["absolute_targets"] = soarm_leader_absolute_targets(
                actions, new_reference, home_joint_pos_rad
            )
            return processed, new_reference

        def _sync_leader_pose(sim_env, actions):
            if leader_reference["absolute_targets"] is None:
                sync_robot_joints(sim_env, actions)
                return
            sync_robot_joints(sim_env, leader_reference["absolute_targets"])

        action_postprocess = soarm_action_postprocess
        on_first_action = _sync_leader_pose if args.teleop_device == "so101_leader" else sync_robot_joints
        if args.teleop_device == "so101_leader":
            action_postprocess = _leader_postprocess
        run_teleop_job(
            ctx,
            args,
            make_teleop_interface=make_teleop_interface,
            action_postprocess=action_postprocess,
            sync_first_action=args.teleop_device == "so101_leader",
            on_first_action=on_first_action,
        )

    def _run_replay(self, args, env, app, controller) -> None:
        from arena.replay import run_recorded_episode

        dataset_path = str(resolve_path(args.replay_dataset_path, self.name))
        ctx = SimpleNamespace(
            env=env, io=None, controller=controller, simulation_app=app, device=args.device, teleop_device=None
        )
        for index in args.replay_episode_index:
            run_recorded_episode(ctx, dataset_path=dataset_path, episode_index=index)
