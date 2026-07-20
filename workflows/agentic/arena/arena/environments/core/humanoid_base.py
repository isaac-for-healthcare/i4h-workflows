# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for G1 humanoid environments.

All three G1 envs share the same asset-registry warmup, the same Zenoh-based
policy runtime, and a common no-op rollout that seeds the EEF pose. Only the
scene composition + task object differ — concrete envs implement
:meth:`get_env` and add any env-specific CLI args on top of the shared
``--embodiment`` / ``--teleop-device`` flags.
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace
from typing import Any

import torch
from arena.dump import SceneDumper, parse_scene_pose_names, should_dump_scene_step
from arena.environments.core.base import AgenticEnvironmentBase, policy_io_factory
from common.config import get_robot_config
from common.utils import nonnegative_int, resolve_path
from tqdm import trange

_G1_CONFIG = get_robot_config("g1")
_G1_ASSEMBLE_TROCAR_EMBODIMENT_NAME = _G1_CONFIG.assemble_trocar_embodiment_name
_G1_LOCOMANIP_EMBODIMENT_NAME = _G1_CONFIG.locomanip_embodiment_name
_G1_LOCOMANIP_TELEOP_EMBODIMENT_NAME = _G1_CONFIG.locomanip_teleop_embodiment_name
_G1_TELEOP_DEVICES = _G1_CONFIG.teleop_devices


WBC_TELEOP_DEVICES = {"keyboard_23d"}


def _episode_indices(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("episode indices must be comma-separated non-negative integers")
    return tuple(nonnegative_int(part) for part in parts)


class HumanoidEnvironmentBase(AgenticEnvironmentBase):
    """Common G1 humanoid environment scaffolding."""

    def __init__(self) -> None:
        from isaaclab_arena.assets.asset_registry import AssetRegistry, DeviceRegistry

        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        default_embodiment = _G1_LOCOMANIP_EMBODIMENT_NAME or _G1_ASSEMBLE_TROCAR_EMBODIMENT_NAME
        parser.add_argument("--embodiment", type=str, default=default_embodiment)
        parser.add_argument("--teleop", action="store_true")
        parser.add_argument(
            "--teleop-device",
            dest="teleop_device",
            type=str,
            default=None,
            choices=_G1_TELEOP_DEVICES,
        )
        parser.add_argument("--teleop-sensitivity", type=float, default=1.0)
        parser.add_argument("--replay", dest="replay_dataset_path", default=None, metavar="DATASET_PATH")
        parser.add_argument("--episode-index", dest="replay_episode_index", type=_episode_indices, default=(0,))
        parser.add_argument("--record", action="store_true")
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
        parser.add_argument(
            "--success-hold-steps",
            dest="success_hold_steps",
            type=int,
            default=1,
            help="Require the success termination to hold for N consecutive env steps "
            "before the episode terminates. Filters transient single-step proximity. "
            "Default 1 = upstream behaviour.",
        )
        parser.add_argument("--record-to", default=None)
        parser.add_argument("--enable-webrtc", action="store_true")
        parser.add_argument("--disable_pinocchio", dest="enable_pinocchio", default=True, action="store_false")

    def configure_args(self, args: argparse.Namespace) -> None:
        super().configure_args(args)
        uses_wbc_teleop_action = getattr(args, "teleop", False) or self._is_zero_action_run(args)
        if uses_wbc_teleop_action and args.teleop_device is None:
            if args.embodiment == _G1_ASSEMBLE_TROCAR_EMBODIMENT_NAME:
                raise ValueError(
                    "assemble_trocar uses a 43D direct-joint action space; "
                    "the default keyboard_23d teleop emits 23D WBC actions and is not compatible. "
                    "Rheo handles trocar teleop through a separate XR WBC teleop env; "
                    "run policy/replay here until that teleop env is ported."
                )

            args.teleop_device = _G1_TELEOP_DEVICES[0]
        if uses_wbc_teleop_action and args.teleop_device in WBC_TELEOP_DEVICES:
            if args.embodiment == _G1_LOCOMANIP_EMBODIMENT_NAME:
                args.embodiment = _G1_LOCOMANIP_TELEOP_EMBODIMENT_NAME
        if args.enable_pinocchio and args.embodiment == "g1_wbc_pink":
            import pinocchio  # noqa: F401

    def register_assets(self) -> None:
        from isaaclab_arena.assets import asset_registry

        asset_registry._assets_registered = True
        import arena.assets.assemble_trocar  # noqa: F401
        import arena.assets.locomanip  # noqa: F401
        import arena.embodiments.g1  # noqa: F401
        import arena.teleop.devices.gamepad  # noqa: F401

    def run(self, args, env, app, controller) -> None:
        if getattr(args, "replay_dataset_path", None):
            self._run_replay(args, env, app, controller)
        elif getattr(args, "teleop", False):
            self._run_teleop(args, env, app, controller)
        elif args.episodes > 0:
            self._run_policy_episodes(args, env, app, controller)
        else:
            self._run_zero(args, env, app)

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
        from arena.teleop.helpers.humanoid import make_teleop_interface
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

    def _run_zero(self, args, env, app) -> None:
        logger = logging.getLogger("arena")
        obs, _ = env.reset()
        action_dim = env.action_space.shape[-1]
        action = torch.zeros(env.num_envs, action_dim, device=env.device)
        if action_dim == 23:
            action = self._seed_initial_eef_pose(obs, action)
            action[:, 19] = float(getattr(args, "teleop_base_height", 0.65))
        args.enable_cameras = True
        io_context = None
        publish_observation = None
        runtime = self.import_runtime_module()
        io_factory = policy_io_factory(runtime)
        io_context = io_factory(env_id=self.name)
        publish_observation = getattr(io_context, "publish_observation", None)
        if publish_observation is None:
            raise RuntimeError(f"{self.name} runtime PolicyIO does not expose publish_observation()")
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
                teleop_device=getattr(args, "teleop_device", None),
                env_id=self.name,
            )
            debug_names = parse_scene_pose_names(getattr(args, "dump_scene_entities", None))
            if should_dump_scene_step(args, 0) and frame_dumper is not None:
                frame_dumper.dump_frames(step=0, observation=obs, env=env)
                frame_dumper.dump_pose(ctx, "after env.reset", step=0, names=debug_names)
                frame_dumper.dump_pose(ctx, "before zero-action step 0", step=0, names=debug_names, actions=action)
            if publish_observation is not None:
                publish_observation(obs)
            for step in trange(args.num_steps, desc=self.name):
                with torch.inference_mode():
                    obs, _, _, _, _ = env.step(action)
                after_step = step + 1
                if should_dump_scene_step(args, after_step) and frame_dumper is not None:
                    frame_dumper.dump_frames(step=after_step, observation=obs, env=env)
                    frame_dumper.dump_pose(
                        ctx, f"after zero-action step {after_step}", step=after_step, names=debug_names
                    )
                if publish_observation is not None:
                    publish_observation(obs)
                if not app.is_running():
                    break
        finally:
            if frame_dumper is not None:
                frame_dumper.close()
            if io_context is not None:
                io_context.__exit__(None, None, None)

    @staticmethod
    def _is_zero_action_run(args: argparse.Namespace) -> bool:
        return (
            not getattr(args, "teleop", False)
            and not getattr(args, "replay_dataset_path", None)
            and getattr(args, "episodes", 0) <= 0
        )

    @staticmethod
    def _seed_initial_eef_pose(obs, action):
        policy_obs = obs.get("policy", {})
        for key, start, end in (
            ("left_eef_pos", 2, 5),
            ("left_eef_quat", 5, 9),
            ("right_eef_pos", 9, 12),
            ("right_eef_quat", 12, 16),
        ):
            value = policy_obs.get(key)
            if value is None:
                continue
            tensor = (
                value.detach().to(device=action.device, dtype=action.dtype)
                if hasattr(value, "detach")
                else torch.as_tensor(value, device=action.device, dtype=action.dtype)
            )
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1 and action.shape[0] > 1:
                tensor = tensor.repeat(action.shape[0], 1)
            action[:, start:end] = tensor[: action.shape[0], : end - start]
        return action

    def _run_policy_episodes(self, args, env, app, controller) -> None:
        from arena.recording import (
            close_recording,
            discard_episode,
            save_episode,
            save_successful_episode,
            setup_recording,
        )

        runtime = self.import_runtime_module()

        hold = int(getattr(args, "success_hold_steps", 1))
        if hold > 1:
            from arena.runtimes.core.success_hold import apply_success_hold

            if apply_success_hold(env, hold, env.unwrapped.num_envs):
                logging.getLogger("arena").info(
                    "success-hold enabled: %s consecutive steps required for env %s", hold, self.name
                )

        record_to = resolve_path(getattr(args, "record_to", None), self.name)
        save_all = bool(getattr(args, "save_all_episodes", False))
        ctx = SimpleNamespace(env=env, controller=controller, simulation_app=app, device=args.device, env_id=self.name)
        ctx.save_episode_cb = (lambda meta=None: save_episode(ctx, metadata=meta)) if record_to else None
        ctx.save_successful_episode_cb = (
            (lambda meta=None: save_successful_episode(ctx, metadata=meta)) if record_to else None
        )
        ctx.discard_episode_cb = (lambda: discard_episode(ctx)) if record_to else None
        ctx.save_all_episodes = save_all
        if record_to:
            logging.getLogger("arena").info(
                "policy job: env=%s episodes=%s record=%s save_all=%s", self.name, args.episodes, record_to, save_all
            )
            setup_recording(ctx, record_to, streaming=True, filter_success=not save_all)
        try:
            runtime.run(ctx, args)
        finally:
            if record_to:
                close_recording(ctx)

    def build_idle_action(self, args: argparse.Namespace, env: Any, obs: Any) -> Any:
        """Hold-pose action for edit-mode keep-open.

        Mirrors the seed used in :meth:`_run_zero`: zeros + observed EEF pose +
        ``base_height_cmd`` so the WBC keeps the floating-base G1 upright while
        the user inspects / edits the scene.
        """
        action_dim = env.action_space.shape[-1]
        action = torch.zeros(env.num_envs, action_dim, device=env.device)
        if action_dim == 23:
            action = self._seed_initial_eef_pose(obs, action)
            action[:, 19] = float(getattr(args, "teleop_base_height", 0.65))
        return action

    def _resolve_teleop_device(self, name: str | None) -> Any:
        if name is None or name == "keyboard_23d":
            return None
        return self.device_registry.get_device_by_name(name)()

    @staticmethod
    def apply_wbc_default_base_height(embodiment: Any, base_height_m: float) -> None:
        """Seed the WBC's initial ``base_height_command`` to ``base_height_m``.

        New G1 envs whose ground/pelvis geometry differs from the locomanip
        default (ground at z=-0.75, pelvis at z=0) should call this from
        ``get_env`` after constructing the embodiment, so the WBC doesn't
        fight the env's geometry whenever the policy's ``base_height_command``
        head is under-trained.

        Example: a scissor-style hybrid env with ground at z=-0.65 should
        pass ``base_height_m=0.65``.
        """
        for action_cfg in vars(getattr(embodiment, "action_config", object())).values():
            if hasattr(action_cfg, "default_base_height"):
                action_cfg.default_base_height = float(base_height_m)

    def _maybe_patch_locomanip_mimic(self, args: argparse.Namespace, embodiment: Any) -> None:
        if args.embodiment == "g1_wbc_pink" and getattr(args, "mimic", False) and not hasattr(args, "auto"):
            from isaaclab_arena.utils.locomanip_mimic_patch import patch_g1_locomanip_mimic

            patch_g1_locomanip_mimic()
            embodiment.get_action_cfg().g1_action.use_p_control = False
