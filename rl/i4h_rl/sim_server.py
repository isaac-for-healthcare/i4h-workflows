# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve a profile-selected Workflow Scene to an isolated RL worker."""

from __future__ import annotations

import argparse
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

from i4h_rl.sim_bridge import bridge_authkey, to_numpy_tree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--env-spacing", type=float, required=True)
    parser.add_argument("--presets", required=True)
    parser.add_argument("--enable-cameras", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _scene_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mode=None,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        solve_relations=True,
        mimic=False,
        presets=args.presets,
        device="cuda:0",
        disable_fabric=False,
        episode_steps=args.max_episode_steps,
        no_cameras=not args.enable_cameras,
        enable_cameras=args.enable_cameras,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from isaaclab.app import AppLauncher

    sim_app = AppLauncher(headless=True, enable_cameras=args.enable_cameras).app
    env = None
    listener = None
    try:
        import gymnasium as gym
        import torch

        from i4h_arena.scenes.base import load_scene

        scene_args = _scene_args(args)
        scene = load_scene(args.scene, scene_args)
        scene.configure_args(scene_args)
        gym_id, env_cfg = scene.gym_spec()
        env_cfg.scene.num_envs = args.num_envs
        render_mode = "rgb_array" if args.enable_cameras else None
        env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode).unwrapped

        args.socket.parent.mkdir(parents=True, exist_ok=True)
        args.socket.unlink(missing_ok=True)
        listener = Listener(str(args.socket), family="AF_UNIX", authkey=bridge_authkey())
        args.ready_file.write_text("ready\n", encoding="utf-8")
        connection = listener.accept()
        try:
            while True:
                command, payload = connection.recv()
                try:
                    if command == "reset":
                        env_ids = payload["env_ids"]
                        if env_ids is not None:
                            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
                        response = env.reset(seed=payload["seed"], env_ids=env_ids)
                    elif command == "step":
                        actions = torch.as_tensor(payload, dtype=torch.float32, device=env.device)
                        response = env.step(actions)
                    elif command == "close":
                        connection.send(("ok", None))
                        break
                    else:
                        raise ValueError(f"unknown simulator command: {command}")
                    connection.send(("ok", to_numpy_tree(response)))
                except Exception:
                    connection.send(("error", traceback.format_exc()))
        finally:
            connection.close()
    finally:
        args.ready_file.unlink(missing_ok=True)
        if listener is not None:
            listener.close()
        args.socket.unlink(missing_ok=True)
        if env is not None:
            env.close()
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
