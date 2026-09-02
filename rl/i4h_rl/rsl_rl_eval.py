# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a workflow-owned RSL-RL checkpoint in Isaac Lab."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import traceback
from importlib import metadata
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher
    from isaaclab_arena.cli.isaaclab_arena_cli import add_isaac_lab_cli_args, add_isaaclab_arena_cli_args

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-policy", help="optional TorchScript policy.pt output path")
    parser.add_argument("--adapter-module", help="optional workflow adapter exposing evaluation_metrics(env)")
    return parser


def _metric_values(module_name: str | None, env) -> dict[str, object]:
    if not module_name:
        return {}
    adapter = importlib.import_module(module_name)
    evaluation_metrics = getattr(adapter, "evaluation_metrics", None)
    if not callable(evaluation_metrics):
        return {}
    values = evaluation_metrics(env)
    if not isinstance(values, dict) or any(not isinstance(name, str) or not name for name in values):
        raise TypeError(f"{module_name}.evaluation_metrics() must return a string-keyed mapping")
    return values


def main() -> int:
    from isaaclab.app import AppLauncher

    args = _parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        # RSL-RL imports configclass as a submodule before Arena imports the
        # decorator. Restore the public package attribute for Arena config
        # declarations, matching the training registration callback.
        import isaaclab.utils as isaaclab_utils
        from isaaclab.utils.configclass import configclass

        isaaclab_utils.configclass = configclass

        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        from i4h_arena.scenes.base import load_scene

        args.episode_steps = 0
        args.no_cameras = True
        args.rl_observations = True
        args.rl_training_mode = False
        scene = load_scene(args.task, args)
        scene.configure_args(args)
        gym_id, env_cfg = scene.gym_spec()
        agent_cfg = load_cfg_from_registry(gym_id, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = args.num_envs

        raw_env = gym.make(gym_id, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if args.export_policy:
            export_policy = Path(args.export_policy).expanduser().resolve()
            export_policy.parent.mkdir(parents=True, exist_ok=True)
            runner.export_policy_to_jit(path=str(export_policy.parent), filename=export_policy.name)
            if not export_policy.is_file():
                raise RuntimeError(f"RSL-RL export did not create {export_policy}")
            print(f"exported TorchScript policy: {export_policy}")

        obs = env.get_observations()
        initial_metrics = _metric_values(args.adapter_module, env.unwrapped)
        metric_minima = {name: torch.full((env.num_envs,), float("inf"), device=env.device) for name in initial_metrics}
        episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        completed: list[dict[str, float | int | bool]] = []
        max_steps = env.max_episode_length * (args.episodes // env.num_envs + 2)

        with torch.inference_mode():
            for _ in range(max_steps):
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)
                episode_steps += 1
                current_metrics = _metric_values(args.adapter_module, env.unwrapped)
                if set(current_metrics) != set(metric_minima):
                    raise RuntimeError("evaluation metric names changed during an episode")
                for name, value in current_metrics.items():
                    if not isinstance(value, torch.Tensor) or value.shape != (env.num_envs,):
                        raise TypeError(f"evaluation metric {name!r} must be a tensor shaped ({env.num_envs},)")
                    metric_minima[name] = torch.minimum(metric_minima[name], value)
                time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).bool()
                done_ids = torch.nonzero(dones.bool(), as_tuple=False).flatten().tolist()
                for env_id in done_ids:
                    episode = {
                        "success": not bool(time_outs[env_id].item()),
                        "steps": int(episode_steps[env_id].item()),
                    }
                    episode.update(
                        {f"min_{name}": float(values[env_id].item()) for name, values in metric_minima.items()}
                    )
                    completed.append(episode)
                    if len(completed) >= args.episodes:
                        break
                if done_ids:
                    reset_ids = torch.as_tensor(done_ids, device=env.device)
                    for values in metric_minima.values():
                        values[reset_ids] = float("inf")
                    episode_steps[reset_ids] = 0
                    policy.reset(dones)
                if len(completed) >= args.episodes:
                    break

        completed = completed[: args.episodes]
        if len(completed) != args.episodes:
            raise RuntimeError(f"evaluation collected {len(completed)}/{args.episodes} episodes")
        successes = sum(bool(item["success"]) for item in completed)
        result: dict[str, object] = {
            "schema_version": 1,
            "task": args.task,
            "checkpoint": str(checkpoint),
            "episodes": args.episodes,
            "successes": successes,
            "success_rate": successes / args.episodes,
            "results": completed,
        }
        for name in metric_minima:
            result[f"mean_min_{name}"] = sum(float(item[f"min_{name}"]) for item in completed) / args.episodes
        if args.export_policy:
            result["exported_policy"] = str(Path(args.export_policy).expanduser().resolve())
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
        env.close()
        return 0
    except BaseException:
        # The active physics backend installs a native shutdown handler that
        # may replace a Python failure with exit status 0. Emit the original
        # traceback and exit before that handler can hide it.
        traceback.print_exc()
        with contextlib.suppress(Exception):
            simulation_app.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        with contextlib.suppress(Exception):
            simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
