#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load a finalized native checkpoint and verify complete hosted state before stepping."""

import argparse
import os
from importlib import metadata
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["I4H_RL_TRAINER_CONFIG"] = str(root / "rl/config/liver_reach_retract_ppo_rsl_rl.yaml")
    from i4h_arena.scenes.base import load_scene
    from i4h_rl.liver_experiment import digest, write_json
    from i4h_rl.liver_host import build_parser, launch_app

    cli = build_parser().parse_args(
        ["--workflow", "liver_reach_retract", "--mode", "idle", "--envs", "16", "--seed", "42", "--headless"]
    )
    scene = load_scene("liver_reach_retract", cli, register_assets=False)
    scene.configure_args(cli)
    with launch_app(cli) as app:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        from i4h_arena.agents.rsl_rl import ProfiledRslRlRunnerCfg

        env = app.make_env(scene)
        cfg = handle_deprecated_rsl_rl_cfg(ProfiledRslRlRunnerCfg(), metadata.version("rsl-rl-lib"))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=None, device=env.device)
        source = torch.load(args.checkpoint, weights_only=False, map_location=env.device)
        runner.load(str(args.checkpoint), map_location=env.device)
        actual = env.capture_complete_state()
        for family in ("numerical", "probe", "task", "rng"):
            for key, expected in source["liver_environment"][family].items():
                np.testing.assert_array_equal(actual[family][key], expected, err_msg=f"{family}/{key}")
        for key, value in source["actor_state_dict"].items():
            assert torch.equal(value, runner.alg.actor.state_dict()[key])

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for left, right in zip(a, b, strict=True):
                    equal(left, right)
            else:
                assert a == b

        equal(source["optimizer_state_dict"], runner.alg.optimizer.state_dict())
        write_json(
            args.output,
            dict(
                checkpoint_sha256=digest(args.checkpoint),
                complete_environment_exact=True,
                actor_distribution_exact=True,
                optimizer_exact=True,
                completed_updates=runner.alg.completed_updates,
                actor_updates=runner.alg.actor_updates,
                source_requalifications=len(runner.alg.resume_qualifications),
            ),
        )
        env.close()


if __name__ == "__main__":
    main()
