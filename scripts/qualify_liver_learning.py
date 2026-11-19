#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise zero-LR and critic-only updates with the exact imported BC actor."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from i4h_rl.liver_experiment import digest, write_json
from i4h_rl.liver_ppo import GuardedPPO


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7341)
    frozen = torch.as_tensor(np.load(args.observations)["observations"], dtype=torch.float32)
    obs = TensorDict({"policy": frozen[:8]}, batch_size=[8])
    source = torch.load(args.checkpoint, weights_only=False, map_location="cpu")["actor"]
    reports = {}
    for name, lr, warmup in [("zero_lr", 0.0, 0), ("critic_invariance", 1e-5, 10)]:
        cfg = dict(
            actor=dict(
                class_name="rsl_rl.models:MLPModel",
                hidden_dims=[128, 128],
                activation="elu",
                obs_normalization=False,
                distribution_cfg=dict(class_name="rsl_rl.modules:GaussianDistribution", init_std=0.02),
            ),
            critic=dict(
                class_name="rsl_rl.models:MLPModel", hidden_dims=[128, 128], activation="elu", obs_normalization=False
            ),
            algorithm=dict(
                class_name="i4h_rl.liver_ppo:GuardedPPO",
                qualification_required=False,
                critic_warmup_updates=warmup,
                learning_rate=lr,
                schedule="fixed",
                num_learning_epochs=4,
                num_mini_batches=4,
            ),
            obs_groups=dict(actor=["policy"], critic=["policy"]),
            num_steps_per_env=8,
            multi_gpu=None,
        )
        alg = GuardedPPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=4), cfg, "cpu")
        alg.actor.load_state_dict(source, strict=True)
        alg.qualification = dict(validation_starts_sha256="synthetic", objective="full_cycle", bc_successes=20)
        alg.evaluator = lambda actor: dict(episodes=20, starts_sha256="synthetic", objective="full_cycle", successes=20)
        before = copy.deepcopy(alg.actor.state_dict())
        before_critic = copy.deepcopy(alg.critic.state_dict())
        with torch.no_grad():
            for tick in range(8):
                alg.act(obs)
                alg.process_env_step(
                    obs, torch.linspace(-1.0, 1.0, 8) + tick * 0.01, torch.zeros(8, dtype=torch.bool), {}
                )
            alg.compute_returns(obs)
        metrics = alg.update()
        assert all(torch.equal(value, alg.actor.state_dict()[key]) for key, value in before.items())
        assert metrics["update_kl"] < 1e-10
        critic_changed = any(
            not torch.equal(value, alg.critic.state_dict()[key]) for key, value in before_critic.items()
        )
        assert critic_changed == (name == "critic_invariance")
        reports[name] = dict(
            passed=True,
            analytic_kl=metrics["update_kl"],
            actor_distribution_and_normalization_unchanged=True,
            critic_changed=critic_changed,
            adaptive_schedule_disabled=True,
            learning_rate=lr,
        )
    write_json(
        args.output,
        dict(
            checkpoint_sha256=digest(args.checkpoint),
            frozen_input_sha256=digest(args.observations),
            rollout="synthetic rewards over frozen real BC input vectors",
            checks=reports,
        ),
    )


if __name__ == "__main__":
    main()
