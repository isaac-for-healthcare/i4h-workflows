#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a native liver actor and its normal Workflows TorchScript export."""

import argparse
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from i4h_rl.liver_experiment import digest, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    obs = torch.as_tensor(np.load(args.observations)["observations"], dtype=torch.float32)
    td = TensorDict({"policy": obs}, batch_size=[len(obs)])
    actor = MLPModel(
        td,
        {"actor": ["policy"]},
        "actor",
        4,
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=dict(class_name="rsl_rl.modules:GaussianDistribution", init_std=0.02),
    )
    actor.load_state_dict(state.get("actor", state.get("actor_state_dict")), strict=True)
    exported = torch.jit.load(str(args.policy)).eval()
    with torch.no_grad():
        means = actor(td)
        outputs = exported(obs)
        torch.testing.assert_close(outputs, means, rtol=0, atol=0)
        actor(td, stochastic_output=True)
        mean, std = actor.output_distribution_params
        kl = actor.get_kl_divergence((mean, std), (outputs, std))
        assert float(kl.max()) == 0.0
        inputs = np.load(args.observations)
        actions = (
            torch.as_tensor(inputs["applied_actions"], dtype=torch.float32) if "applied_actions" in inputs else means
        )
        log_prob = actor.get_output_log_prob(actions)
        exported_log_prob = torch.distributions.Normal(outputs, std).log_prob(actions).sum(-1)
        torch.testing.assert_close(log_prob, exported_log_prob, rtol=0, atol=0)
        torch.testing.assert_close(std, torch.full_like(std, 0.02), rtol=1e-6, atol=0)
    write_json(
        args.output,
        dict(
            checkpoint_sha256=digest(args.checkpoint),
            policy_sha256=digest(args.policy),
            samples=len(obs),
            means_exact=True,
            log_prob_max_abs_error=float((log_prob - exported_log_prob).abs().max()),
            max_analytic_kl=float(kl.max()),
            std_channels=std[0].tolist(),
        ),
    )


if __name__ == "__main__":
    main()
