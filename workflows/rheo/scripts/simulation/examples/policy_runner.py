# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy runner that mirrors the upstream script with success-hold support."""

import random

import numpy as np
import torch
import tqdm
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli
from scripts.utils.policy_tasks import create_success_hold_wrapper
from simulation.examples.policy_runner_cli import create_policy, setup_policy_argument_parser, validate_policy_args
from simulation.register_and_patch import register_workflow_assets, register_workflow_cli

register_workflow_cli()


def main():
    """Script to run an IsaacLab Arena environment with a zero-action agent."""
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, _ = args_parser.parse_known_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Add policy-related arguments to the parser
        register_workflow_assets()
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        validate_policy_args(args_cli)
        # Build scene
        arena_builder = get_arena_builder_from_cli(args_cli)

        # Disable recorder to avoid HDF5 file lock conflicts
        env_name, env_cfg = arena_builder.build_registered()
        if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
            print("[INFO] Disabling recorders (not needed for policy runner)")
            env_cfg.recorders = None

        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()

        # Wrap success term with hold logic if specified (AFTER reset)
        if args_cli.success_hold_steps > 1:
            if hasattr(env.cfg, "terminations") and hasattr(env.cfg.terminations, "success"):
                print(
                    "[INFO] Success hold enabled: task must maintain success for"
                    f" {args_cli.success_hold_steps} consecutive steps"
                )
                original_success_term = env.cfg.terminations.success
                wrapped_success_term = create_success_hold_wrapper(
                    original_success_term,
                    args_cli.success_hold_steps,
                    args_cli.num_envs,
                )
                env.cfg.terminations.success = wrapped_success_term
                if hasattr(env, "termination_manager"):
                    if (
                        hasattr(env.termination_manager, "_term_name_to_term_idx")
                        and "success" in env.termination_manager._term_name_to_term_idx
                    ):
                        success_idx = env.termination_manager._term_name_to_term_idx["success"]
                        env.termination_manager._term_cfgs[success_idx] = wrapped_success_term
                        print(f"[INFO] Successfully wrapped success term in termination_manager at index {success_idx}")
                    else:
                        print("[WARNING] Could not find 'success' term in termination_manager")
            else:
                print("[WARNING] Environment does not have a success termination term, skipping success hold wrapping")

        # NOTE(xinjieyao, 2025-09-29): General rule of thumb is to have as many non-standard python
        # library imports after app launcher as possible, otherwise they will likely stall the sim
        # app. Given current SimulationAppContext setup, use lazy import to handle policy-related
        # deps inside create_policy() function to bringup sim app.
        policy, num_steps = create_policy(args_cli)
        policy.set_task_description(env.cfg.isaaclab_arena_env.task.get_task_description())

        # NOTE(xinjieyao, 2025-10-07): lazy import to prevent app stalling caused by omni.kit
        from isaaclab_arena.metrics.metrics import compute_metrics

        for _ in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)
                if terminated.any() or truncated.any():
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")
        env.close()


if __name__ == "__main__":
    main()
