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

"""Workflow policy runner CLI helpers."""

import argparse

from isaaclab_arena.examples.policy_runner_cli import (
    add_gr00t_closedloop_arguments,
    add_replay_arguments,
    add_replay_lerobot_arguments,
    add_zero_action_arguments,
    get_isaaclab_arena_environments_cli_parser,
)
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy


def setup_policy_argument_parser(args_parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure parser with Workflow-specific policy arguments."""
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)

    args_parser.add_argument(
        "--policy_type",
        type=str,
        choices=["zero_action", "replay", "replay_lerobot", "gr00t_closedloop"],
        required=True,
        help="Type of policy to use",
    )
    args_parser.add_argument(
        "--success_hold_steps",
        type=int,
        default=1,
        help="Consecutive steps success must be held before task is considered successful.",
    )

    add_zero_action_arguments(args_parser)
    add_replay_arguments(args_parser)
    add_replay_lerobot_arguments(args_parser)
    add_gr00t_closedloop_arguments(args_parser)

    return args_parser


def validate_policy_args(args: argparse.Namespace) -> None:
    """Validate policy-related arguments after parsing."""
    if args.policy_type == "replay" and args.replay_file_path is None:
        raise ValueError("--replay_file_path is required when using --policy_type replay")
    if args.policy_type == "replay_lerobot" and args.config_yaml_path is None:
        raise ValueError("--config_yaml_path is required when using --policy_type replay_lerobot")
    if args.policy_type == "gr00t_closedloop" and args.policy_config_yaml_path is None:
        raise ValueError("--policy_config_yaml_path is required when using --policy_type gr00t_closedloop")


def create_policy(args: argparse.Namespace) -> tuple[PolicyBase, int]:
    """Create the appropriate policy based on the arguments and return (policy, num_steps)."""
    if args.policy_type == "replay":
        policy = ReplayActionPolicy(args.replay_file_path, args.episode_name)
        num_steps = len(policy)
    elif args.policy_type == "zero_action":
        policy = ZeroActionPolicy()
        num_steps = args.num_steps
    elif args.policy_type == "replay_lerobot":
        from isaaclab_arena_gr00t.replay_lerobot_action_policy import ReplayLerobotActionPolicy

        assert args.num_envs == 1, "Only single environment evaluation is supported for replay Lerobot action policy"
        policy = ReplayLerobotActionPolicy(
            args.config_yaml_path, num_envs=args.num_envs, device=args.device, trajectory_index=args.trajectory_index
        )
        # Use custom max_steps if provided to optionally playing partial sequence in one trajectory
        if args.max_steps is not None:
            num_steps = args.max_steps
        else:
            num_steps = policy.get_trajectory_length(policy.get_trajectory_index())

    elif args.policy_type == "gr00t_closedloop":
        from simulation.gr00t_closedloop_policy import CustomGr00tClosedloopPolicy

        policy = CustomGr00tClosedloopPolicy(
            args.policy_config_yaml_path, num_envs=args.num_envs, device=args.policy_device
        )
        num_steps = args.num_steps
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")
    return policy, num_steps
