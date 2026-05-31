# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML-backed policy routing helper for the top-level policy/run.sh."""

from __future__ import annotations

import argparse

from common.config import policy_routings, policy_stack_for_env


def _shorten(text: str, max_length: int = 48) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve agentic policy routing from env YAML.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-envs", action="store_true", help="Print env -> stack rows with language descriptions.")
    group.add_argument("--envs", action="store_true", help="Print env ids, one per line.")
    group.add_argument("--stack-for-env", metavar="ENV_ID", help="Print the stack for one env id.")
    args = parser.parse_args()

    if args.list_envs:
        for routing in policy_routings():
            print(f"{routing.env_id:<32s} {routing.stack:<12s} {_shorten(routing.language_description)}")
        return

    if args.envs:
        for routing in policy_routings():
            print(routing.env_id)
        return

    if args.stack_for_env:
        print(policy_stack_for_env(args.stack_for_env))


if __name__ == "__main__":
    main()
