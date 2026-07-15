# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML-backed policy routing helper for the top-level policy/run.sh."""

from __future__ import annotations

import argparse
import json
import os
from urllib.request import urlopen

from common.config import get_policy_config, policy_routings, policy_stack_for_env


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
    group.add_argument("--health-port-for-env", metavar="ENV_ID", help="Print the policy health port for one env id.")
    group.add_argument("--health-url-for-env", metavar="ENV_ID", help="Print the policy readyz URL for one env id.")
    group.add_argument(
        "--health-state-for-env", metavar="ENV_ID", help="Print ready policy health JSON for one env id."
    )
    group.add_argument(
        "--matches-health", metavar="ENV_ID", help="Exit 0 if the env's ready policy matches requested model flags."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-repo", default=None)
    parser.add_argument("--g1-model-repo", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--env", default=None, help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()

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
        return

    if args.health_port_for_env:
        print(get_policy_config(args.health_port_for_env).required_health_port)
        return

    if args.health_url_for_env:
        print(f"http://127.0.0.1:{get_policy_config(args.health_url_for_env).required_health_port}/readyz")
        return

    if args.health_state_for_env:
        data = _read_health(get_policy_config(args.health_state_for_env).required_health_port)
        if not _is_ready(data):
            raise SystemExit(1)
        print(json.dumps(data, sort_keys=True))
        return

    if args.matches_health:
        data = _read_health(get_policy_config(args.matches_health).required_health_port)
        if not _is_ready(data) or not _matches_request(args.matches_health, data, args):
            raise SystemExit(1)
        print(json.dumps(data, sort_keys=True))


def _read_health(port: int) -> dict:
    try:
        with urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def _is_ready(data: dict) -> bool:
    return data.get("state") in {"waiting_for_samples", "running"}


def _matches_request(env_id: str, data: dict, args: argparse.Namespace) -> bool:
    has_model_constraint = bool(args.model_path or args.model_repo or args.g1_model_repo or args.model_revision)

    # Older policy daemons may not publish metadata yet. The health port is already
    # env-specific, so a ready response is enough when no model override was requested.
    if not data.get("env") and not has_model_constraint:
        return True

    if data.get("env") != env_id:
        return False

    if args.model_path:
        served = data.get("model_path")
        if not served:
            return False
        if os.path.realpath(os.path.expanduser(served)) != os.path.realpath(os.path.expanduser(args.model_path)):
            return False

    model_repo = args.model_repo or args.g1_model_repo
    if model_repo and data.get("model_repo") != model_repo:
        return False

    if args.model_revision and data.get("model_revision") != args.model_revision:
        return False

    return True


if __name__ == "__main__":
    main()
