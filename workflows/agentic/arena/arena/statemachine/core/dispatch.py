# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI and dispatch glue between an env and its state-machine module."""

from __future__ import annotations

import argparse
import importlib
from typing import Any


def add_state_machine_cli_args(parser: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--state-machine",
        action="store_true",
        help=help_text or "Run this environment's scripted state-machine rollout instead of policy inference.",
    )


def configure_state_machine_args(
    args: argparse.Namespace,
    *,
    action_device: str | None = None,
    default_episodes: int = 1,
    disable_cameras: bool = False,
) -> None:
    if not getattr(args, "state_machine", False):
        return
    if action_device is not None:
        args.action_device = action_device
    if disable_cameras:
        args.disable_cameras = True
        args.enable_cameras = False
    if getattr(args, "episodes", 0) <= 0:
        args.episodes = default_episodes


def state_machine_requested(args: Any) -> bool:
    return bool(getattr(args, "state_machine", False) and not getattr(args, "replay_dataset_path", None))


def run_state_machine_module(
    module_path: str,
    *,
    args: argparse.Namespace,
    env: Any,
    app: Any,
    controller: Any,
) -> None:
    """Import ``module_path`` and hand off to its ``run_state_machine`` entry point."""
    module = importlib.import_module(module_path)
    run_state_machine = getattr(module, "run_state_machine", None)
    if run_state_machine is None:
        raise AttributeError(f"{module_path} must define run_state_machine(args=..., env=..., app=..., controller=...)")
    run_state_machine(args=args, env=env, app=app, controller=controller)
