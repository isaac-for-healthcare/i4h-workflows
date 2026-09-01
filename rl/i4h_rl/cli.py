# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight user-facing resolver for profile-selected RL backends."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from i4h_rl.artifacts import resolve_output_path
from i4h_rl.backend_loader import load_backend
from i4h_rl.contract import validate_workflow_contract
from i4h_rl.profile import ProfileError, RLProfile, available_profiles, load_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.sh rl", description="Train a workflow policy with RL.")
    parser.add_argument("workflow", help="workflow name, or list/show")
    parser.add_argument("show_workflow", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", help="local starting checkpoint for trainers that require one")
    parser.add_argument(
        "--checkpoint",
        "--rl-model-path",
        dest="rl_model_path",
        help="native trainer checkpoint or run bundle used for evaluation/export",
    )
    parser.add_argument("--output-dir", help="exported policy/checkpoint destination")
    parser.add_argument("--train-config", help="resolved RLinf config.yaml used to create a checkpoint")
    parser.add_argument("--resume-dir", help="supported trainer run directory to resume")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--episodes", type=int, help="evaluation episodes for backends that expose episode counts")
    parser.add_argument("--run-dir")
    parser.add_argument("--eval", action="store_true", dest="only_eval")
    parser.add_argument("--video", action="store_true", help="record supported trainer output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--runtime-python",
        help="trainer Python for RLinf, or the combined simulator/trainer Python for RSL-RL",
    )
    parser.add_argument("--sim-runtime-python", help="Isaac Sim Python used by an isolated RLinf simulator")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides")
    return parser


def _render(profile: RLProfile, *, num_envs: int, epochs: int) -> str:
    observation_summary = f"{profile.state_dof}-D state"
    if profile.cameras:
        observation_summary += " + " + ", ".join(profile.cameras)
    return "\n".join(
        (
            f"workflow: {profile.workflow}",
            f"scene: {profile.scene}",
            f"trainer: {profile.trainer} ({profile.algorithm})",
            f"adapter: {profile.adapter_module or 'none'}",
            f"train task: {profile.train_task_id}",
            f"eval task: {profile.eval_task_id}",
            f"instruction: {profile.task_description}",
            f"observations: {observation_summary}",
            f"actions: {profile.policy_action_dof}-D policy -> {profile.action_dof}-D scene",
            f"parallel envs: {num_envs}",
            f"max epochs: {epochs}",
            f"trainer config: {profile.trainer_config}",
        )
    )


def _validate_override(value: str) -> str:
    key, separator, setting = value.partition("=")
    if not separator or not key.strip() or not setting.strip():
        raise SystemExit(f"--set requires KEY=VALUE, got {value!r}")
    return value


def _workflows_root() -> Path:
    return Path(os.environ.get("I4H_WORKFLOWS", Path(__file__).resolve().parents[2])).resolve()


def _export(args: argparse.Namespace, profile: RLProfile, workflows_root: Path) -> int:
    unsupported = {
        "--model-path": args.model_path,
        "--resume-dir": args.resume_dir,
        "--run-dir": args.run_dir,
        "--eval": args.only_eval,
        "--video": args.video,
        "--dry-run": args.dry_run,
        "--set": args.overrides,
        "--num-envs": args.num_envs,
        "--epochs": args.epochs,
        "--episodes": args.episodes,
        "--sim-runtime-python": args.sim_runtime_python,
    }
    used = [name for name, value in unsupported.items() if value]
    if used:
        raise SystemExit(f"export does not accept {', '.join(used)}")
    if not args.rl_model_path:
        raise SystemExit("export requires --checkpoint")
    if not args.output_dir:
        raise SystemExit("export requires --output-dir")
    output = resolve_output_path(workflows_root, args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"--output-dir must be absent or empty: {output}")
    backend = load_backend(profile.trainer)
    validate_workflow_contract(profile, workflows_root)
    backend.validate_profile(profile, workflows_root)
    return backend.export(args, profile, workflows_root)


def _launch(args: argparse.Namespace, profile: RLProfile, workflows_root: Path) -> int:
    num_envs = args.num_envs if args.num_envs is not None else profile.default_num_envs
    epochs = args.epochs if args.epochs is not None else profile.default_epochs
    episodes = args.episodes if args.episodes is not None else 20
    if num_envs < 1 or epochs < 1:
        raise SystemExit("--num-envs and --epochs must be positive")
    if episodes < 1:
        raise SystemExit("--episodes must be positive")
    if args.output_dir:
        raise SystemExit("--output-dir is only valid with the export operation")
    if args.train_config:
        raise SystemExit("--train-config is only valid with the export operation")
    args.episodes = episodes
    overrides = tuple(_validate_override(value) for value in args.overrides)
    backend = load_backend(profile.trainer)
    validate_workflow_contract(profile, workflows_root)
    backend.validate_profile(profile, workflows_root)
    print(_render(profile, num_envs=num_envs, epochs=epochs))
    backend.validate_launch(args, profile, workflows_root)
    print("operation: evaluation" if args.only_eval else "operation: RL training")
    if overrides:
        print("overrides: " + " ".join(overrides))
    if args.dry_run:
        print("dry-run: configuration resolved; simulator and trainer were not launched")
        return 0
    return backend.launch(
        args,
        profile,
        workflows_root,
        num_envs=num_envs,
        epochs=epochs,
        overrides=overrides,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.workflow == "list":
            if args.show_workflow:
                raise SystemExit("list does not take a workflow")
            for name, path in available_profiles().items():
                profile = RLProfile.load(path)
                print(f"{name}\t{profile.trainer}\t{profile.algorithm}\t{profile.scene}")
            return 0
        if args.workflow not in {"show", "export"} and args.show_workflow:
            raise SystemExit(f"unexpected positional argument: {args.show_workflow}")
        workflow = args.show_workflow if args.workflow in {"show", "export"} else args.workflow
        if not workflow:
            raise SystemExit(f"{args.workflow} requires a workflow")
        profile = load_profile(workflow)
        if args.workflow == "show":
            workflows_root = _workflows_root()
            backend = load_backend(profile.trainer)
            validate_workflow_contract(profile, workflows_root)
            backend.validate_profile(profile, workflows_root)
            print(_render(profile, num_envs=profile.default_num_envs, epochs=profile.default_epochs))
            return 0
        workflows_root = _workflows_root()
        if args.workflow == "export":
            return _export(args, profile, workflows_root)
        return _launch(args, profile, workflows_root)
    except (ProfileError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    sys.exit(main())
