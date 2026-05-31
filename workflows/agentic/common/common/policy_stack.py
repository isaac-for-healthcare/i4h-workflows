# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from common.config import environment_config_path, get_environment_config, get_policy_config, policy_routings_for_stack


@dataclass(frozen=True)
class PolicyBackend:
    env_id: str
    description: str
    infer_module: str
    train_module: str | None = None
    add_args_attr: str = "add_args"
    run_attr: str = "run"

    @property
    def train_module_path(self) -> str:
        return self.train_module or f"policy.{self.env_id}.train.train"

    def load_run(self) -> Callable[[argparse.Namespace], None]:
        return getattr(importlib.import_module(self.infer_module), self.run_attr)

    def load_add_args(self) -> Callable[[argparse.ArgumentParser], None]:
        return getattr(importlib.import_module(self.infer_module), self.add_args_attr)


def build_backends(stack: str) -> tuple[PolicyBackend, ...]:
    return tuple(
        PolicyBackend(
            env_id=routing.env_id,
            description=routing.description,
            infer_module=routing.infer_module,
            train_module=routing.train_module,
            add_args_attr=routing.add_args_attr,
            run_attr=routing.run_attr,
        )
        for routing in policy_routings_for_stack(stack)
    )


def backend_map(backends: tuple[PolicyBackend, ...]) -> dict[str, PolicyBackend]:
    return {backend.env_id: backend for backend in backends}


def get_backend(backends: tuple[PolicyBackend, ...], env_id: str) -> PolicyBackend:
    by_env = backend_map(backends)
    try:
        return by_env[env_id]
    except KeyError as exc:
        choices = ", ".join(sorted(by_env))
        raise KeyError(f"unknown policy env {env_id!r}; choose one of: {choices}") from exc


def known_env_ids(backends: tuple[PolicyBackend, ...]) -> tuple[str, ...]:
    return tuple(backend.env_id for backend in backends)


def policy_default(env_id: str, field_name: str, fallback=None):
    try:
        return getattr(get_policy_config(env_id), field_name) or fallback
    except Exception:
        return fallback


def policy_train_default(env_id: str, field_name: str, fallback=None):
    try:
        policy_config = get_environment_config(env_id).get("policy") or {}
        train_config = policy_config.get("train") or {}
        return train_config.get(field_name, fallback)
    except Exception:
        return fallback


def default_base_model(env_id: str, fallback: str) -> str:
    return policy_default(env_id, "model_repo", fallback)


def policy_cli_main(backends: tuple[PolicyBackend, ...]) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    warnings.filterwarnings("ignore", message="Accessing config attribute `interleave_self_attention`.*")

    base, _ = _base_parser().parse_known_args()
    _configure_logging(base.verbose)

    if base.list_envs:
        for env in known_env_ids(backends):
            print(env)
        return
    if not base.env:
        print("--env is required (use --list-envs to see options)", file=sys.stderr)
        raise SystemExit(2)

    try:
        backend = get_backend(backends, base.env)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    if base.dry_run:
        print(
            f"[agentic-policy] dry run ok: env={base.env} module={backend.infer_module} config={environment_config_path()}"
        )
        return

    add_args = backend.load_add_args()
    run = backend.load_run()
    args = _full_parser(base.env, add_args).parse_args(sys.argv[1:])
    run(args)


def train_cli_main(backends: tuple[PolicyBackend, ...]) -> None:
    trainable_backends = tuple(backend for backend in backends if backend.train_module is not None)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--env", choices=known_env_ids(trainable_backends), required=True)
    known, rest = base.parse_known_args()

    os.environ["AGENTIC_POLICY_ENV_ID"] = known.env
    module = importlib.import_module(get_backend(trainable_backends, known.env).train_module_path)
    sys.argv = [sys.argv[0], *rest]
    module.main()


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", default=None)
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _full_parser(env: str, add_args: Callable[[argparse.ArgumentParser], Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the agentic policy service for env={env}.", conflict_handler="resolve"
    )
    parser.add_argument("--env", default=env, choices=(env,))
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-repo", type=str, default=None)
    parser.add_argument("--model-revision", type=str, default=None)
    add_args(parser)
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )
    if verbose:
        return
    for logger_name in (
        "transformers",
        "transformers_modules",
        "transformers_modules.eagle2_hg_model",
        "huggingface_hub",
        "PIL",
        "matplotlib",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_warning()
    except Exception:
        pass
