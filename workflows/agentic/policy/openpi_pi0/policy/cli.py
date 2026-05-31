# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry-driven policy CLI for the agentic workflow.

Reads ``--env``, looks up the matching :class:`PolicyBackend` in
:mod:`policy.registry`, imports its infer module, parses ``argv`` against
the module's argument schema, and calls ``infer.run(args)``.

Adding a new env's policy requires no edits here.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

from common.config import environment_config_path
from policy.registry import get_backend, known_env_ids


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env", default=None)
    p.add_argument("--list-envs", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _full_parser(env: str, add_args) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Run the agentic policy service for env={env}.", conflict_handler="resolve"
    )
    p.add_argument("--env", default=env, choices=(env,))
    p.add_argument("--list-envs", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--model-repo", type=str, default=None)
    p.add_argument("--model-revision", type=str, default=None)
    add_args(p)
    return p


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )
    if verbose:
        return
    for logger_name in (
        "absl",
        "flax",
        "jax",
        "huggingface_hub",
        "PIL",
        "matplotlib",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    warnings.filterwarnings("ignore", message="Accessing config attribute `interleave_self_attention`.*")

    base, _ = _base_parser().parse_known_args()
    _configure_logging(base.verbose)

    if base.list_envs:
        for env in known_env_ids():
            print(env)
        return
    if not base.env:
        print("--env is required (use --list-envs to see options)", file=sys.stderr)
        raise SystemExit(2)

    try:
        backend = get_backend(base.env)
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


if __name__ == "__main__":
    main()
