# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finetuning entry for the openpi PI0 stack.

    uv run --project tasks/openpi_pi0 i4h-openpi-pi0-train \
        --task openpi_pi0/ultrasound_liver_scan --dataset <lerobot-dir>

openpi owns its own training loop and config registry, so this is a thin shim:
it resolves defaults from the task manifest — the same block the inference
server reads — and hands off.

Weights & Biases logging is opt-in, as it is for the GR00T trainers. openpi's
``TrainConfig`` turns it on by default and then logs in at start-up, which on a
host without a W&B account ends the run before the first step. Turn it on with
``--wandb`` or ``train.wandb: true`` in the task manifest; the shell's
environment never decides it. ``WANDB_MODE`` still says how a requested run
logs: ``offline`` writes it locally without an account, and ``disabled`` turns
logging off.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from i4h_tasks.openpi_pi0.runtime import configure_xla, cuda_compute_capability

logger = logging.getLogger("i4h_tasks.openpi_pi0.train")

#: openpi's experiment name for every fine-tune this shim starts; checkpoints
#: land in ``<output_dir>/<config>/<EXP_NAME>``.
EXP_NAME = "finetune"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openpi-pi0-train", description="Finetune a PI0 checkpoint.")
    parser.add_argument("--task", required=True, help="task id, e.g. openpi_pi0/ultrasound_liver_scan")
    parser.add_argument(
        "--dataset",
        "--dataset-path",
        dest="dataset",
        required=True,
        help="local LeRobot dataset directory",
    )
    parser.add_argument("--config", default=None, help="openpi training config name; defaults to [task.train].config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="log to Weights & Biases (default: the task manifest's train.wandb, else off)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def wandb_enabled(
    flag: bool | None, train: Mapping[str, Any], environ: Mapping[str, str] = os.environ
) -> tuple[bool, str]:
    """Whether to log to W&B, and what decided it.

    ``--wandb``/``--no-wandb`` wins, then the manifest's ``train.wandb``; both
    default to off. A requested run still honours ``WANDB_MODE=disabled``, so
    openpi records no W&B run id that a later resume would look for.
    """
    if flag is not None:
        enabled, reason = flag, ("--wandb" if flag else "--no-wandb")
    else:
        declared = train.get("wandb", False)
        if not isinstance(declared, bool):
            raise ValueError(f"train.wandb must be true or false, got {declared!r}")
        enabled, reason = declared, f"train.wandb: {str(declared).lower()} in the task manifest"
    mode = environ.get("WANDB_MODE", "").strip().lower()
    if enabled and mode == "disabled":
        return False, "WANDB_MODE=disabled"
    if enabled and mode == "offline":
        reason += ", WANDB_MODE=offline"
    return enabled, reason


def resumes_without_wandb_run(checkpoint_dir: Path, *, overwrite: bool) -> bool:
    """Whether openpi will resume a run that was trained with W&B off.

    openpi resumes when the checkpoint directory holds a step past 0, and then
    reads the W&B run id from ``wandb_id.txt``, which it writes only when W&B
    is on. Turning W&B on for such a resume fails with ``FileNotFoundError``.
    """
    if overwrite or not checkpoint_dir.is_dir():
        return False
    steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())
    return steps not in ([], [0]) and not (checkpoint_dir / "wandb_id.txt").is_file()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[openpi-pi0-train] %(message)s")

    from i4h_common.training import require_trainable

    try:
        spec = require_trainable(args.task)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    project = args.task.partition("/")[0]
    if project != "openpi_pi0":
        print(f"error: {args.task} is served by {project!r}, not 'openpi_pi0'.", file=sys.stderr)
        return 2

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        print(f"error: local LeRobot dataset not found: {dataset}", file=sys.stderr)
        return 2

    # LeRobot resolves <LEROBOT_HOME>/<repo_id>. A converted local dataset is
    # conventionally <root>/<owner>/<name>, so expose those two pieces.
    repo_id = f"{dataset.parent.name}/{dataset.name}"
    os.environ["LEROBOT_HOME"] = str(dataset.parent.parent)
    os.environ.setdefault("HF_LEROBOT_HOME", str(dataset.parent.parent))

    config_name = args.config or str(spec.train.get("config", "robotic_ultrasound"))
    output_dir = Path(args.output_dir or spec.train.get("output_dir", "/tmp/pi0_liver_scan")).expanduser().resolve()
    max_steps = args.max_steps if args.max_steps is not None else int(spec.train.get("max_steps", 30_000))
    batch_size = args.batch_size if args.batch_size is not None else int(spec.train.get("batch_size", 32))
    try:
        use_wandb, wandb_reason = wandb_enabled(args.wandb, spec.train)
    except ValueError as exc:
        print(f"error: {args.task}: {exc}", file=sys.stderr)
        return 2
    checkpoint_dir = output_dir / config_name / EXP_NAME
    if use_wandb and resumes_without_wandb_run(checkpoint_dir, overwrite=args.overwrite):
        logger.warning(
            "%s holds a run trained without Weights & Biases, so there is no W&B run to resume; logging stays "
            "off. Pass --overwrite or a new --output-dir to start a logged run.",
            checkpoint_dir,
        )
        use_wandb, wandb_reason = False, "resuming a run trained without W&B"

    resolved = {
        "task": args.task,
        "config": config_name,
        "dataset": str(dataset),
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "num_gpus": args.num_gpus,
        "wandb": use_wandb,
        "wandb_reason": wandb_reason,
    }
    if args.dry_run:
        for key, value in resolved.items():
            print(f"  {key} = {value!r}")
        return 0

    logger.info("finetuning %s with openpi config %s", args.task, config_name)
    if use_wandb:
        logger.info("Weights & Biases logging is on (%s)", wandb_reason)
    else:
        logger.info("Weights & Biases logging is off (%s); pass --wandb to log there", wandb_reason)
    configure_xla(cuda_compute_capability())
    from openpi import train as openpi_train  # noqa: PLC0415
    from openpi.training.config import DataConfigFactory  # noqa: PLC0415

    from i4h_tasks.openpi_pi0.ultrasound.config import get_config  # noqa: PLC0415
    from i4h_tasks.openpi_pi0.ultrasound.utils import compute_normalization_stats  # noqa: PLC0415

    overrides = {
        "checkpoint_base_dir": str(output_dir),
        "assets_base_dir": str(output_dir / "assets"),
        "batch_size": batch_size,
        "num_train_steps": max_steps,
        "fsdp_devices": args.num_gpus,
        "overwrite": args.overwrite,
        "resume": not args.overwrite,
        "wandb_enabled": use_wandb,
    }
    if args.save_steps is not None:
        overrides.update(save_interval=args.save_steps, keep_period=None)
    cfg = dataclasses.replace(
        get_config(name=config_name, repo_id=repo_id, exp_name=EXP_NAME),
        **overrides,
    )

    data_config = cfg.data
    if isinstance(data_config, DataConfigFactory):
        data_config = data_config.create(cfg.assets_dirs, cfg.model)
    stats_file = cfg.assets_dirs / str(data_config.repo_id) / "norm_stats.json"
    if not stats_file.exists():
        compute_normalization_stats(cfg)
    openpi_train.main(cfg)
    print(f"finetuned {args.task} -> {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
