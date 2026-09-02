# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finetuning entry for the gr00t_n17 stack.

    uv run --project tasks/gr00t_n17 i4h-gr00t-n17 -train \\
        --task gr00t_n17/<name> --dataset <lerobot-dir> [--output-dir ...]

Defaults come from the same task manifest the inference server reads, keeping
training and serving configuration aligned.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("i4h_tasks.gr00t_n17.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gr00t_n17-train", description="Finetune a gr00t_n17 checkpoint.")
    parser.add_argument("--task", required=True, help="task id, e.g. gr00t_n17/scissor_pick_and_place")
    parser.add_argument("--dataset", required=True, nargs="+", help="LeRobot dataset dir(s) or HF repo id(s)")
    parser.add_argument("--output-dir", default=None, help="overrides [task.train].output_dir")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--base-model", default=None, help="overrides [task.train].base_model")
    parser.add_argument("--tune-visual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-projector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-diffusion-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="resolve config and print it, then stop")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[gr00t_n17-train] %(message)s")

    from i4h_common.training import require_trainable, resolve_dataset

    try:
        task_spec = require_trainable(args.task)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    project = args.task.partition("/")[0]
    if project != "gr00t_n17":
        print(
            f"error: {args.task} is served by {project!r}, not 'gr00t_n17'. Run that project's trainer instead.",
            file=sys.stderr,
        )
        return 2

    from i4h_tasks.gr00t_n17 import _finetune

    _finetune._TASK_ID = args.task
    overrides = {
        k: v
        for k, v in (
            ("output_dir", args.output_dir),
            ("max_steps", args.max_steps),
            ("save_steps", args.save_steps),
            ("batch_size", args.batch_size),
            ("num_gpus", args.num_gpus),
            ("base_model_path", args.base_model),
            ("tune_visual", args.tune_visual),
            ("tune_projector", args.tune_projector),
            ("tune_diffusion_model", args.tune_diffusion_model),
        )
        if v is not None
    }
    for field_name in ("output_dir", "max_steps", "save_steps", "batch_size", "tune_visual"):
        if field_name not in overrides and task_spec.train.get(field_name) is not None:
            overrides[field_name] = task_spec.train[field_name]
    cfg = _finetune.TrainConfig(dataset_path=[resolve_dataset(d) for d in args.dataset], **overrides)

    if args.dry_run:
        for field_name in sorted(vars(cfg)):
            print(f"  {field_name} = {getattr(cfg, field_name)!r}")
        return 0

    _finetune.run(cfg)
    print(f"finetuned {args.task} -> {cfg.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
