# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finetuning entry for the gr00t_n17_tactile stack.

    uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile-train \\
        --task gr00t_n17_tactile/h2_sharpa_tactile --dataset <lerobot-dir> [--output-dir ...] [--dry-run]

Defaults come from the task manifest the inference server reads. The dataset is
what changes between manipulation tasks; the recipe does not, which is why it
is an argument here rather than a manifest per object.

The launcher is Isaac-GR00T's ``launch_finetune.py`` with the tactile patch
series applied (``patches/README.md``). This module builds its argument list
from the manifest; ``--dry-run`` prints it and stops.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from i4h_tasks.gr00t_n17_tactile.tactile.sharpa import asset_uris

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.train")

PROJECT = "gr00t_n17_tactile"
DEFAULT_TASK = f"{PROJECT}/h2_sharpa_tactile"
#: Second checkout of the pinned Isaac-GR00T N1.7 revision, with the patch series.
GROOT_CHECKOUT = "third_party/Isaac-GR00T-1.7-tactile"
LAUNCH_SCRIPT = "gr00t/experiment/launch_finetune.py"

#: Manifest ``train`` keys that pass straight through, and the launcher flag each becomes.
_PASSTHROUGH = {
    "max_steps": "--max_steps",
    "save_steps": "--save_steps",
    "batch_size": "--global_batch_size",
    "num_gpus": "--num_gpus",
    "learning_rate": "--learning_rate",
    "state_dropout_prob": "--state_dropout_prob",
}


def modality_config_file(module: str) -> Path:
    """Resolve ``model.modality_config`` to the file the launcher loads by path."""
    path = Path(__file__).resolve().parent / f"{module}.py"
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("config*.py"))
        raise FileNotFoundError(f"no modality config {module!r} at {path}; have {available}")
    return path


def finetune_argv(spec: Any, dataset: str, overrides: dict[str, Any] | None = None) -> list[str]:
    """The launcher's argument list for one dataset.

    Pure: nothing is resolved or downloaded, so ``--dry-run`` shows exactly what
    a run would do. Asset entries are URIs (``hf://`` or a ``third_party/`` path)
    that the run resolves right before launching.
    """
    train = {**spec.train, **(overrides or {})}
    tactile = spec.model.get("tactile") or {}
    assets = asset_uris(tactile)
    argv = [
        LAUNCH_SCRIPT,
        "--base_model_path",
        str(train.get("base_model", "nvidia/GR00T-N1.7-3B")),
        "--dataset_path",
        dataset,
        "--embodiment_tag",
        str(spec.model.get("embodiment_tag", "new_embodiment")),
        "--output_dir",
        str(train.get("output_dir", "/tmp/gr00t_h2_sharpa_tactile")),
        "--modality_config_path",
        str(modality_config_file(str(spec.model.get("modality_config", "config")))),
    ]
    for key, flag in _PASSTHROUGH.items():
        if train.get(key) is not None:
            argv += [flag, str(train[key])]
    argv.append("--tune_visual" if train.get("tune_visual", True) else "--no-tune_visual")
    if not train.get("use_tactile", True):
        return argv
    argv += [
        "--use_tactile",
        "--num_tactile_tokens",
        str(tactile.get("num_tokens", 10)),
        "--sharpa_encoder_ckpt_path",
        assets["deform_encoder"],
        "--sharpa_left_urdf_path",
        assets["left_urdf"],
        "--sharpa_right_urdf_path",
        assets["right_urdf"],
        "--tactile_dropout_prob",
        str(tactile.get("dropout_prob", 0.0)),
    ]
    force = tactile.get("force") or {}
    if force.get("enabled"):
        argv += [
            "--use_tactile_force",
            "--force_window",
            str(force.get("window", 16)),
            "--trex_vqvae_ckpt_path",
            assets["force_encoder"],
        ]
    delay = tactile.get("delay_aug") or {}
    if delay.get("enabled"):
        argv += [
            "--delay_aug",
            "--tau_split",
            str(delay.get("tau_split", 0.5)),
            "--force_delay_max",
            str(delay.get("force_delay_max", 16)),
        ]
    return argv


def run(argv: list[str]) -> None:
    """Launch the patched finetune with ``argv`` from the GR00T checkout."""
    raise NotImplementedError(
        f"the finetune launch ({GROOT_CHECKOUT}/{argv[0]}) lands with the GR00T patch series; "
        "see tasks/gr00t_n17_tactile/patches/README.md"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gr00t_n17_tactile-train", description="Finetune a tactile N1.7 checkpoint.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="task id, e.g. gr00t_n17_tactile/h2_sharpa_tactile")
    parser.add_argument("--dataset", required=True, help="LeRobot v2.x dataset dir with the tactile columns")
    parser.add_argument("--output-dir", default=None, help="overrides [task.train].output_dir")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--base-model", default=None, help="overrides [task.train].base_model")
    parser.add_argument("--tune-visual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--tactile", action=argparse.BooleanOptionalAction, default=None, help="vision-only baseline when off"
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve the manifest, print the launcher argv, stop")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="[gr00t_n17_tactile-train] %(message)s"
    )

    from i4h_common.training import require_trainable, resolve_dataset

    try:
        spec = require_trainable(args.task)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.task.partition("/")[0] != PROJECT:
        print(f"error: {args.task} is not served by {PROJECT!r}; run that project's trainer instead.", file=sys.stderr)
        return 2
    overrides = {
        key: value
        for key, value in (
            ("output_dir", args.output_dir),
            ("max_steps", args.max_steps),
            ("save_steps", args.save_steps),
            ("batch_size", args.batch_size),
            ("num_gpus", args.num_gpus),
            ("learning_rate", args.learning_rate),
            ("base_model", args.base_model),
            ("tune_visual", args.tune_visual),
            ("use_tactile", args.tactile),
        )
        if value is not None
    }
    launcher_argv = finetune_argv(spec, resolve_dataset(args.dataset), overrides)
    if args.dry_run:
        print(f"cd {GROOT_CHECKOUT} && python " + " ".join(launcher_argv))
        return 0
    run(launcher_argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
