# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""openpi PI0 fine-tuning entry for the ultrasound liver-scan env.

Two CLI surfaces are accepted:

* Full openpi-style run: ``--config``, ``--exp-name``, ``--repo-id`` (the
  registered training-config governs num_train_steps, batch_size, etc.).
* Pipeline / smoke run: ``--dataset-path``, ``--output-dir``,
  ``--max-steps``, ``--save-steps``, ``--batch-size``, ``--num-gpus`` —
  mirrors the gr00t_n15 train CLI so the e2e harness can drive both
  stacks the same way. These flags override the registered config via
  ``dataclasses.replace``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import sys


def _preconfigure_gpu_visibility() -> None:
    """Restrict CUDA_VISIBLE_DEVICES before JAX initializes.

    JAX can't run a single compiled executable across heterogeneous GPU
    architectures, and its default data-parallel sharding picks up every
    visible device. We honor ``--num-gpus N`` (defaulting to 1) by pinning
    visibility to the first N devices via a tiny argv pre-parse.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    n = 1
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--num-gpus" and i + 1 < len(argv):
            try:
                n = max(1, int(argv[i + 1]))
            except ValueError:
                pass
            break
        if tok.startswith("--num-gpus="):
            try:
                n = max(1, int(tok.split("=", 1)[1]))
            except ValueError:
                pass
            break
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_preconfigure_gpu_visibility()

from openpi import train  # noqa: E402
from openpi.training.config import DataConfigFactory  # noqa: E402
from policy.ultrasound_liver_scan.config import get_config  # noqa: E402
from policy.ultrasound_liver_scan.utils import compute_normalization_stats  # noqa: E402

logger = logging.getLogger("policy.ultrasound_liver_scan.train")


def _ensure_norm_stats(config) -> None:
    data_config = config.data
    if isinstance(data_config, DataConfigFactory):
        data_config = data_config.create(config.assets_dirs, config.model)

    output_path = config.assets_dirs / data_config.repo_id
    stats_file = output_path / "norm_stats.json"
    if not os.path.exists(stats_file):
        logger.info("normalisation stats missing at %s; computing", stats_file)
        compute_normalization_stats(config)
    else:
        logger.info("normalisation stats found at %s; skipping", stats_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI0 fine-tune for ultrasound liver-scan.")
    parser.add_argument("--config", default="robotic_ultrasound", help="Registered config name.")
    parser.add_argument(
        "--exp-name",
        dest="exp_name",
        default=None,
        help="Experiment name (checkpoints/logs). Defaults to 'e2e' when " "--dataset-path is given, else required.",
    )
    parser.add_argument(
        "--repo-id",
        dest="repo_id",
        default=None,
        help="Dataset repo id under HF_LEROBOT_HOME. If omitted with " "--dataset-path, derived from the path.",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        default=None,
        help="Path to a LeRobot dataset directory. The grandparent becomes "
        "HF_LEROBOT_HOME and 'parent/name' becomes the repo_id.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Checkpoint base dir (overrides config.checkpoint_base_dir). "
        "Assets (norm stats) are written under <output-dir>/assets.",
    )
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=None,
        help="Override num_train_steps for a smoke / short run.",
    )
    parser.add_argument("--save-steps", dest="save_steps", type=int, default=None, help="Override save_interval.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Override per-step batch size.")
    parser.add_argument(
        "--num-gpus",
        dest="num_gpus",
        type=int,
        default=1,
        help="Number of GPUs. Only 1 is supported in this entrypoint; >1 " "will FSDP-shard via openpi (fsdp_devices).",
    )
    args = parser.parse_args()

    if args.dataset_path is not None:
        ds = pathlib.Path(args.dataset_path).expanduser().resolve()
        if not ds.exists():
            parser.error(f"--dataset-path does not exist: {ds}")
        repo_id = f"{ds.parent.name}/{ds.name}"
        if args.repo_id is None:
            args.repo_id = repo_id
        elif args.repo_id != repo_id:
            logger.warning("--repo-id (%s) differs from path-derived id (%s); using --repo-id", args.repo_id, repo_id)
        # LeRobot resolves <root>/<repo_id> where root comes from LEROBOT_HOME (the
        # third_party/lerobot-6674e36 install vendored into this venv reads that
        # exact env var). The e2e harness exports HF_LEROBOT_HOME upstream; bridge
        # both names so either entry path works.
        os.environ["LEROBOT_HOME"] = str(ds.parent.parent)
        os.environ.setdefault("HF_LEROBOT_HOME", str(ds.parent.parent))
        if args.exp_name is None:
            args.exp_name = "e2e"

    if args.repo_id is None:
        args.repo_id = "i4h/robotic_ultrasound"
    if args.exp_name is None:
        parser.error("--exp-name is required (unless --dataset-path is given)")

    return args


def _apply_overrides(config, args: argparse.Namespace):
    overrides: dict = {}
    if args.max_steps is not None:
        overrides["num_train_steps"] = args.max_steps
    if args.save_steps is not None:
        overrides["save_interval"] = args.save_steps
        # keep_period must divide cleanly into save_interval or be None; for smoke runs we
        # don't need long-term retention, so disable the "keep every N" rule.
        overrides["keep_period"] = None
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.output_dir is not None:
        out = pathlib.Path(args.output_dir).expanduser().resolve()
        overrides["checkpoint_base_dir"] = str(out)
        overrides["assets_base_dir"] = str(out / "assets")
        # Pipeline runs always start from a fresh output dir; never silently resume.
        overrides["overwrite"] = True
        overrides["resume"] = False
        overrides["wandb_enabled"] = False
    if args.num_gpus and args.num_gpus > 1:
        overrides["fsdp_devices"] = args.num_gpus

    if not overrides:
        return config
    logger.info("applying CLI overrides to TrainConfig: %s", overrides)
    return dataclasses.replace(config, **overrides)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    args = _parse_args()
    config = get_config(name=args.config, repo_id=args.repo_id, exp_name=args.exp_name)
    config = _apply_overrides(config, args)
    _ensure_norm_stats(config)
    train.main(config)


if __name__ == "__main__":
    main()
