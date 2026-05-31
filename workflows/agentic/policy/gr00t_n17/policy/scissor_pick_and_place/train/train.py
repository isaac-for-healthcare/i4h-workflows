# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 fine-tuning entry for the SO-ARM scissor pick-and-place env.

Wraps Isaac-GR00T's ``launch_finetune.py``. Defaults mirror
``workflows/so_arm_starter/scripts/training/gr00t_n1_7/train.py``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro

logger = logging.getLogger("policy.scissor_pick_and_place.train")


@dataclass
class TrainConfig:
    """Configuration for GR00T N1.7 fine-tuning on a SO-ARM LeRobot dataset.

    Wraps Isaac-GR00T's ``launch_finetune.py``; defaults mirror
    ``workflows/so_arm_starter/scripts/training/gr00t_n1_7/train.py``.
    """

    dataset_path: List[str]
    """Path to the dataset directory (LeRobot format)."""

    output_dir: str = "/tmp/gr00t_so_arm"
    """Directory to save model checkpoints."""

    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    """Path or HuggingFace model ID for the base model."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to use for training."""

    batch_size: int = 32
    """Global batch size for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard"] = "tensorboard"
    """Where to report training metrics."""

    state_dropout_prob: float = 0.2
    """Probability of dropping out state inputs during training."""

    modality_config_path: str | None = None
    """Override the local config.py with a different modality config (debug)."""


def _groot_root() -> Path:
    # train.py -> train -> scissor_pick_and_place -> policy (pkg) -> gr00t_n17 -> policy -> agentic
    return Path(__file__).resolve().parents[5] / "third_party" / "Isaac-GR00T-1.7"


def _modality_config_path() -> Path:
    # Same registration used at inference — launch_finetune.py loads this
    # file by path, fires ``register_modality_config(...)`` as a side effect.
    return Path(__file__).resolve().parents[1] / "config.py"


def _build_cmd(cfg: TrainConfig, launch_script: Path, dataset_path: str) -> list[str]:
    # Route through our shim for argv setup; no gr00t modifications.
    cmd = [
        sys.executable,
        "-m",
        "policy.scissor_pick_and_place.train._launcher",
        str(launch_script),
        "--base_model_path",
        cfg.base_model_path,
        "--dataset_path",
        dataset_path,
        "--embodiment_tag",
        cfg.embodiment_tag,
        "--output_dir",
        cfg.output_dir,
        "--save_steps",
        str(cfg.save_steps),
        "--max_steps",
        str(cfg.max_steps),
        "--warmup_ratio",
        str(cfg.warmup_ratio),
        "--weight_decay",
        str(cfg.weight_decay),
        "--learning_rate",
        str(cfg.learning_rate),
        "--global_batch_size",
        str(cfg.batch_size),
        "--dataloader_num_workers",
        str(cfg.dataloader_num_workers),
        "--num_gpus",
        str(cfg.num_gpus),
        "--state_dropout_prob",
        str(cfg.state_dropout_prob),
        "--modality_config_path",
        cfg.modality_config_path or str(_modality_config_path()),
    ]
    if cfg.tune_llm:
        cmd.append("--tune_llm")
    if cfg.tune_visual:
        cmd.append("--tune_visual")
    if not cfg.tune_projector:
        cmd.append("--no-tune_projector")
    if not cfg.tune_diffusion_model:
        cmd.append("--no-tune_diffusion_model")
    if cfg.report_to == "wandb":
        cmd.append("--use_wandb")
    return cmd


def run(cfg: TrainConfig) -> None:
    if not cfg.dataset_path:
        raise SystemExit("--dataset-path is required")
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if cfg.num_gpus > available_gpus:
        raise SystemExit(f"requested {cfg.num_gpus} GPUs but only {available_gpus} available")

    groot_root = _groot_root()
    launch_script = groot_root / "gr00t" / "experiment" / "launch_finetune.py"
    if not launch_script.exists():
        raise SystemExit(f"Isaac-GR00T launch_finetune.py not found at {launch_script}. Run ./setup.sh first.")

    logger.info("GR00T N1.7 scissor fine-tune configuration:")
    for k, v in vars(cfg).items():
        logger.info("  %s = %s", k, v)

    cmd = _build_cmd(cfg, launch_script, cfg.dataset_path[0])
    logger.info("Running: %s", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=str(groot_root)).returncode)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    run(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
