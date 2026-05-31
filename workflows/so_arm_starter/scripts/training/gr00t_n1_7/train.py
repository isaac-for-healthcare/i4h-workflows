# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro


@dataclass
class ArgsConfig:
    """Configuration for GR00T N1.7 fine-tuning.

    This wraps the upstream ``launch_finetune.py`` from Isaac-GR00T,
    providing a convenient CLI for N1.7 fine-tuning.
    """

    dataset_path: List[str]
    """Path to the dataset directory (LeRobot format)."""

    output_dir: str = "/tmp/gr00t_n1_7"
    """Directory to save model checkpoints."""

    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    """Path or HuggingFace model ID for the base model."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to use for training."""

    batch_size: int = 32
    """Global batch size for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

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

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for training."""

    modality_config_path: str | None = None
    """Path to a custom modality config Python file. If None, uses the default."""

    state_dropout_prob: float = 0.2
    """Probability of dropping out state inputs during training."""


def _find_groot_root() -> Path:
    """Resolve Isaac-GR00T repo root relative to this file."""
    # train.py -> gr00t_n1_7 -> training -> scripts -> so_arm_starter
    #          -> workflows -> i4h-workflows-internal
    return Path(__file__).resolve().parents[5] / "third_party" / "Isaac-GR00T"


def main(config: ArgsConfig):
    groot_root = _find_groot_root()
    launch_script = groot_root / "gr00t" / "experiment" / "launch_finetune.py"

    if not launch_script.exists():
        raise FileNotFoundError(
            f"Isaac-GR00T launch_finetune.py not found at {launch_script}. "
            "Make sure third_party/Isaac-GR00T is cloned."
        )

    cmd = [
        sys.executable,
        str(launch_script),
        "--base_model_path",
        config.base_model_path,
        "--dataset_path",
        config.dataset_path[0],
        "--embodiment_tag",
        config.embodiment_tag,
        "--output_dir",
        config.output_dir,
        "--save_steps",
        str(config.save_steps),
        "--max_steps",
        str(config.max_steps),
        "--warmup_ratio",
        str(config.warmup_ratio),
        "--weight_decay",
        str(config.weight_decay),
        "--learning_rate",
        str(config.learning_rate),
        "--global_batch_size",
        str(config.batch_size),
        "--dataloader_num_workers",
        str(config.dataloader_num_workers),
        "--num_gpus",
        str(config.num_gpus),
        "--state_dropout_prob",
        str(config.state_dropout_prob),
    ]

    if config.tune_llm:
        cmd.append("--tune_llm")
    if config.tune_visual:
        cmd.append("--tune_visual")
    if not config.tune_projector:
        cmd.append("--no-tune_projector")
    if not config.tune_diffusion_model:
        cmd.append("--no-tune_diffusion_model")
    if config.modality_config_path:
        cmd.extend(["--modality_config_path", str(Path(config.modality_config_path).resolve())])
    if config.report_to == "wandb":
        cmd.append("--use_wandb")

    print("=" * 60)
    print("GR00T N1.7 Fine-Tuning")
    print("=" * 60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")

    sys.exit(subprocess.run(cmd, cwd=str(groot_root)).returncode)


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    assert config.num_gpus > 0, "num_gpus must be > 0"

    main(config)
