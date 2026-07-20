# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.5 fine-tuning entry for the SO-ARM scissor pick-and-place env.

Runs in-process via ``GR00T_N1_5.from_pretrained`` + ``TrainRunner`` (no
subprocess into Isaac-GR00T's launch_finetune.py — that helper is N1.7-only).
Mirrors ``workflows/soarm/training/training/train.py`` at commit c578a30.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro
from common.policy_stack import default_base_model, policy_default, policy_train_default
from huggingface_hub.constants import HF_HOME

logger = logging.getLogger("policy.scissor_pick_and_place.train")
_ENV_ID = "scissor_pick_and_place"


def _quiet_noisy_dependencies() -> None:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    for module, message in (
        (r"torchvision\.io\._video_deprecation_warning", "The video decoding and encoding capabilities.*"),
        (r"albumentations.*", "A new version of Albumentations is available.*"),
        (r"torch\.backends.*", "Please use the new API settings to control TF32 behavior.*"),
    ):
        warnings.filterwarnings("ignore", message=message, category=UserWarning, module=module)
    logging.getLogger("transformers_modules.eagle2_hg_model").setLevel(logging.WARNING)


@dataclass
class TrainConfig:
    dataset_path: list[str]

    output_dir: str = policy_train_default(_ENV_ID, "output_dir", "/tmp/gr00t_so_arm")
    data_config: str = policy_default(_ENV_ID, "data_config", "so100_dualcam")

    batch_size: int = 32
    max_steps: int = policy_train_default(_ENV_ID, "max_steps", 10000)
    save_steps: int = policy_train_default(_ENV_ID, "save_steps", 1000)
    num_gpus: int = 1

    base_model_path: str = field(default_factory=lambda: default_base_model(_ENV_ID, "nvidia/GR00T-N1.5-3B"))

    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False

    dataloader_num_workers: int = 8
    report_to: Literal["wandb", "tensorboard"] = "tensorboard"
    embodiment_tag: str = "new_embodiment"
    video_backend: Literal["decord", "torchvision_av"] = "decord"
    resume: bool = False
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    save_final_model: bool = False


def _train(cfg: TrainConfig) -> None:
    from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment import runner as gr00t_runner
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.utils.peft import get_lora_model
    from transformers import TrainingArguments

    embodiment_tag = EmbodimentTag(cfg.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[cfg.data_config]
    data_config_cls.video_keys = ["video.room", "video.wrist"]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    def single_dataset(path: str) -> LeRobotSingleDataset:
        return LeRobotSingleDataset(
            dataset_path=path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=cfg.video_backend,
        )

    if len(cfg.dataset_path) == 1:
        train_dataset = single_dataset(_resolve_dataset_path(cfg.dataset_path[0]))
    else:
        single = [single_dataset(_resolve_dataset_path(path)) for path in cfg.dataset_path]
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in single],
            mode="train",
            balance_dataset_weights=cfg.balance_dataset_weights,
            balance_trajectory_weights=cfg.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=cfg.base_model_path,
        tune_llm=cfg.tune_llm,
        tune_visual=cfg.tune_visual,
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if cfg.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            action_head_only=not cfg.lora_full_model,
        )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=cfg.max_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=8,
        report_to=cfg.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    if not cfg.save_final_model:

        def _skip_final_model_save(*, trainer, output_dir):
            print(
                f"Skipping duplicate final model save at {output_dir}; use checkpoint-* instead.",
                flush=True,
            )

        gr00t_runner.safe_save_model_for_hf_trainer = _skip_final_model_save

    gr00t_runner.TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=cfg.resume,
    ).train()


def _resolve_dataset_path(path_or_repo_id: str) -> str:
    path = Path(path_or_repo_id).expanduser()
    if path.exists():
        return str(path)

    cached_path = Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME) / "lerobot")) / path_or_repo_id
    if cached_path.exists():
        return str(cached_path)

    raise SystemExit(f"dataset path does not exist: {path_or_repo_id}")


def _validate(cfg: TrainConfig) -> None:
    if not cfg.dataset_path:
        raise SystemExit("--dataset-path is required")
    for path in cfg.dataset_path:
        _resolve_dataset_path(path)
    if cfg.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")
    if cfg.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if cfg.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if cfg.save_steps < 1:
        raise SystemExit("--save-steps must be >= 1")
    if cfg.video_backend == "decord" and importlib.util.find_spec("decord") is None:
        raise SystemExit(
            "decord is not installed; use --video-backend torchvision_av if torchvision VideoReader exists"
        )
    if cfg.video_backend == "torchvision_av":
        import torchvision

        if not hasattr(torchvision.io, "VideoReader"):
            raise SystemExit("torchvision_av requires torchvision.io.VideoReader; use --video-backend decord")


def _available_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def _run_torchrun(cfg: TrainConfig) -> int:
    if cfg.num_gpus > _available_gpus():
        raise SystemExit(f"requested {cfg.num_gpus} GPUs but only {_available_gpus()} available")

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["IS_TORCHRUN"] = "1"
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={cfg.num_gpus}",
        "--nnodes=1",
        str(Path(__file__).absolute()),
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, env=env).returncode


def main() -> None:
    _quiet_noisy_dependencies()
    cfg = tyro.cli(TrainConfig)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    _validate(cfg)

    logger.info("GR00T N1.5 SO-ARM finetune configuration")
    for k, v in vars(cfg).items():
        logger.info("%s = %s", k, v)

    if cfg.num_gpus == 1:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        _train(cfg)
        return

    if os.environ.get("IS_TORCHRUN", "0") == "1":
        _train(cfg)
        return

    sys.exit(_run_torchrun(cfg))


if __name__ == "__main__":
    main()
