# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared GR00T N1.6 fine-tuning entry for G1 locomanip envs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import torch
import tyro

logger = logging.getLogger("policy.locomanip.train")


def _workflow_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _selected_env_id() -> str:
    return os.environ.get("AGENTIC_POLICY_ENV_ID", "locomanip")


def _default_base_model(env_id: str, fallback: str) -> str:
    """Resolve the env's pretrained checkpoint from config, falling back if absent."""
    import os

    os.environ.setdefault("WORKFLOW_ROOT", str(_workflow_root()))
    try:
        from common.config import get_policy_config

        repo = get_policy_config(env_id).model_repo
        if repo:
            return repo
    except Exception:
        pass
    return fallback


def env_train_default(env_id: str, field_name: str, fallback=None):
    os.environ.setdefault("WORKFLOW_ROOT", str(_workflow_root()))
    try:
        from common.config import get_environment_config

        policy_config = get_environment_config(env_id).get("policy") or {}
        train_config = policy_config.get("train") or {}
        return train_config.get(field_name, fallback)
    except Exception:
        return fallback


@dataclass
class TrainConfig:
    """Fine-tune GR00T N1.6 on a G1 locomanip dataset."""

    dataset_path: List[str]
    env_id: str = field(default_factory=_selected_env_id)
    output_dir: str | None = None

    base_model_path: str | None = None
    embodiment_tag: str = "NEW_EMBODIMENT"
    modality_config_path: str | None = None
    video_backend: str = "decord"

    batch_size: int = 32
    max_steps: int | None = None
    save_steps: int | None = None
    num_gpus: int = 1

    tune_llm: bool = False
    tune_visual: bool = True
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    dataloader_num_workers: int = 8
    report_to: Literal["wandb", "tensorboard"] = "tensorboard"

    color_jitter_params: List[str] = field(
        default_factory=lambda: ["brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08"]
    )


def _groot_root() -> Path:
    return _workflow_root() / "third_party" / "Isaac-GR00T-1.6"


def _apply_env_defaults(cfg: TrainConfig) -> TrainConfig:
    if cfg.output_dir is None:
        cfg.output_dir = env_train_default(cfg.env_id, "output_dir", f"/tmp/gr00t_{cfg.env_id}")
    if cfg.base_model_path is None:
        cfg.base_model_path = _default_base_model(cfg.env_id, "nvidia/GR00T-N1.6-3B")
    if cfg.max_steps is None:
        max_steps = env_train_default(cfg.env_id, "max_steps", 60000)
        cfg.max_steps = int(max_steps if max_steps is not None else 60000)
    if cfg.save_steps is None:
        save_steps = env_train_default(cfg.env_id, "save_steps", 10000)
        cfg.save_steps = int(save_steps if save_steps is not None else 10000)
    if cfg.modality_config_path is None:
        cfg.modality_config_path = env_train_default(cfg.env_id, "modality_config_path")
    return cfg


def _modality_config_path(cfg: TrainConfig) -> Path:
    if cfg.modality_config_path:
        path = Path(cfg.modality_config_path).expanduser()
        return path.resolve() if path.is_absolute() else (_workflow_root() / path).resolve()
    return Path(__file__).resolve().parents[1] / "config.py"


def _load_modality_config(modality_config_path: Path) -> None:
    import importlib.util

    if not modality_config_path.exists() or modality_config_path.suffix != ".py":
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")
    spec = importlib.util.spec_from_file_location(modality_config_path.stem, modality_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load modality config: {modality_config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Loaded modality config: %s", modality_config_path)


def _color_jitter_params(raw: list[str]) -> dict[str, float] | None:
    if not raw:
        return None
    if len(raw) % 2:
        raise ValueError("color_jitter_params must contain key/value pairs")
    return {raw[i]: float(raw[i + 1]) for i in range(0, len(raw), 2)}


def _build_gr00t_config(cfg: TrainConfig, dataset_path: str):
    from gr00t.configs.base_config import get_default_config

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": cfg.embodiment_tag.lower(),
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = cfg.tune_llm
    config.model.tune_visual = cfg.tune_visual
    config.model.tune_projector = cfg.tune_projector
    config.model.tune_diffusion_model = cfg.tune_diffusion_model
    config.model.color_jitter_params = _color_jitter_params(cfg.color_jitter_params)

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = cfg.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = cfg.batch_size
    config.training.dataloader_num_workers = cfg.dataloader_num_workers
    config.training.learning_rate = cfg.learning_rate
    config.training.gradient_accumulation_steps = 1
    config.training.output_dir = cfg.output_dir
    config.training.save_steps = cfg.save_steps
    config.training.num_gpus = cfg.num_gpus
    config.training.use_wandb = cfg.report_to == "wandb"
    config.training.max_steps = cfg.max_steps
    config.training.weight_decay = cfg.weight_decay
    config.training.warmup_ratio = cfg.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = 2**10
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = int(1e5)
    config.data.video_backend = cfg.video_backend
    return config


def _run_gr00t_training(config) -> None:
    import warnings

    import torch.distributed as dist
    import wandb
    from gr00t.experiment.experiment import setup_logging, warn_configs
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.experiment.utils import BestMetricCheckpointCallback, CheckpointFormatCallback
    from gr00t.model import MODEL_REGISTRY
    from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions
    from omegaconf import OmegaConf
    from transformers import TrainingArguments, set_seed

    warn_configs(config)
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    setup_logging()
    set_seed(config.data.seed)
    config.validate()

    output_dir = Path(config.training.output_dir)
    if config.training.experiment_name is not None:
        output_dir = output_dir / config.training.experiment_name
    experiment_name = config.training.experiment_name or output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)
    with open(output_dir / "wandb_config.json", "w") as f:
        json.dump({"project": config.training.wandb_project, "run_id": experiment_name}, f)

    if config.training.use_wandb and global_rank == 0:
        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config={**config.__dict__, "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown")},
            tags=[config.data.mode],
        )

    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    processor.save_pretrained(processor_dir)

    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None
    if config.training.batch_size is not None:
        warnings.warn("batch_size overrides global_batch_size", stacklevel=2)
        per_device_train_batch_size = config.training.batch_size
    else:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
    )
    trainer.add_callback(
        CheckpointFormatCallback(run_name=experiment_name, exp_cfg_dir=save_cfg_dir, processor_dir=processor_dir)
    )
    if config.training.save_best_eval_metric_name != "":
        trainer.add_callback(
            BestMetricCheckpointCallback(
                metric_name=config.training.save_best_eval_metric_name,
                greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                exp_cfg_dir=save_cfg_dir,
            )
        )

    if hasattr(train_dataset, "get_initial_actions"):
        initial_actions = train_dataset.get_initial_actions()
        if initial_actions:
            save_initial_actions(initial_actions, save_cfg_dir / INITIAL_ACTIONS_FILENAME)

    logger.info("Starting GR00T training")
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    logger.info("Model saved to %s", output_dir)

    if (
        config.training.assert_loss_less_than is not None
        and trainer.loss.item() > config.training.assert_loss_less_than
    ):
        raise AssertionError(f"Loss too high: {trainer.loss.item()} vs {config.training.assert_loss_less_than})")

    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if eval_dataset is not None and hasattr(eval_dataset, "close"):
        eval_dataset.close()


def run(cfg: TrainConfig) -> None:
    cfg = _apply_env_defaults(cfg)
    if not cfg.dataset_path:
        raise SystemExit("--dataset-path is required")
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if cfg.num_gpus > available_gpus:
        raise SystemExit(f"requested {cfg.num_gpus} GPUs but only {available_gpus} available")

    groot_root = _groot_root()
    if not (groot_root / "gr00t").exists():
        raise SystemExit(f"Isaac-GR00T-1.6 source not found at {groot_root}. Run ./setup.sh first.")

    logger.info("GR00T N1.6 locomanip fine-tune configuration:")
    for k, v in vars(cfg).items():
        logger.info("  %s = %s", k, v)

    # Pin CUDA visibility before GR00T initializes torch distributed/DataParallel.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(cfg.num_gpus))

    _load_modality_config(_modality_config_path(cfg))
    config = _build_gr00t_config(cfg, cfg.dataset_path[0])
    logger.info("Running GR00T training directly (CUDA_VISIBLE_DEVICES=%s)", os.environ.get("CUDA_VISIBLE_DEVICES"))
    _run_gr00t_training(config)


def run_cli(config_cls: type[TrainConfig] = TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    run(tyro.cli(config_cls))


def make_main(config_cls: type[TrainConfig]):
    def _main() -> None:
        run_cli(config_cls)

    return _main


def main() -> None:
    run_cli(TrainConfig)


if __name__ == "__main__":
    main()
