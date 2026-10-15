# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight shared Hydra composition and recording policy."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException


def compose_config(path: Path, overrides=()) -> DictConfig:
    try:
        with initialize_config_dir(config_dir=str(path.resolve().parent), version_base="1.3"):
            return compose(config_name=path.stem, overrides=list(overrides))
    except (HydraException, OmegaConfBaseException) as exc:
        detail = str(exc).split("\nSet the environment variable", 1)[0].strip()
        raise SystemExit(f"cannot compose RLinf config: {detail}") from exc


def configure_launch(
    cfg: DictConfig,
    *,
    model_path: Path,
    run_dir: Path,
    num_envs: int,
    max_epochs: int,
    only_eval: bool,
    video: bool,
    resume_dir: Path | None = None,
    rl_model_path: Path | None = None,
) -> bool:
    """Apply launch arguments before either preflight or workers register adapters."""
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(run_dir.resolve())
        cfg.runner.max_epochs = max_epochs
        cfg.runner.only_eval = only_eval
        cfg.env.train.total_num_envs = num_envs
        cfg.env.eval.total_num_envs = num_envs
        cfg.actor.model.model_path = str(model_path.resolve())
        cfg.rollout.model.model_path = str(model_path.resolve())
        if only_eval:
            # An embodied-eval worker reads rollout.model directly, so give
            # it the complete actor contract plus its rollout overrides.
            cfg.runner.task_type = "embodied_eval"
            cfg.rollout.model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
        if resume_dir is not None:
            cfg.runner.resume_dir = str(resume_dir.resolve())
        if rl_model_path is not None:
            cfg.rollout.model.rl_model_path = str(rl_model_path.resolve())
    return configure_video(cfg, only_eval=only_eval, video=video)


def write_runtime_config(cfg: DictConfig, run_dir: Path) -> Path:
    """Persist the effective config for adapters reading RLINF_CONFIG_FILE."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir.resolve() / "config.yaml"
    # RLinf registers custom resolvers on import; keep interpolations until
    # then while storing the literal checkpoint path needed by registration.
    OmegaConf.save(cfg, path, resolve=False)
    return path


def configure_video(cfg: DictConfig, *, only_eval: bool, video: bool) -> bool:
    """Only standalone evaluations require successful requested recordings."""
    requested = video or bool(OmegaConf.select(cfg, "env.eval.video_cfg.save_video", default=False))
    with open_dict(cfg):
        if video:
            cfg.env.eval.video_cfg.save_video = True
        cfg.env.train.video_cfg.raise_on_error = False
        cfg.env.eval.video_cfg.raise_on_error = only_eval and requested
    return only_eval and requested


def eval_video_dir(cfg: DictConfig, run_dir: Path) -> Path:
    """Where the launcher's config writes evaluation videos for this run."""
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(run_dir.resolve())
    return Path(cfg.env.eval.video_cfg.video_base_dir)
