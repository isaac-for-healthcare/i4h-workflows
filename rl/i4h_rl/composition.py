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


def configure_video(cfg: DictConfig, *, only_eval: bool, video: bool) -> bool:
    """Only standalone evaluations require successful requested recordings."""
    requested = video or bool(OmegaConf.select(cfg, "env.eval.video_cfg.save_video", default=False))
    with open_dict(cfg):
        if video:
            cfg.env.eval.video_cfg.save_video = True
        cfg.env.train.video_cfg.raise_on_error = False
        cfg.env.eval.video_cfg.raise_on_error = only_eval and requested
    return only_eval and requested
