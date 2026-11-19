# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 fine-tuning entry for SO-ARM scissor pick-and-place.

Wraps Isaac-GR00T's ``launch_finetune.py`` while retaining the released
SO-ARM checkpoint defaults.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import torch
import tyro

logger = logging.getLogger("i4h_tasks.gr00t_n17.train")

VideoBackend = Literal["torchcodec", "decord"]
VIDEO_BACKENDS: tuple[VideoBackend, ...] = ("torchcodec", "decord")
DECORD_UNSUPPORTED_CODECS = frozenset({"av1"})


@dataclass
class TrainConfig:
    """Configuration for GR00T N1.7 fine-tuning on a SO-ARM LeRobot dataset.

    Wraps Isaac-GR00T's ``launch_finetune.py`` with the released SO-ARM
    checkpoint defaults.
    """

    dataset_path: list[str]
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

    video_backend: VideoBackend | None = None
    """Decoder for the dataset's camera videos; unset picks the first of ``VIDEO_BACKENDS`` that is installed.

    GR00T defaults to torchcodec and raises rather than falling back when it is
    missing. No torchcodec wheel built for the pinned torch exists for Linux
    aarch64, where this project installs decord2 instead.
    """

    def __post_init__(self) -> None:
        if self.video_backend is None:
            self.video_backend = resolve_video_backend()


def resolve_video_backend() -> VideoBackend:
    """Return the first backend in ``VIDEO_BACKENDS`` that is installed.

    Only a backend that is not installed is skipped: the lock installs
    torchcodec on x86_64 and decord2 on aarch64, so a missing one is the
    platform's normal state. An installed backend that fails to import, such as
    a torchcodec whose native libraries no longer match torch or FFmpeg, is a
    broken install and stops the run instead of quietly switching decoder.
    """
    for name in VIDEO_BACKENDS:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                _broken_video_backend(name, error)
            logger.info("video backend %s is not installed; trying the next one", name)
            continue
        except Exception as error:  # noqa: BLE001 - torchcodec raises RuntimeError/OSError when its native libs fail to load
            _broken_video_backend(name, error)
        return name
    raise SystemExit(f"no video backend for the dataset's camera videos; install one of {', '.join(VIDEO_BACKENDS)}")


def _broken_video_backend(name: str, error: Exception) -> NoReturn:
    others = ", ".join(f"--video-backend {other}" for other in VIDEO_BACKENDS if other != name)
    raise SystemExit(
        f"video backend {name} is installed but failed to import ({type(error).__name__}: {error}); "
        f"repair it, or pass {others} if that backend is installed"
    ) from error


def check_video_codecs(dataset_path: str, backend: VideoBackend | None) -> None:
    """Refuse a dataset whose camera videos the chosen backend cannot decode.

    decord2 fails on the first AV1 packet, LeRobot's default codec, partway
    into training rather than at startup. Checks the first video in each
    directory, since a dataset is encoded with one codec per conversion.
    """
    if backend != "decord":
        return
    first_videos: dict[Path, Path] = {}
    for video in sorted(Path(dataset_path).glob("videos/**/*.mp4")):
        first_videos.setdefault(video.parent, video)
    if not first_videos:
        return
    import av

    for video in first_videos.values():
        with av.open(str(video)) as container:
            codec = container.streams.video[0].codec_context.codec.canonical_name
        if codec in DECORD_UNSUPPORTED_CODECS:
            raise SystemExit(
                f"{video} is encoded as {codec}, which the decord video backend cannot decode; "
                "convert the dataset again with the default --video-codec h264, re-encode its videos as H.264, "
                "or train where torchcodec is installed"
            )


def _groot_root() -> Path:
    # _finetune.py -> gr00t_n17 package -> tasks package -> component -> tasks -> repository
    return Path(__file__).resolve().parents[4] / "third_party" / "Isaac-GR00T-1.7"


def _modality_config_path() -> Path:
    # Same registration used at inference — launch_finetune.py loads this
    # file by path, fires ``register_modality_config(...)`` as a side effect.
    return Path(__file__).resolve().parent / "config.py"


def _build_cmd(cfg: TrainConfig, launch_script: Path, dataset_path: str) -> list[str]:
    # The shim's own options precede the launch script; launch_finetune.py has
    # no flag for them and tyro would reject them.
    shim_args = []
    if cfg.video_backend is not None:
        shim_args.append(f"--video-backend={cfg.video_backend}")
    cmd = [
        sys.executable,
        "-m",
        "i4h_tasks.gr00t_n17._launcher",
        *shim_args,
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
    logger.info("decoding camera videos with %s", cfg.video_backend)
    check_video_codecs(cfg.dataset_path[0], cfg.video_backend)

    cmd = _build_cmd(cfg, launch_script, cfg.dataset_path[0])
    logger.info("Running: %s", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=str(groot_root)).returncode)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    run(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
