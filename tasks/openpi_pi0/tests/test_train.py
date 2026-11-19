# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PI0 finetuning runs without a Weights & Biases account unless one is asked for (NVBug 6941887).

These tests drive openpi itself; ``test_openpi_pi0_train_options.py`` holds the
ones the light CI suite can run without it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from i4h_tasks.openpi_pi0 import train


def _finetune(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> object:
    """Run the shim up to openpi's training loop and return the config it would train with."""
    from openpi import train as openpi_train

    from i4h_tasks.openpi_pi0.ultrasound import utils

    captured: list[object] = []
    monkeypatch.setattr(openpi_train, "main", captured.append)
    monkeypatch.setattr(utils, "compute_normalization_stats", lambda _cfg: None)
    monkeypatch.setattr(utils.LeRobotDataConfig, "create", lambda self, *_: SimpleNamespace(repo_id="local/demo"))
    dataset = tmp_path / "lerobot" / "local" / "demo"
    dataset.mkdir(parents=True)

    code = train.main(
        [
            "--task",
            "openpi_pi0/ultrasound_liver_scan",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--max-steps",
            "2",
            *argv,
        ]
    )

    assert code == 0
    (cfg,) = captured
    return cfg


def test_a_host_without_wandb_trains_with_logging_off(monkeypatch, tmp_path) -> None:
    """openpi's TrainConfig defaults to wandb_enabled=True; the shim must not inherit that."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    cfg = _finetune(monkeypatch, tmp_path)

    assert cfg.wandb_enabled is False
    assert cfg.num_train_steps == 2


def test_wandb_follows_the_flag_not_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "key")

    assert _finetune(monkeypatch, tmp_path / "key").wandb_enabled is False
    assert _finetune(monkeypatch, tmp_path / "flag", "--wandb").wandb_enabled is True


def test_the_shim_predicts_openpis_checkpoint_directory(monkeypatch, tmp_path) -> None:
    """The resume check looks where openpi will resume from."""
    cfg = _finetune(monkeypatch, tmp_path)

    assert cfg.checkpoint_dir == (tmp_path / "checkpoints" / "robotic_ultrasound" / train.EXP_NAME).resolve()


def test_a_resume_with_wandb_of_a_run_trained_without_it_does_not_crash(monkeypatch, tmp_path) -> None:
    """openpi's init_wandb read the missing wandb_id.txt and raised FileNotFoundError."""
    from openpi import train as openpi_train

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    checkpoints = tmp_path / "checkpoints" / "robotic_ultrasound" / train.EXP_NAME
    (checkpoints / "100").mkdir(parents=True)

    cfg = _finetune(monkeypatch, tmp_path, "--wandb")

    assert cfg.wandb_enabled is False
    openpi_train.init_wandb(cfg, resuming=True, enabled=cfg.wandb_enabled)


def test_openpi_starts_without_an_account_when_logging_is_off(monkeypatch, tmp_path) -> None:
    """The call that raised ``UsageError: No API key configured`` succeeds with logging off."""
    from openpi import train as openpi_train

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("NETRC", str(tmp_path / "no-netrc"))
    cfg = _finetune(monkeypatch, tmp_path)

    openpi_train.init_wandb(cfg, resuming=False, enabled=cfg.wandb_enabled)
