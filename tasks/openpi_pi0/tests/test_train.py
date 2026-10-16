# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PI0 finetuning runs without a Weights & Biases account unless one is asked for (NVBug 6941887)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from i4h_tasks.openpi_pi0 import train


@pytest.mark.parametrize(
    ("flag", "environ", "expected"),
    [
        (None, {}, False),
        (None, {"WANDB_API_KEY": "   "}, False),
        (None, {"WANDB_API_KEY": "key"}, True),
        (True, {}, True),
        (False, {"WANDB_API_KEY": "key"}, False),
    ],
)
def test_wandb_is_opt_in(flag, environ, expected) -> None:
    assert train.wandb_enabled(flag, environ) is expected


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


def test_wandb_follows_the_flag_or_a_configured_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _finetune(monkeypatch, tmp_path / "flag", "--wandb").wandb_enabled is True

    monkeypatch.setenv("WANDB_API_KEY", "key")
    assert _finetune(monkeypatch, tmp_path / "key").wandb_enabled is True
    assert _finetune(monkeypatch, tmp_path / "opt-out", "--no-wandb").wandb_enabled is False


def test_dry_run_reports_the_wandb_decision(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    dataset = tmp_path / "local" / "demo"
    dataset.mkdir(parents=True)

    assert train.main(["--task", "openpi_pi0/ultrasound_liver_scan", "--dataset", str(dataset), "--dry-run"]) == 0

    assert "wandb = False" in capsys.readouterr().out


def test_openpi_starts_without_an_account_when_logging_is_off(monkeypatch, tmp_path) -> None:
    """The call that raised ``UsageError: No API key configured`` succeeds with logging off."""
    from openpi import train as openpi_train

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("NETRC", str(tmp_path / "no-netrc"))
    cfg = _finetune(monkeypatch, tmp_path)

    openpi_train.init_wandb(cfg, resuming=False, enabled=cfg.wandb_enabled)
