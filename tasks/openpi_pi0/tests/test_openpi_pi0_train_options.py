# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PI0 fine-tuning logs to Weights & Biases only when asked to (NVBug 6941887).

These tests need neither openpi nor JAX, so the light CI suite runs them from
the ``workflows`` environment; ``test_train.py`` covers the openpi hand-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.openpi_pi0 import train  # noqa: E402

TASK = "openpi_pi0/ultrasound_liver_scan"


@pytest.mark.parametrize(
    ("flag", "manifest", "environ", "expected"),
    [
        # The environment never turns logging on: not a key, a login, or a mode.
        (None, {}, {}, False),
        (None, {}, {"WANDB_API_KEY": "key"}, False),
        (None, {}, {"WANDB_MODE": "offline"}, False),
        (None, {"wandb": False}, {"WANDB_API_KEY": "key"}, False),
        # The manifest, then the flag, do.
        (None, {"wandb": True}, {}, True),
        (True, {}, {}, True),
        (True, {"wandb": False}, {}, True),
        (False, {"wandb": True}, {"WANDB_API_KEY": "key"}, False),
        # A requested run honours WANDB_MODE.
        (True, {}, {"WANDB_MODE": "offline"}, True),
        (True, {}, {"WANDB_MODE": "disabled"}, False),
        (None, {"wandb": True}, {"WANDB_MODE": "Disabled "}, False),
    ],
)
def test_wandb_is_an_explicit_opt_in(flag, manifest, environ, expected) -> None:
    enabled, _reason = train.wandb_enabled(flag, manifest, environ)

    assert enabled is expected


def test_the_decision_names_what_made_it() -> None:
    assert train.wandb_enabled(None, {}, {})[1] == "train.wandb: false in the task manifest"
    assert train.wandb_enabled(True, {}, {"WANDB_MODE": "offline"})[1] == "--wandb, WANDB_MODE=offline"
    assert train.wandb_enabled(True, {}, {"WANDB_MODE": "disabled"})[1] == "WANDB_MODE=disabled"


def test_a_non_boolean_manifest_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="train.wandb must be true or false"):
        train.wandb_enabled(None, {"wandb": "yes"}, {})


def _checkpoints(root: Path, *steps: int, wandb_id: bool = False) -> Path:
    directory = root / "robotic_ultrasound" / train.EXP_NAME
    directory.mkdir(parents=True)
    for step in steps:
        (directory / str(step)).mkdir()
    (directory / f"{max(steps, default=0)}.orbax-checkpoint-tmp-1").mkdir()
    if wandb_id:
        (directory / "wandb_id.txt").write_text("abc123")
    return directory


def test_a_resume_of_a_run_trained_without_wandb_is_recognised(tmp_path) -> None:
    """openpi reads wandb_id.txt on resume, and writes it only when W&B was on."""
    assert train.resumes_without_wandb_run(_checkpoints(tmp_path / "off", 0, 1000), overwrite=False)
    assert not train.resumes_without_wandb_run(_checkpoints(tmp_path / "on", 1000, wandb_id=True), overwrite=False)
    # openpi does not resume these: nothing, or only step 0, was saved; or --overwrite wipes it.
    assert not train.resumes_without_wandb_run(_checkpoints(tmp_path / "empty"), overwrite=False)
    assert not train.resumes_without_wandb_run(_checkpoints(tmp_path / "zero", 0), overwrite=False)
    assert not train.resumes_without_wandb_run(_checkpoints(tmp_path / "wiped", 1000), overwrite=True)
    assert not train.resumes_without_wandb_run(tmp_path / "missing", overwrite=False)


def _dry_run(tmp_path: Path, capsys, *argv: str) -> str:
    dataset = tmp_path / "lerobot" / "local" / "demo"
    dataset.mkdir(parents=True, exist_ok=True)
    code = train.main(
        ["--task", TASK, "--dataset", str(dataset), "--output-dir", str(tmp_path / "out"), "--dry-run", *argv]
    )
    assert code == 0
    return capsys.readouterr().out


def test_dry_run_reports_the_wandb_decision(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    assert "wandb = False" in _dry_run(tmp_path, capsys)
    assert "wandb = True" in _dry_run(tmp_path, capsys, "--wandb")


def test_wandb_stays_off_when_resuming_a_run_trained_without_it(monkeypatch, tmp_path, capsys, caplog) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    _checkpoints(tmp_path / "out", 0, 100)

    with caplog.at_level("WARNING"):
        out = _dry_run(tmp_path, capsys, "--wandb")

    assert "wandb = False" in out
    assert "resuming a run trained without W&B" in out
    assert "no W&B run to resume" in caplog.text
