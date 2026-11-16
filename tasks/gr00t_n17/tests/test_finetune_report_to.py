# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fine-tune reports the metrics destination the trainer actually gets.

Isaac-GR00T's ``launch_finetune.py`` takes only ``--use_wandb`` and otherwise
hands the trainer ``report_to="none"``. Printing ``tensorboard`` sent users
looking for event files that are never written (NVBug 7048541).
"""

from __future__ import annotations

from pathlib import Path

from i4h_tasks.gr00t_n17 import _finetune

TASK_ID = "gr00t_n17/catheter_navigation"
LAUNCH_SCRIPT = Path("/groot/gr00t/experiment/launch_finetune.py")


def test_by_default_only_the_console_gets_the_metrics(tmp_path: Path, capsys) -> None:
    from i4h_tasks.gr00t_n17.train import main

    assert main(["--task", TASK_ID, "--dataset", str(tmp_path), "--video-backend", "decord", "--dry-run"]) == 0
    assert "report_to = 'none'" in capsys.readouterr().out
    cfg = _finetune.TrainConfig(dataset_path=["d"], video_backend="decord")
    assert "--use_wandb" not in _finetune._build_cmd(cfg, LAUNCH_SCRIPT, "d")


def test_wandb_reaches_the_launcher() -> None:
    cfg = _finetune.TrainConfig(dataset_path=["d"], video_backend="decord", report_to="wandb")
    assert "--use_wandb" in _finetune._build_cmd(cfg, LAUNCH_SCRIPT, "d")
