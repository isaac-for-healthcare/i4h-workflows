# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n16 import _finetune
from i4h_tasks.gr00t_n16.train import main


def test_run_pins_requested_gpus_before_cuda_query(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def available_gpus() -> int:
        assert _finetune.os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        return 1

    monkeypatch.setattr(_finetune, "_available_gpus", available_gpus)
    monkeypatch.setattr(_finetune, "_groot_root", lambda: _finetune.Path("/missing"))

    with pytest.raises(SystemExit, match="source not found"):
        _finetune.run(_finetune.TrainConfig(dataset_path=["dataset"], num_gpus=1))


def test_run_preserves_explicit_gpu_selection(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(_finetune, "_available_gpus", lambda: 1)
    monkeypatch.setattr(_finetune, "_groot_root", lambda: _finetune.Path("/missing"))

    with pytest.raises(SystemExit, match="source not found"):
        _finetune.run(_finetune.TrainConfig(dataset_path=["dataset"], num_gpus=1))

    assert _finetune.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_train_uses_task_modality_config(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("i4h_common.training.resolve_dataset", lambda path: path)
    monkeypatch.setattr(
        "i4h_common.training.require_trainable",
        lambda _task_id: SimpleNamespace(train={"modality_config_path": "custom_headcam.py"}),
    )

    assert main(["--task", "gr00t_n16/test_task", "--dataset", str(tmp_path), "--dry-run"]) == 0

    assert "custom_headcam.py" in capsys.readouterr().out
