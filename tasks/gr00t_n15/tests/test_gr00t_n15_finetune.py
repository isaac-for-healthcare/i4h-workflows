# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n15._finetune import _pin_single_gpu


def test_pin_single_gpu_defaults_to_first_device(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _pin_single_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_pin_single_gpu_preserves_explicit_selection(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    _pin_single_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def _train_argv(*extra: str) -> list[str]:
    return ["--task", "gr00t_n15/scissor_pick_and_place", "--dataset", "lerobot/svla_so101_pickplace", *extra]


def test_a_dry_run_rejects_a_step_count_the_real_run_rejects(monkeypatch) -> None:
    """NVBug 7035864: --dry-run accepted --max-steps 0 and exited 0."""
    from i4h_common import training
    from i4h_tasks.gr00t_n15 import train

    monkeypatch.setattr(training, "resolve_dataset", lambda _d: pytest.fail("a dry run must not resolve datasets"))

    with pytest.raises(SystemExit, match="--max-steps must be >= 1"):
        train.main(_train_argv("--max-steps", "0", "--dry-run"))


def test_a_dry_run_does_not_download_a_hugging_face_dataset(monkeypatch, capsys) -> None:
    from i4h_common import training
    from i4h_tasks.gr00t_n15 import _finetune, train

    # The light test environment has no video decoder; the trainer's own does.
    monkeypatch.setattr(_finetune.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(training, "resolve_dataset", lambda _d: pytest.fail("a dry run must not resolve datasets"))

    assert train.main(_train_argv("--max-steps", "1", "--dry-run")) == 0
    assert "dataset_path = ['lerobot/svla_so101_pickplace']" in capsys.readouterr().out
