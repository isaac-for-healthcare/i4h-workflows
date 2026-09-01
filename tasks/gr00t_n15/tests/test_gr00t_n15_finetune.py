# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

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
