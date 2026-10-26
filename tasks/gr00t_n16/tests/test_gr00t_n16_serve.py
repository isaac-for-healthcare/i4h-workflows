# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``serve`` sets up before GR00T's server takes over. Runs without ``gr00t``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n16.serve import TASKS, resolve_model_path  # noqa: E402


def test_a_missing_local_checkpoint_is_not_looked_up_on_the_hub(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        resolve_model_path(str(tmp_path / "checkpoint-1000"), "nvidia/repo", "main")


def test_an_existing_checkpoint_is_served_from_disk(tmp_path) -> None:
    assert resolve_model_path(str(tmp_path), "nvidia/repo", "main") == str(tmp_path)


def test_a_task_without_a_published_checkpoint_needs_one() -> None:
    with pytest.raises(ValueError, match="--model-path"):
        resolve_model_path(None, None, None)


def test_every_served_task_has_a_manifest() -> None:
    manifests = Path(__file__).resolve().parents[1] / "i4h_tasks" / "gr00t_n16" / "manifest"
    assert {path.stem for path in manifests.glob("*.yaml")} == set(TASKS)
