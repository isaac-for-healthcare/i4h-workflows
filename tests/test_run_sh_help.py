# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Help-text regression tests for the shared workflow launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
RUN_SH = WORKFLOWS_ROOT / "run.sh"
WORKFLOW = "scissor_pick_and_place"

# Options run.sh does not parse itself but forwards to the Arena CLI through the
# catch-all branch. They work either way, so only the help text keeps them
# discoverable.
FORWARDED_SIMULATION_OPTIONS = (
    ("--presets", "newton"),
    ("--device", "cuda:0"),
)


def _help_text() -> str:
    result = subprocess.run([str(RUN_SH), "--help"], cwd=WORKFLOWS_ROOT, check=True, capture_output=True, text=True)
    return result.stdout


@pytest.mark.parametrize(("option", "_value"), FORWARDED_SIMULATION_OPTIONS)
def test_forwarded_simulation_options_are_listed_in_help(option: str, _value: str) -> None:
    assert option in _help_text()


@pytest.mark.parametrize(("option", "value"), FORWARDED_SIMULATION_OPTIONS)
def test_forwarded_simulation_options_are_accepted(tmp_path: Path, option: str, value: str) -> None:
    result = subprocess.run(
        [
            str(RUN_SH),
            WORKFLOW,
            "--dry-run",
            "--no-backend",
            "--run-dir",
            str(tmp_path / "run"),
            option,
            value,
        ],
        cwd=WORKFLOWS_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
