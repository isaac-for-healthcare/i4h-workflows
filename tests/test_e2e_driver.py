# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The e2e driver grades recordings with the workflow's annotate settings (NVBug 6941887)."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from i4h_engine.loader import load_workflow_module

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
E2E = WORKFLOWS_ROOT / "scripts" / "e2e" / "run.sh"


def _dry_run(tmp_path: Path, env: str, *options: str) -> list[list[str]]:
    result = subprocess.run(
        [str(E2E), "--env", env, "--dry-run", "--run-dir", str(tmp_path / "run"), *options],
        cwd=WORKFLOWS_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


def _command(commands: list[list[str]], marker: str) -> list[str]:
    return next(command for command in commands if marker in command)


def _option(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def test_the_liver_scan_is_graded_by_its_visible_criterion_over_both_cameras(tmp_path) -> None:
    declared = load_workflow_module("ultrasound_liver_scan").workflow.annotate

    annotate = _command(_dry_run(tmp_path, "ultrasound_liver_scan"), "i4h-annotator")

    assert _option(annotate, "--task") == declared.criterion
    assert _option(annotate, "--task") != "Perform a liver ultrasound."
    assert _option(annotate, "--camera") == "room,wrist"
    assert _option(annotate, "--judge") == "trajectory"


def test_training_keeps_the_policy_prompt(tmp_path) -> None:
    convert = _command(_dry_run(tmp_path, "ultrasound_liver_scan"), "convert")

    assert _option(convert, "--task") == "Perform a liver ultrasound."


def test_a_workflow_without_annotate_settings_is_graded_as_before(tmp_path) -> None:
    annotate = _command(_dry_run(tmp_path, "scissor_pick_and_place"), "i4h-annotator")

    assert annotate[annotate.index("i4h-annotator") + 1 :] == [
        "--task",
        "Grip the scissors and put it into the tray",
        "offline",
        str(tmp_path / "run/data/expanded.hdf5"),
        "--write",
        "--filter",
        str(tmp_path / "run/data/filtered.hdf5"),
    ]


@pytest.mark.parametrize("stage", ["--record", "--replay"])
def test_headless_reaches_recording_and_replay(tmp_path, stage) -> None:
    commands = _dry_run(tmp_path, "scissor_pick_and_place", "--headless")

    assert "--headless" in _command(commands, stage)


def test_recording_opens_the_viewport_by_default(tmp_path) -> None:
    commands = _dry_run(tmp_path, "scissor_pick_and_place")

    assert "--headless" not in _command(commands, "--record")
