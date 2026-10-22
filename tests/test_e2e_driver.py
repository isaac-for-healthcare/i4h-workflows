# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The e2e driver runs the workflow's Arena environment and grades with its scene's settings (NVBug 6941887)."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from i4h_common.manifest import load_scene_spec

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
E2E = WORKFLOWS_ROOT / "scripts" / "e2e" / "run.sh"
PHANTOM = WORKFLOWS_ROOT / "arena" / "i4h_arena" / "scenes" / "manifest" / "panda_phantom.yaml"


def _run(tmp_path: Path, env: str, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(E2E), "--env", env, "--dry-run", "--run-dir", str(tmp_path / "run"), *options],
        cwd=WORKFLOWS_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _dry_run(tmp_path: Path, env: str, *options: str) -> list[list[str]]:
    result = _run(tmp_path, env, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    # Background servers print as "<command> > <log> &".
    lines = [line[2:].removesuffix(" &") for line in result.stdout.splitlines() if line.startswith("+ ")]
    return [shlex.split(line) for line in lines]


def _command(commands: list[list[str]], marker: str, occurrence: int = 0) -> list[str]:
    return [command for command in commands if marker in command][occurrence]


def _option(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def test_the_liver_scan_is_graded_by_its_scenes_visible_criterion_over_both_cameras(tmp_path) -> None:
    declared = load_scene_spec(PHANTOM).annotate

    annotate = _command(_dry_run(tmp_path, "ultrasound_liver_scan"), "i4h-annotator")

    assert declared is not None
    assert _option(annotate, "--task") == declared.criterion
    assert _option(annotate, "--task") != "Perform a liver ultrasound."
    assert _option(annotate, "--camera") == "room,wrist"
    assert _option(annotate, "--judge") == "trajectory"


def test_training_keeps_the_policy_prompt(tmp_path) -> None:
    convert = _command(_dry_run(tmp_path, "ultrasound_liver_scan"), "convert")

    assert _option(convert, "--task") == "Perform a liver ultrasound."
    assert _option(convert, "--robot") == "panda"


def test_recording_runs_the_arena_environment_against_the_policy_server(tmp_path) -> None:
    commands = _dry_run(tmp_path, "ultrasound_liver_scan")
    server = _command(commands, "i4h_tasks.openpi_pi0.serve")
    record = _command(commands, "isaaclab_arena.evaluation.policy_runner")

    assert commands.index(server) < commands.index(record)
    assert _option(record, "--policy_type") == "i4h_arena.policies.pi0.LiverScanPi0Policy"
    assert _option(record, "--remote_port") == _option(server, "--port")
    assert _option(record, "--external_environment_class_path") == "i4h_arena.environments:PandaPhantom"
    assert _option(record, "--recorder_dataset_export_dir_path") == str(tmp_path / "run/data")
    assert _option(record, "--recorder_dataset_filename") == "recording"


def test_validation_serves_the_finetuned_checkpoint(tmp_path) -> None:
    commands = _dry_run(tmp_path, "ultrasound_liver_scan")
    server = _command(commands, "i4h_tasks.openpi_pi0.serve", occurrence=1)
    validate = _command(commands, "isaaclab_arena.evaluation.policy_runner", occurrence=1)

    assert _option(server, "--checkpoint") == "<latest-checkpoint>"
    assert "--viz" not in validate
    assert _option(validate, "--output_base_dir") == str(tmp_path / "run/validate/outputs")


def test_replay_uses_isaac_lab_with_the_same_environment(tmp_path) -> None:
    replay = _command(_dry_run(tmp_path, "ultrasound_liver_scan"), "--dataset_file")

    assert "third_party/IsaacLab-e3f7d9f/scripts/tools/replay_demos.py" in replay
    assert _option(replay, "--external_environment_class_path") == "i4h_arena.environments:PandaPhantom"
    assert _option(replay, "--task") == "panda_phantom"
    assert "--enable_cameras" not in replay


@pytest.mark.parametrize("marker", ["isaaclab_arena.evaluation.policy_runner", "--dataset_file"])
def test_headless_closes_the_viewport_for_recording_and_replay(tmp_path, marker) -> None:
    assert "--viz" in _command(_dry_run(tmp_path, "ultrasound_liver_scan"), marker)
    assert "--viz" not in _command(_dry_run(tmp_path, "ultrasound_liver_scan", "--headless"), marker)


def test_a_workflow_without_an_e2e_pipeline_is_refused(tmp_path) -> None:
    result = _run(tmp_path, "endoluminal_navigation")

    assert result.returncode == 2
    assert "has no e2e pipeline" in result.stderr


def test_the_scissors_validate_their_finetuned_checkpoint_with_the_n15_server(tmp_path) -> None:
    commands = _dry_run(tmp_path, "scissor_pick_and_place")
    server = _command(commands, "i4h_tasks.gr00t_n15.serve", occurrence=1)
    record = _command(commands, "isaaclab_arena.evaluation.policy_runner")
    annotate = _command(commands, "i4h-annotator")

    assert _option(server, "--model-path") == "<latest-checkpoint>"
    assert _option(record, "--embodiment") == "so101_scissors"
    assert _option(record, "--external_environment_class_path") == "i4h_arena.environments:SoarmScissors"
    # A scene without annotate settings is graded against the policy prompt.
    assert annotate[annotate.index("i4h-annotator") + 1 : annotate.index("offline")] == [
        "--task",
        "Grip the scissors and put it into the tray",
    ]
