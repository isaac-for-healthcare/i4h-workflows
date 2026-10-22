# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The e2e driver runs the workflow through run.sh and grades with its scene's settings (NVBug 6941887)."""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from i4h_common.manifest import load_scene_spec

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
E2E = WORKFLOWS_ROOT / "scripts" / "e2e" / "run.sh"
PHANTOM = WORKFLOWS_ROOT / "arena" / "i4h_arena" / "scenes" / "manifest" / "panda_phantom.yaml"
SPEC = importlib.util.spec_from_file_location("i4h_run_workflow", WORKFLOWS_ROOT / "scripts" / "run_workflow.py")
run_workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_workflow
SPEC.loader.exec_module(run_workflow)


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
    lines = [line[2:] for line in result.stdout.splitlines() if line.startswith("+ ")]
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


def _run_sh(commands: list[list[str]], option: str, occurrence: int = 0) -> list[str]:
    """The ``occurrence``-th ``./run.sh`` stage that passes ``option``, without ``./run.sh``."""
    return [command[1:] for command in commands if command[0] == "./run.sh" and option in command][occurrence]


def test_recording_runs_the_policy_mode_against_its_server(tmp_path) -> None:
    record = _run_sh(_dry_run(tmp_path, "ultrasound_liver_scan"), "--policy")
    built = run_workflow.plan(record)

    assert record[0] == "ultrasound_liver_scan"
    assert _option(record, "--record") == str(tmp_path / "run/data/recording.hdf5")
    assert "i4h_tasks.openpi_pi0.serve" in built.server
    assert _option(built.command, "--policy_type") == "i4h_arena.policies.pi0.LiverScanPi0Policy"
    assert _option(built.command, "--remote_port") == _option(built.server, "--port")
    assert _option(built.command, "--external_environment_class_path") == "i4h_arena.environments:PandaPhantom"
    assert _option(built.command, "--recorder_dataset_export_dir_path") == str(tmp_path / "run/data")
    assert _option(built.command, "--recorder_dataset_filename") == "recording"


def test_validation_serves_the_finetuned_checkpoint_headless(tmp_path) -> None:
    validate = _run_sh(_dry_run(tmp_path, "ultrasound_liver_scan"), "--policy", occurrence=1)
    built = run_workflow.plan(validate)

    assert _option(validate, "--checkpoint") == "<latest-checkpoint>"
    assert built.server[-2] == "--checkpoint" and built.server[-1].endswith("<latest-checkpoint>")
    assert "--viz" not in built.command
    assert _option(built.command, "--output_base_dir") == str(tmp_path / "run/validate/outputs")


def test_replay_uses_isaac_lab_with_the_same_environment(tmp_path) -> None:
    replay = _run_sh(_dry_run(tmp_path, "ultrasound_liver_scan"), "--replay")
    command = run_workflow.plan(replay).command

    assert _option(replay, "--replay") == str(tmp_path / "run/data/filtered.hdf5")
    assert command[1].endswith("scripts/tools/replay_demos.py")
    assert _option(command, "--external_environment_class_path") == "i4h_arena.environments:PandaPhantom"
    assert _option(command, "--task") == "panda_phantom"
    assert _option(command, "--select_episodes") == "0"
    assert "--enable_cameras" not in command


@pytest.mark.parametrize("option", ["--policy", "--replay"])
def test_headless_closes_the_viewport_for_recording_and_replay(tmp_path, option) -> None:
    assert "--headless" not in _run_sh(_dry_run(tmp_path, "ultrasound_liver_scan"), option)
    assert "--headless" in _run_sh(_dry_run(tmp_path, "ultrasound_liver_scan", "--headless"), option)


def test_a_workflow_without_an_e2e_pipeline_is_refused(tmp_path) -> None:
    result = _run(tmp_path, "endoluminal_navigation")

    assert result.returncode == 2
    assert "has no e2e pipeline" in result.stderr


def test_the_scissors_validate_their_finetuned_checkpoint_with_the_n15_server(tmp_path) -> None:
    commands = _dry_run(tmp_path, "scissor_pick_and_place")
    record = run_workflow.plan(_run_sh(commands, "--policy"))
    validate = run_workflow.plan(_run_sh(commands, "--policy", occurrence=1))
    annotate = _command(commands, "i4h-annotator")

    assert "i4h_tasks.gr00t_n15.serve" in validate.server
    assert validate.server[-2] == "--model-path" and validate.server[-1].endswith("<latest-checkpoint>")
    assert _option(record.command, "--embodiment") == "so101_scissors"
    assert _option(record.command, "--external_environment_class_path") == "i4h_arena.environments:SoarmScissors"
    # A scene without annotate settings is graded against the policy prompt.
    assert annotate[annotate.index("i4h-annotator") + 1 : annotate.index("offline")] == [
        "--task",
        "Grip the scissors and put it into the tray",
    ]
