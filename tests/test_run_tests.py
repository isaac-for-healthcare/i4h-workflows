# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_tests.py"
SPEC = importlib.util.spec_from_file_location("i4h_run_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_tests = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_tests
SPEC.loader.exec_module(run_tests)


def test_parser_accepts_gpu_suite() -> None:
    assert run_tests._parser().parse_args(["--suite", "gpu"]).suite == "gpu"


def test_gpu_smoke_workflows() -> None:
    assert [(smoke.workflow, smoke.mode_args) for smoke in run_tests.GPU_SMOKES] == [
        ("surgical_reach_psm", ("--rule-based",)),
        ("assemble_trocar", ("--policy",)),
        ("ultrasound_liver_scan", ("--rule-based",)),
        ("scissor_pick_and_place", ("--rule-based",)),
        ("endoluminal_navigation", ("--mode", "demo")),
    ]


def test_coverage_command_uses_parallel_coverage_data() -> None:
    command = run_tests.SUITES["light"][0].argv(coverage=True)

    assert f"coverage=={run_tests.COVERAGE_VERSION}" in command
    assert command[command.index("coverage") + 1 : command.index("coverage") + 3] == ["run", "--parallel-mode"]
    assert command[-2:] == ["tasks/rsl_rl/tests", "tasks/teleop/tests"]


def _patient_twin_command() -> run_tests.TestCommand:
    return next(command for command in run_tests.SUITES["tools"] if command.name == "patient twin")


def _record_suite_runs(monkeypatch, root: Path) -> list[str]:
    projects: list[str] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        projects.append(argv[argv.index("--project") + 1])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.setattr(run_tests.shutil, "which", lambda _name: "uv")
    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--suite", "tools"])
    return projects


def test_a_suite_is_skipped_without_its_component_checkout(tmp_path, monkeypatch, capsys) -> None:
    projects = _record_suite_runs(monkeypatch, tmp_path)

    assert run_tests.main() == 0
    assert "tools/patient_twin" not in projects
    assert "tools/mimic" in projects
    assert "skipped, no component checkout: patient twin" in capsys.readouterr().out


def test_a_suite_runs_once_its_component_checkout_exists(tmp_path, monkeypatch) -> None:
    checkout = _patient_twin_command().requires_checkout
    assert checkout is not None
    (tmp_path / checkout).mkdir(parents=True)
    projects = _record_suite_runs(monkeypatch, tmp_path)

    assert run_tests.main() == 0
    assert "tools/patient_twin" in projects


def test_gpu_smoke_runs_rollout_and_inspects_recording(tmp_path, monkeypatch) -> None:
    recording = tmp_path / run_tests.GPU_RECORDING
    recording.touch()
    rollout_output = f"==> run dir {tmp_path}\n1/1 episodes succeeded (1 attempts)\n"
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        output = rollout_output if argv[0] == "./run.sh" else None
        return subprocess.CompletedProcess(argv, 0, stdout=output)

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)

    smoke = next(smoke for smoke in run_tests.GPU_SMOKES if smoke.workflow == "ultrasound_liver_scan")
    assert run_tests._run_gpu_smoke(smoke, timeout=30)
    assert calls[0][:3] == ["./run.sh", "ultrasound_liver_scan", "--rule-based"]
    assert "--headless" in calls[0]
    assert "--record-failures" in calls[0]
    assert calls[0][calls[0].index("--attempts") + 1] == "1"
    assert calls[1][-3:] == ["inspect", str(recording), "--segments"]


def test_gpu_smoke_accepts_unsuccessful_episode(tmp_path, monkeypatch) -> None:
    recording = tmp_path / run_tests.GPU_RECORDING
    recording.touch()
    rollout_output = f"==> run dir {tmp_path}\n0/1 episodes succeeded (1 attempts)\n"
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        returncode = 1 if argv[0] == "./run.sh" else 0
        output = rollout_output if argv[0] == "./run.sh" else None
        return subprocess.CompletedProcess(argv, returncode, stdout=output)

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)

    assert run_tests._run_gpu_smoke(run_tests.GPU_SMOKES[1], timeout=30)
    assert calls[1][-3:] == ["inspect", str(recording), "--segments"]


def test_gpu_smoke_requires_episode_summary(tmp_path, monkeypatch) -> None:
    (tmp_path / run_tests.GPU_RECORDING).touch()

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout=f"==> run dir {tmp_path}\n")

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)

    assert not run_tests._run_gpu_smoke(run_tests.GPU_SMOKES[2], timeout=30)
