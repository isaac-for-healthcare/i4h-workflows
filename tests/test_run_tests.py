# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_tests.py"
SPEC = importlib.util.spec_from_file_location("i4h_run_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_tests = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_tests
SPEC.loader.exec_module(run_tests)


def test_parser_accepts_gpu_suite() -> None:
    assert run_tests._parser().parse_args(["--suite", "gpu"]).suite == "gpu"


def test_gpu_smoke_workflows() -> None:
    assert [(smoke.workflow, smoke.mode) for smoke in run_tests.GPU_SMOKES] == [
        ("surgical_reach_psm", "rule-based"),
        ("surgical_lift_needle", "rule-based"),
        ("assemble_trocar", "policy"),
        ("ultrasound_liver_scan", "rule-based"),
        ("scissor_pick_and_place", "rule-based"),
        ("endoluminal_navigation", "demo"),
    ]


class _FakeRun:
    """run.sh that writes what the policy runner would, or nothing."""

    def __init__(self, calls: list[list[str]], success: str | None = "true", hang: bool = False) -> None:
        self.calls, self.success, self.hang = calls, success, hang

    def __call__(self, argv: list[str], **_kwargs):
        self.calls.append(argv)
        self.argv = argv
        return self

    def wait(self, timeout: float | None = None) -> int:
        if self.hang and timeout is not None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.success is not None:
            run_dir = Path(self.argv[self.argv.index("--run-dir") + 1])
            (run_dir / "outputs" / "run").mkdir(parents=True)
            (run_dir / "outputs" / "run" / "episode_results_rank0.jsonl").write_text(f'{{"success": {self.success}}}\n')
            Path(self.argv[self.argv.index("--record") + 1]).touch()
        return 0

    def terminate(self) -> None:
        self.calls.append(["terminate"])


def _smoke(workflow: str):
    return next(smoke for smoke in run_tests.GPU_SMOKES if smoke.workflow == workflow)


def test_a_workflow_smoke_records_one_headless_episode_through_run_sh(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(run_tests.subprocess, "Popen", _FakeRun(calls))
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert run_tests._run_workflow_smoke(_smoke("assemble_trocar"), timeout=30)
    command = calls[0]
    assert command[0].endswith("run.sh")
    assert command[1:4] == ["assemble_trocar", "--mode", "policy"]
    assert command[command.index("--episodes") + 1] == "1"
    assert "--headless" in command


def test_a_workflow_smoke_requires_an_episode_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_tests.subprocess, "Popen", _FakeRun([], success=None))
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert not run_tests._run_workflow_smoke(_smoke("surgical_lift_needle"), timeout=30)


def test_a_workflow_smoke_that_hangs_is_stopped_like_ctrl_c(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(run_tests.subprocess, "Popen", _FakeRun(calls, hang=True))
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert not run_tests._run_workflow_smoke(_smoke("surgical_reach_psm"), timeout=30)
    assert calls[-1] == ["terminate"]


def test_coverage_command_uses_parallel_coverage_data() -> None:
    command = run_tests.SUITES["light"][0].argv(coverage=True)

    assert f"coverage=={run_tests.COVERAGE_VERSION}" in command
    assert command[command.index("coverage") + 1 : command.index("coverage") + 3] == ["run", "--parallel-mode"]
    assert command[command.index("--project") + 1] == "common"
    assert command[-1] == "rl/tests/test_rsl_rl_arm_runtime.py"


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


@pytest.mark.parametrize("suite", ["ci", "rl"])
def test_rl_suite_executes_when_source_exists(tmp_path, monkeypatch, suite):
    (tmp_path / "third_party/RLinf-a4b6abe").mkdir(parents=True)
    projects = _record_suite_runs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--suite", suite])
    assert run_tests.main() == 0
    assert "rl" in projects


def test_ci_requires_rl_source(tmp_path, monkeypatch):
    _record_suite_runs(monkeypatch, tmp_path)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--suite", "rl"])
    assert run_tests.main() == 1


@pytest.mark.parametrize("suite", ["gpu", "dependency-pins"])
def test_dependency_pin_smoke_failure_is_not_skipped(monkeypatch, suite):
    calls = []
    monkeypatch.setattr(run_tests.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--suite", suite, "--timeout", "30"])
    monkeypatch.setattr(run_tests, "_run_workflow_smoke", lambda *_: True)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(run_tests, "_run_command", run)
    assert run_tests.main() == 1
    assert len(calls) == 1
    assert "scripts/validate_dependency_pins.py" in calls[0][0]
    assert calls[0][1]["timeout"] == 90
