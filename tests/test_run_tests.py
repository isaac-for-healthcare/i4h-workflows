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


def test_gpu_smoke_environments() -> None:
    assert [(smoke.environment, smoke.policy_type) for smoke in run_tests.GPU_SMOKES] == [
        ("psm_reach", "i4h_arena.policies.reach.CartesianReachPolicy"),
        ("psm_needle", "i4h_arena.policies.lift.CartesianLiftPolicy"),
        ("g1_trocar", "i4h_arena.policies.gr00t_n15.Gr00tN15RemotePolicy"),
        ("panda_phantom", "i4h_arena.policies.scan.ProbeSweepPolicy"),
        ("soarm_scissors", "i4h_arena.policies.scissors.ScissorPickPlacePolicy"),
        ("endoluminal_navigation", "i4h_arena.policies.catheter.CatheterSweepPolicy"),
    ]


def _smoke(environment: str):
    return next(smoke for smoke in run_tests.GPU_SMOKES if smoke.environment == environment)


def test_policy_runner_smoke_records_one_episode(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        output = Path(argv[argv.index("--recorder_dataset_export_dir_path") + 1])
        (output / "outputs" / "run").mkdir(parents=True)
        (output / "outputs" / "run" / "episode_results_rank0.jsonl").write_text('{"success": true}\n')
        (output / run_tests.GPU_RECORDING).touch()
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert run_tests._run_policy_runner_smoke(_smoke("psm_reach"), timeout=30)
    command = calls[0]
    assert command[command.index("-m") + 1] == "isaaclab_arena.evaluation.policy_runner"
    assert command[command.index("--external_environment_class_path") + 1] == "i4h_arena.environments:PsmReach"
    assert command[command.index("--num_episodes") + 1] == "1"


def test_policy_runner_smoke_serves_the_policy_during_the_rollout(tmp_path, monkeypatch) -> None:
    events: list[str] = []

    class FakeServer:
        def __init__(self, argv, **_kwargs) -> None:
            events.append("server started: " + " ".join(argv[2:4]))

        def terminate(self) -> None:
            events.append("server stopped")

        def wait(self, timeout: float) -> int:
            return 0

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        events.append("rollout")
        output = Path(argv[argv.index("--recorder_dataset_export_dir_path") + 1])
        (output / "outputs" / "run").mkdir(parents=True)
        (output / "outputs" / "run" / "episode_results_rank0.jsonl").write_text('{"success": false}\n')
        (output / run_tests.GPU_RECORDING).touch()
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(run_tests.subprocess, "Popen", FakeServer)
    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert run_tests._run_policy_runner_smoke(_smoke("g1_trocar"), timeout=30)
    assert events == ["server started: i4h_tasks.gr00t_n15.serve assemble_trocar", "rollout", "server stopped"]


def test_policy_runner_smoke_requires_an_episode_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_tests.subprocess, "run", lambda argv, **_kw: subprocess.CompletedProcess(argv, 0, stdout="")
    )
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert not run_tests._run_policy_runner_smoke(_smoke("psm_needle"), timeout=30)


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
    monkeypatch.setattr(run_tests, "_run_policy_runner_smoke", lambda *_: True)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(run_tests, "_run_command", run)
    assert run_tests.main() == 1
    assert len(calls) == 1
    assert "scripts/validate_dependency_pins.py" in calls[0][0]
    assert calls[0][1]["timeout"] == 90


def test_the_endoluminal_smoke_runs_the_catheter_on_newton(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        output = Path(argv[argv.index("--recorder_dataset_export_dir_path") + 1])
        (output / "outputs" / "run").mkdir(parents=True)
        (output / "outputs" / "run" / "episode_results_rank0.jsonl").write_text('{"success": null}\n')
        (output / run_tests.GPU_RECORDING).touch()
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(run_tests.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    assert run_tests._run_policy_runner_smoke(_smoke("endoluminal_navigation"), timeout=30)
    command = calls[0]
    # Launcher options go before the environment, its own options after it.
    assert command.index("--presets") < command.index("endoluminal_navigation") < command.index("--episode_steps")
