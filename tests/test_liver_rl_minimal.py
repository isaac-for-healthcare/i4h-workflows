# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The minimal liver RL example driver plans, resumes and fails without a simulator."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("liver_rl_minimal", ROOT / "scripts" / "liver_rl_minimal.py")
driver = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = driver
_spec.loader.exec_module(driver)

STAGES = [
    "evaluation-starts",
    "snapshots-8",
    "snapshots-1",
    "demonstrations",
    "behaviour-cloning",
    "import",
    "evaluate-bc",
    "ppo",
    "select-ppo",
    "export",
    "evaluate-ppo",
    "run",
]


def _plan(tmp_path, *argv):
    return driver.stages(driver._parser().parse_args(list(argv)), tmp_path)


def test_stages_run_in_order_inside_one_run_directory(tmp_path):
    plan = _plan(tmp_path, "--headless")
    assert [stage.name for stage in plan] == STAGES
    for stage in plan:
        assert stage.outputs and all(tmp_path in path.parents for path in stage.outputs)
    by_name = {stage.name: stage.command for stage in plan}
    assert "--headless" in by_name["demonstrations"] and "--headless" in by_name["run"]
    # PPO resumes the imported BC actor with the unqualified minimal algorithm and 8 worlds.
    ppo = by_name["ppo"]
    assert ppo[ppo.index("--checkpoint") + 1] == str(tmp_path / "bc/imported/native_bc.pt")
    assert ppo[ppo.index("--num-envs") + 1] == "8"
    assert "agent.algorithm.class_name=i4h_rl.liver_ppo:MinimalLiverPPO" in ppo
    # BC and PPO are scored on the same starts; only the checkpoint and output differ.
    bc, ppo = by_name["evaluate-bc"], by_name["evaluate-ppo"]
    differs = [i for i, (a, b) in enumerate(zip(bc, ppo, strict=True)) if a != b]
    assert [bc[i - 1] for i in differs] == ["--checkpoint", "--output"]
    assert bc[bc.index("--episodes") + 1] == "40" and bc[bc.index("--starts") + 1] == str(tmp_path / "starts/test.json")
    run = by_name["run"]
    assert run[1:3] == ["liver_reach_retract", "--policy"]
    assert run[run.index("--checkpoint") + 1] == str(tmp_path / "export")


def test_environment_selects_run_snapshots_and_full_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("I4H_LIVER_OBJECTIVE", "reach_only")
    env = driver._environment(tmp_path)
    assert env["I4H_LIVER_SNAPSHOTS"] == str(tmp_path / "snapshots")
    assert env["I4H_LIVER_OBJECTIVE"] == "full_cycle"
    assert str(ROOT / "rl") in env["PYTHONPATH"].split(":")


def test_dry_run_prints_every_stage_without_running(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert driver.main(["--dry-run", "--run-dir", str(tmp_path / "run"), "--from-stage", "ppo"]) == 0
    out = capsys.readouterr().out
    assert out.count("[skip]") == STAGES.index("ppo") and out.count("[run ]") == len(STAGES) - STAGES.index("ppo")
    assert not (tmp_path / "run").exists()


def test_missing_numerical_source_stops_before_any_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver, "_missing_dependencies", lambda: ["Numerical Sim source /absent (private)"])
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert driver.main(["--run-dir", str(tmp_path / "run")]) == 2
    assert "Numerical Sim source" in capsys.readouterr().err


def test_resume_requires_earlier_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver, "_missing_dependencies", lambda: [])
    assert driver.main(["--run-dir", str(tmp_path), "--from-stage", "import"]) == 2
    assert "evaluation-starts outputs are missing" in capsys.readouterr().err


class _Process:
    """A stage process that exits with ``returncode`` or, when None, never exits on its own."""

    def __init__(self, returncode, write=None):
        self.returncode = returncode
        self.signals = []
        if write is not None:
            write.parent.mkdir(parents=True, exist_ok=True)
            write.write_bytes(b"complete")

    def poll(self):
        return self.returncode if not self.signals or self.returncode is not None else -15

    def terminate(self):
        self.signals.append("TERM")

    def kill(self):
        self.signals.append("KILL")

    def wait(self, timeout=None):
        return -15


@pytest.fixture(autouse=True)
def fake_process_cleanup(monkeypatch):
    original = driver._stop_process_group

    def stop(process):
        if isinstance(process, _Process):
            process.terminate()
        else:
            original(process)

    monkeypatch.setattr(driver, "_stop_process_group", stop)


def test_failed_stage_records_summary_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_missing_dependencies", lambda: [])
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        return _Process(1)

    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen)
    assert driver.main(["--run-dir", str(tmp_path)]) == 1
    assert len(calls) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    stage = summary["stages"][0]
    assert stage["name"] == "evaluation-starts" and stage["returncode"] == 1 and stage["forced_exit"] is False
    assert summary["evaluation"] == {"bc": None, "ppo": None}


def test_stuck_shutdown_after_complete_outputs_is_ended_and_recorded(tmp_path, monkeypatch):
    stage = driver.Stage("snapshots-8", "settle", ["sim"], [tmp_path / "settled_8.npz"], outputs_before_shutdown=True)
    process = None

    def fake_popen(command, **kwargs):
        nonlocal process
        process = _Process(None, write=tmp_path / "settled_8.npz")
        return process

    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    clock = iter(range(0, 10_000, 50))
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(clock))
    assert driver._execute(stage, {}, tmp_path / "stage.log", exit_grace=120) == (0, True)
    assert process.signals == ["TERM"]
    assert "ending shutdown" in (tmp_path / "stage.log").read_text()


def test_stage_without_early_outputs_is_never_ended(tmp_path, monkeypatch):
    stage = driver.Stage("ppo", "train", ["train"], [tmp_path / "model_final.pt"])
    polls = iter([None, None, 0])

    class Finishing(_Process):
        def poll(self):
            return next(polls)

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: Finishing(None, write=tmp_path / "model_final.pt"))
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    assert driver._execute(stage, {}, tmp_path / "stage.log", exit_grace=0) == (0, False)


def test_rerun_sets_previous_outputs_aside(tmp_path):
    stage = next(s for s in _plan(tmp_path) if s.name == "snapshots-8")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "settled_8.npz").write_bytes(b"old")
    (tmp_path / "snapshots" / "manifest_8.json").write_text("{}")
    moved = driver._set_aside(stage, "stamp")
    assert sorted(Path(p).name for p in moved) == ["manifest_8.json.replaced-stamp", "settled_8.npz.replaced-stamp"]
    assert not (tmp_path / "snapshots" / "settled_8.npz").exists()
    export = next(s for s in _plan(tmp_path) if s.name == "export")
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "policy.pt").write_bytes(b"old")
    assert driver._set_aside(export, "stamp") == [str(tmp_path / "export.replaced-stamp")]


def test_stale_outputs_do_not_end_a_running_stage(tmp_path, monkeypatch):
    output = tmp_path / "bc.json"
    output.write_text("{}")
    old = output.stat().st_mtime - 3600
    import os

    os.utime(output, (old, old))
    stage = driver.Stage("evaluate-bc", "score", ["sim"], [output], outputs_before_shutdown=True)
    polls = iter([None] * 5 + [0])

    class Running(_Process):
        def poll(self):
            return next(polls)

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: Running(None))
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    assert driver._execute(stage, {}, tmp_path / "stage.log", exit_grace=0) == (0, False)


def _completed_plan(tmp_path):
    plan = _plan(tmp_path)
    records = []
    score = dict(episodes=40, successes=20, success_rate=0.5, wilson_95=[0.3, 0.7], failures={})
    for stage in plan:
        for path in stage.outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(score) if stage.name.startswith("evaluate-") else "original")
        records.append(
            dict(
                name=stage.name,
                command=stage.command,
                returncode=0,
                missing_outputs=[],
                outputs={str(p): driver._sha256(p) for p in stage.outputs},
            )
        )
    (tmp_path / "summary.json").write_text(json.dumps(dict(stages=records, evaluation=dict(bc=score, ppo=score))))
    return plan


@pytest.mark.parametrize("change", ["checkpoint", "contract", "command", "missing_record"])
def test_resume_rejects_unverified_inputs(tmp_path, monkeypatch, capsys, change):
    _completed_plan(tmp_path)
    monkeypatch.setattr(driver, "_missing_dependencies", lambda: [])
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: pytest.fail("started an unverified run"))
    argv = ["--run-dir", str(tmp_path), "--from-stage", "ppo"]
    if change in ("checkpoint", "contract"):
        name = "native_bc.pt" if change == "checkpoint" else "contract.json"
        (tmp_path / "bc/imported" / name).write_text("replaced")
    elif change == "command":
        argv += ["--bc-epochs", "999"]
    else:
        (tmp_path / "summary.json").write_text("{}")
    assert driver.main(argv) == 2
    assert "changed or unverified" in capsys.readouterr().err


def test_resume_invalidates_downstream_scores_before_starting(tmp_path, monkeypatch):
    _completed_plan(tmp_path)
    monkeypatch.setattr(driver, "_missing_dependencies", lambda: [])

    def execute(*args):
        saved = json.loads((tmp_path / "summary.json").read_text())
        assert saved["evaluation"]["bc"]["successes"] == 20
        assert saved["evaluation"]["ppo"] is None
        assert "evaluate-ppo" not in [r["name"] for r in saved["stages"]]
        return 1, False

    monkeypatch.setattr(driver, "_execute", execute)
    assert driver.main(["--run-dir", str(tmp_path), "--from-stage", "ppo"]) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["evaluation"]["ppo"] is None


def test_numerical_dependency_error_uses_current_public_source(tmp_path, monkeypatch):
    from i4h_common import paths

    monkeypatch.setattr(paths, "numerical_source", lambda root: tmp_path / "absent")
    missing = " ".join(driver._missing_dependencies())
    assert "isaac-for-healthcare/i4h-physics-simulation repository" in missing
    assert "private" not in missing and "internal" not in missing


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_interrupted_driver_stops_stage_and_grandchild(tmp_path, stop_signal):
    """Exercise real POSIX process groups without a simulator or GPU."""
    pid_file = tmp_path / "pids.json"
    child_code = (
        "import os, subprocess, sys, time, json; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"open({str(pid_file)!r}, 'w').write(json.dumps([os.getpid(), child.pid])); "
        "time.sleep(120)"
    )
    parent_code = (
        "import importlib.util, sys, signal, os; from pathlib import Path; "
        f"s=importlib.util.spec_from_file_location('driver', {str(ROOT / 'scripts/liver_rl_minimal.py')!r}); "
        "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); "
        "signal.signal(signal.SIGTERM, lambda s,f: sys.exit(128+s)); "
        f"m._execute(m.Stage('test', '', [sys.executable, '-c', {child_code!r}]), "
        f"os.environ.copy(), Path({str(tmp_path / 'stage.log')!r}), 120)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pids = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists() and pid_file.read_text():
                pids = json.loads(pid_file.read_text())
                break
            time.sleep(0.02)
        assert len(pids) == 2, "stage did not start"
        parent.send_signal(stop_signal)
        assert parent.wait(timeout=10) != 0
        for pid in pids:
            # A killed orphan can briefly remain a zombie until init reaps it.
            status = Path(f"/proc/{pid}/status")
            if status.exists():
                assert "State:\tZ" in status.read_text()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for pid in pids:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
